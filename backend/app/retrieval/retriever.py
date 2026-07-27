from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.models.api import ChatMessage, RetrievedChunk
from app.retrieval.chunk_quality import (
    broad_quality,
    content_fingerprint,
    experience_quality,
    is_strong_experience_evidence,
    is_weak_broad_evidence,
)
from app.retrieval.diversity import mmr_select
from app.retrieval.hybrid_search import hybrid_score, label_boost, phrase_score
from app.retrieval.label_match import (
    collect_key_value_fields,
    education_research_penalty,
    exact_label_match_score,
    find_exact_label_payloads,
    label_family,
)
from app.retrieval.numeric_facts import numeric_answer_boost
from app.retrieval.query_understanding import (
    EXPERIENCE_QUERY_TYPES,
    PREFERRED_TYPES,
    SUMMARY_QUERY_TYPES,
    QueryUnderstanding,
    understand_query,
)
from app.retrieval.reranker import simple_rerank
from app.retrieval.section_match import overview_penalty, section_heading_boost
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalInspection:
    original_question: str
    normalized_question: str
    resolved_question: str
    query_type: str
    expanded_question: str
    expanded_terms: List[str] = field(default_factory=list)
    subject_name: Optional[str] = None
    initial_candidates: List[RetrievedChunk] = field(default_factory=list)
    reranked_candidates: List[RetrievedChunk] = field(default_factory=list)
    diversity_selected: List[RetrievedChunk] = field(default_factory=list)
    final_chunks: List[RetrievedChunk] = field(default_factory=list)
    detected_key_value_fields: List[Dict[str, Any]] = field(default_factory=list)
    exact_label_matches: List[Dict[str, Any]] = field(default_factory=list)


class Retriever:
    def __init__(
        self,
        *,
        settings: Settings,
        embeddings: EmbeddingProvider,
        store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store

    async def retrieve(
        self,
        *,
        company_id: str,
        question: str,
        top_k: int | None = None,
        history: Sequence[ChatMessage] | None = None,
        subject_name: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], QueryUnderstanding, RetrievalInspection]:
        entities, doc_name, headings = await self._document_context(company_id)
        understanding = understand_query(
            question,
            history=history,
            subject_name=subject_name,
            document_entities=entities,
            document_name=doc_name,
            document_headings=headings,
        )
        # Final evidence size: 3–6. Candidate pool 15–20 before rerank.
        final_k = top_k or min(6, max(5, self.settings.retrieval_top_k))
        final_k = max(3, min(6, final_k))
        candidate_k = max(15, min(20, self.settings.retrieval_candidate_k))

        empty_inspection = RetrievalInspection(
            original_question=understanding.original_question,
            normalized_question=understanding.normalized_question,
            resolved_question=understanding.resolved_question,
            query_type=understanding.query_type,
            expanded_question=understanding.expanded_question,
            expanded_terms=understanding.expanded_terms,
            subject_name=understanding.subject_name,
        )
        if understanding.query_type in {"greeting", "unsupported", "help"}:
            return [], understanding, empty_inspection

        retrieval_query = " ".join(
            [understanding.expanded_question, *understanding.expanded_terms]
        ).strip()
        rerank_question = understanding.resolved_question or understanding.original_question
        label_question = " ".join(
            [
                understanding.original_question,
                understanding.resolved_question,
                understanding.expanded_question,
            ]
        )

        query_vector = await self.embeddings.embed_query(retrieval_query)
        dense_hits = await self.store.search(
            company_id=company_id,
            query_vector=query_vector,
            top_k=candidate_k,
        )

        combined_hits: Dict[str, Dict[str, Any]] = {}
        for hit in dense_hits:
            payload = hit.get("payload") or {}
            key = str(payload.get("content_hash") or payload.get("chunk_id") or id(payload))
            combined_hits[key] = {
                "payload": payload,
                "dense": float(hit.get("score") or 0.0),
                "lexical": 0.0,
                "phrase": 0.0,
                "label_boost": 0.0,
            }

        payloads = await self.store.list_payloads(company_id)
        detected_kv = collect_key_value_fields(payloads)
        exact_payloads = find_exact_label_payloads(
            payloads,
            question=label_question,
            query_type=understanding.query_type,
        )
        exact_match_diag = [
            {
                "label": p.get("label") or p.get("title"),
                "value": p.get("value"),
                "content": p.get("content"),
                "page_number": p.get("page_number"),
                "surrounding_heading": p.get("section_title"),
            }
            for p in exact_payloads
        ]

        for payload in payloads:
            content = str(payload.get("content") or "")
            if not content:
                continue
            lexical = hybrid_score(retrieval_query, content)
            phrase = max(
                phrase_score(understanding.original_question, content),
                phrase_score(rerank_question, content),
            )
            label = label_boost(understanding.query_type, content, payload)
            kv_boost, _family = exact_label_match_score(
                question=label_question,
                query_type=understanding.query_type,
                payload=payload,
            )
            if lexical < 0.04 and phrase < 0.15 and label < 0.12 and kv_boost < 0.35:
                continue
            key = str(payload.get("content_hash") or payload.get("chunk_id") or id(payload))
            existing = combined_hits.get(key)
            if existing:
                existing["lexical"] = max(existing["lexical"], lexical)
                existing["phrase"] = max(existing["phrase"], phrase)
                existing["label_boost"] = max(existing["label_boost"], label)
                existing["kv_boost"] = max(float(existing.get("kv_boost") or 0.0), kv_boost)
            else:
                combined_hits[key] = {
                    "payload": payload,
                    "dense": 0.0,
                    "lexical": lexical,
                    "phrase": phrase,
                    "label_boost": label,
                    "kv_boost": kv_boost,
                }

        # Always include exact label matches even if dense/lexical missed them.
        for payload in exact_payloads:
            key = str(payload.get("content_hash") or payload.get("chunk_id") or id(payload))
            kv_boost, _family = exact_label_match_score(
                question=label_question,
                query_type=understanding.query_type,
                payload=payload,
            )
            if key not in combined_hits:
                combined_hits[key] = {
                    "payload": payload,
                    "dense": 0.0,
                    "lexical": hybrid_score(retrieval_query, str(payload.get("content") or "")),
                    "phrase": 0.0,
                    "label_boost": 0.2,
                    "kv_boost": kv_boost,
                }
            else:
                combined_hits[key]["kv_boost"] = max(
                    float(combined_hits[key].get("kv_boost") or 0.0), kv_boost
                )

        preferred = set(PREFERRED_TYPES.get(understanding.query_type, ["universal"]))
        is_experience = understanding.query_type in EXPERIENCE_QUERY_TYPES
        is_broad = understanding.query_type in SUMMARY_QUERY_TYPES or understanding.query_type in {
            "general",
            "interest",
        }
        scored: List[Dict[str, Any]] = []
        for item in combined_hits.values():
            payload = item["payload"]
            content = str(payload.get("content") or "")
            record_type = str(payload.get("record_type") or "universal")
            dense = item["dense"]
            lexical = item["lexical"] or hybrid_score(retrieval_query, content)
            phrase = item["phrase"] or phrase_score(rerank_question, content)
            label = item["label_boost"] or label_boost(
                understanding.query_type, content, payload
            )
            kv_boost = float(item.get("kv_boost") or 0.0)
            if kv_boost <= 0:
                kv_boost, matched_family = exact_label_match_score(
                    question=label_question,
                    query_type=understanding.query_type,
                    payload=payload,
                )
            else:
                matched_family = None
            type_boost = 0.1 if record_type in preferred else 0.0
            if record_type == "universal":
                type_boost = max(type_boost, 0.05)
            if str(payload.get("content_type") or "") == "key_value":
                type_boost = max(type_boost, 0.08)

            quality_boost = 0.0
            quality_reason = "n/a"
            name_boost = 0.0
            if is_experience:
                quality_boost, quality_reason = experience_quality(content, payload)
                name_boost = 0.0
            elif is_broad:
                quality_boost, quality_reason = broad_quality(content, payload)
                name_boost = 0.0
                if understanding.query_type == "interest":
                    if re_search_interest(content):
                        quality_boost += 0.55
                        quality_reason = "interest_match"
                    else:
                        quality_boost -= 0.15
            else:
                person = str(payload.get("person_name") or "")
                if understanding.subject_name and person:
                    if understanding.subject_name.lower() in person.lower():
                        name_boost = 0.1
                if (
                    understanding.subject_name
                    and understanding.subject_name.lower() in content.lower()
                ):
                    name_boost = max(name_boost, 0.06)

            research_penalty = education_research_penalty(
                content, understanding.query_type, label_question
            )
            quality_boost += research_penalty
            if research_penalty < 0:
                quality_reason = "research_narrative_penalty"

            heading_boost = section_heading_boost(label_question, payload)
            ov_penalty = overview_penalty(
                label_question, understanding.query_type, payload
            )
            num_boost = numeric_answer_boost(label_question, content)
            quality_boost += ov_penalty
            if ov_penalty < 0:
                quality_reason = "overview_downrank"
            if heading_boost > 0:
                quality_reason = "section_heading_match"
            if num_boost > 0:
                quality_reason = "numeric_fact_match"

            if kv_boost >= 0.35:
                quality_reason = f"exact_label_match:{matched_family or 'label'}"

            if is_experience and quality_boost <= -0.7 and kv_boost < 0.35:
                continue
            if is_broad and quality_boost <= -0.7 and kv_boost < 0.35:
                continue

            # Answer-type compatibility: soft boost for policy/list content on policy Qs.
            type_compat = 0.0
            if understanding.query_type in {
                "policy",
                "price",
                "quantity",
                "date",
                "accessibility",
            }:
                ctype = str(payload.get("content_type") or "")
                if ctype in {"list", "paragraph", "key_value"}:
                    type_compat += 0.08
                if ctype == "heading" and heading_boost <= 0:
                    type_compat -= 0.05

            combined = (
                dense * 0.30
                + lexical * 0.18
                + phrase * 0.12
                + label * 0.08
                + heading_boost * 0.55
                + num_boost
                + type_boost
                + name_boost
                + quality_boost
                + kv_boost
                + type_compat
            )
            scored.append(
                {
                    "payload": payload,
                    "dense": dense,
                    "lexical": lexical,
                    "phrase": phrase,
                    "label_boost": label,
                    "heading_boost": heading_boost,
                    "numeric_boost": num_boost,
                    "type_boost": type_boost,
                    "name_boost": name_boost,
                    "quality_boost": quality_boost,
                    "quality_reason": quality_reason,
                    "kv_boost": kv_boost,
                    "combined": combined,
                }
            )

        scored.sort(key=lambda item: item["combined"], reverse=True)

        # Force distinctive query-token chunks into the candidate pool.
        distinctive = [
            token
            for token in re.findall(r"[a-z0-9]{5,}", label_question.lower())
            if token
            not in {
                "about",
                "which",
                "where",
                "their",
                "there",
                "would",
                "could",
                "should",
                "policy",
                "private",
                "events",
                "serve",
                "allowed",
                "items",
                "prohibited",
                "location",
                "accessible",
                "planned",
                "opening",
                "period",
                "membership",
                "volunteer",
                "orientation",
            }
        ]
        # Always keep high-signal policy tokens even if shorter/common filters removed them.
        for token in re.findall(r"[a-z0-9]{4,}", label_question.lower()):
            if token in {
                "alcohol",
                "compost",
                "refund",
                "pets",
                "pet",
                "meat",
                "plastic",
                "wheelchair",
                "founder",
                "ensemble",
                "music",
            }:
                if token not in distinctive:
                    distinctive.append(token)

        if distinctive:
            existing_keys = {
                str(
                    (item["payload"].get("content_hash") or item["payload"].get("chunk_id") or id(item))
                )
                for item in scored
            }
            for payload in payloads:
                content = str(payload.get("content") or "")
                if not content:
                    continue
                lower = content.lower()
                heading = str(payload.get("section_title") or "").lower()
                hits = sum(1 for token in distinctive if token in lower or token in heading)
                if hits <= 0:
                    continue
                key = str(payload.get("content_hash") or payload.get("chunk_id") or id(payload))
                if key in existing_keys:
                    # Boost already-scored matches.
                    for item in scored:
                        p = item["payload"]
                        pk = str(p.get("content_hash") or p.get("chunk_id") or id(p))
                        if pk == key:
                            item["combined"] = float(item["combined"]) + 0.35 * hits
                            item["heading_boost"] = max(
                                float(item.get("heading_boost") or 0.0), 0.25 * hits
                            )
                            break
                    continue
                scored.append(
                    {
                        "payload": payload,
                        "dense": 0.0,
                        "lexical": hybrid_score(retrieval_query, content),
                        "phrase": phrase_score(rerank_question, content),
                        "label_boost": 0.1,
                        "heading_boost": 0.4 * hits,
                        "numeric_boost": numeric_answer_boost(label_question, content),
                        "type_boost": 0.05,
                        "name_boost": 0.0,
                        "quality_boost": 0.2,
                        "quality_reason": "exact_query_token",
                        "kv_boost": 0.0,
                        "combined": 0.5 + 0.35 * hits,
                    }
                )
                existing_keys.add(key)
            scored.sort(key=lambda item: item["combined"], reverse=True)

        initial = scored[:candidate_k]

        # Rerank against the original resolved question; keep exact label boost.
        if self.settings.enable_reranker:
            initial = simple_rerank(rerank_question, initial)
            for item in initial:
                extra = 0.0
                if is_experience or is_broad:
                    extra = float(item.get("quality_boost", 0.0)) * 0.35
                extra += float(item.get("kv_boost") or 0.0) * 0.5
                extra += float(item.get("heading_boost") or 0.0) * 0.4
                extra += float(item.get("numeric_boost") or 0.0) * 0.35
                item["rerank"] = float(item.get("rerank", item["combined"])) + extra

        initial.sort(key=lambda item: item.get("rerank", item["combined"]), reverse=True)

        initial_chunks = [
            _to_retrieved(item, company_id=company_id, preferred=preferred, development=True)
            for item in initial[:candidate_k]
        ]
        reranked_chunks = list(initial_chunks)

        final_items = mmr_select(
            initial,
            limit=final_k,
            prefer_topic_diversity=is_broad or understanding.query_type == "interest",
        )
        diversity_chunks = [
            _to_retrieved(item, company_id=company_id, preferred=preferred, development=True)
            for item in final_items
        ]

        if understanding.query_type in {"education", "major"}:
            # Prefer exact major/degree/minor labels over research narrative.
            label_items = [
                item
                for item in initial
                if float(item.get("kv_boost") or 0.0) >= 0.35
            ]
            if label_items:
                final_items = (label_items + [
                    item for item in final_items if item not in label_items
                ])[:final_k]
        elif understanding.query_type == "school":
            from app.retrieval.institution import content_has_institution_evidence

            school_items = [
                item
                for item in initial
                if content_has_institution_evidence(
                    "\n".join(
                        part
                        for part in (
                            str(item["payload"].get("content") or ""),
                            str(item["payload"].get("organization") or ""),
                            (
                                f"{item['payload'].get('label')}: {item['payload'].get('value')}"
                                if item["payload"].get("label")
                                and item["payload"].get("value")
                                else ""
                            ),
                        )
                        if part
                    )
                )
                or label_family_of(item["payload"]) == "organization"
            ]
            # Exclude pure major key-value lines from school answers.
            school_items = [
                item
                for item in school_items
                if label_family_of(item["payload"]) != "major"
                and not re.match(
                    r"(?i)^(?:nominee'?s?\s*)?major\s*:",
                    str(item["payload"].get("content") or "").strip(),
                )
            ]
            if school_items:
                final_items = school_items[:final_k]
        elif understanding.query_type == "leadership":
            from app.retrieval.fact_types import LEADERSHIP_TITLE_RE

            q = understanding.resolved_question.lower()
            if re.search(r"\b(found|establish|group|organization|club)\b", q):
                lead_items = [
                    item
                    for item in initial
                    if re.search(
                        r"(?i)\b(founder|co-?founder|founded|established)\b",
                        str(item["payload"].get("content") or ""),
                    )
                ]
            else:
                lead_items = [
                    item
                    for item in initial
                    if LEADERSHIP_TITLE_RE.search(str(item["payload"].get("content") or ""))
                ]
            final_items = lead_items[:final_k]
        elif understanding.query_type in {"gpa", "birthday", "favorite_food"}:
            from app.retrieval.fact_types import evidence_matches_fact_type

            typed = [
                item
                for item in initial
                if evidence_matches_fact_type(
                    str(item["payload"].get("content") or ""),
                    understanding.query_type,
                )
            ]
            final_items = typed[:final_k]
        elif understanding.query_type == "identity":
            name_items = [
                item
                for item in initial
                if float(item.get("kv_boost") or 0.0) >= 0.35
                and label_family_of(item["payload"]) in {"name", "email", "major", None}
            ]
            # Always put name key-value fields first for identity questions.
            name_kv = [
                item
                for item in initial
                if label_family_of(item["payload"]) == "name"
                or (
                    str(item["payload"].get("content_type") or "") == "key_value"
                    and re.search(
                        r"(?i)\bname\b",
                        str(item["payload"].get("label") or item["payload"].get("content") or ""),
                    )
                )
            ]
            if name_kv:
                final_items = (name_kv + [
                    item for item in (name_items or final_items) if item not in name_kv
                ])[:final_k]
            elif name_items:
                final_items = (name_items + [
                    item for item in final_items if item not in name_items
                ])[:final_k]
        elif is_experience:
            strong = [
                item
                for item in initial
                if is_strong_experience_evidence(str(item["payload"].get("content") or ""))
            ]
            # Work questions should surface multiple distinct roles when present.
            if strong:
                final_items = mmr_select(
                    strong,
                    limit=final_k,
                    prefer_topic_diversity=True,
                )
            else:
                extras = [
                    item
                    for item in initial
                    if is_strong_experience_evidence(str(item["payload"].get("content") or ""))
                ]
                final_items = (extras + final_items)[:final_k]
                if not any(
                    is_strong_experience_evidence(str(item["payload"].get("content") or ""))
                    for item in final_items
                ):
                    final_items = []
        elif is_broad:
            filtered = [
                item
                for item in final_items
                if not is_weak_broad_evidence(str(item["payload"].get("content") or ""))
                or float(item.get("kv_boost") or 0.0) >= 0.35
            ]
            if filtered:
                final_items = filtered
            if understanding.query_type == "interest":
                interest_hits = [
                    item
                    for item in (filtered or initial)
                    if re_search_interest(str(item["payload"].get("content") or ""))
                ]
                if interest_hits:
                    final_items = mmr_select(
                        interest_hits,
                        limit=final_k,
                        prefer_topic_diversity=False,
                    )
            elif all(
                is_weak_broad_evidence(str(item["payload"].get("content") or ""))
                and float(item.get("kv_boost") or 0.0) < 0.35
                for item in final_items
            ):
                richer = [
                    item
                    for item in initial
                    if not is_weak_broad_evidence(str(item["payload"].get("content") or ""))
                    or float(item.get("kv_boost") or 0.0) >= 0.35
                ]
                final_items = mmr_select(
                    richer or initial,
                    limit=final_k,
                    prefer_topic_diversity=True,
                )
        elif understanding.query_type == "instrument":
            # Only keep evidence that actually mentions an instrument.
            from app.retrieval.label_match import INSTRUMENT_RE

            hits = [
                item
                for item in initial
                if INSTRUMENT_RE.search(str(item["payload"].get("content") or ""))
            ]
            final_items = hits[:final_k]
        elif understanding.query_type in {
            "policy",
            "price",
            "quantity",
            "date",
            "accessibility",
        }:
            # Prefer heading-matched and numeric-compatible section content.
            ranked = sorted(
                initial,
                key=lambda item: (
                    float(item.get("heading_boost") or 0.0)
                    + float(item.get("numeric_boost") or 0.0)
                    + float(item.get("rerank", item.get("combined", 0.0)))
                ),
                reverse=True,
            )
            # Drop overview/profile noise for specific policy facts.
            from app.retrieval.section_match import is_overview_chunk

            filtered = [
                item
                for item in ranked
                if not is_overview_chunk(item["payload"])
                or float(item.get("heading_boost") or 0.0) >= 0.3
            ]
            # Force-include chunks that contain distinctive query tokens.
            forced = _force_token_matches(initial, label_question)
            merged = []
            seen = set()
            for item in forced + (filtered or ranked):
                key = id(item)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
            final_items = merged[:final_k]

        # Deduplicate overview / repeated fingerprints before generation.
        final_items = _dedupe_retrieval_items(final_items, limit=final_k)

        results = [
            _to_retrieved(
                item,
                company_id=company_id,
                preferred=preferred,
                development=self.settings.is_development,
            )
            for item in final_items
        ]
        inspection = RetrievalInspection(
            original_question=understanding.original_question,
            normalized_question=understanding.normalized_question,
            resolved_question=understanding.resolved_question,
            query_type=understanding.query_type,
            expanded_question=understanding.expanded_question,
            expanded_terms=understanding.expanded_terms,
            subject_name=understanding.subject_name,
            initial_candidates=initial_chunks,
            reranked_candidates=reranked_chunks,
            diversity_selected=diversity_chunks,
            final_chunks=results,
            detected_key_value_fields=detected_kv,
            exact_label_matches=exact_match_diag,
        )
        return results, understanding, inspection

    async def _document_context(
        self, company_id: str
    ) -> Tuple[List[str], Optional[str], List[str]]:
        try:
            docs = await self.store.list_documents(company_id)
        except Exception:  # noqa: BLE001
            return [], None, []
        if not docs:
            return [], None, []
        primary = sorted(docs, key=lambda d: d.uploaded_at, reverse=True)[0]
        entities = list(primary.primary_entities or [])
        headings = list(primary.document_headings or [])
        if not entities:
            try:
                payloads = await self.store.list_payloads(company_id)
                entities = _entities_from_payloads(payloads)
            except Exception:  # noqa: BLE001
                pass
        return entities, primary.document_name, headings


def _dedupe_top(scored: Sequence[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in scored:
        content = str(item["payload"].get("content") or "")
        fingerprint = content_fingerprint(content)
        if not fingerprint or fingerprint in seen:
            continue
        # Near-duplicate: if fingerprint is substring of an existing selection, skip.
        if any(fingerprint in existing or existing in fingerprint for existing in seen):
            continue
        seen.add(fingerprint)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _to_retrieved(
    item: Dict[str, Any],
    *,
    company_id: str,
    preferred: set,
    development: bool,
) -> RetrievedChunk:
    payload = item["payload"]
    diagnostics = None
    if development:
        diagnostics = {
            "dense_score": round(float(item.get("dense", 0.0)), 4),
            "lexical_score": round(float(item.get("lexical", 0.0)), 4),
            "semantic_score": round(float(item.get("dense", 0.0)), 4),
            "phrase_score": round(float(item.get("phrase", 0.0)), 4),
            "label_boost": round(float(item.get("label_boost", 0.0)), 4),
            "heading_boost": round(float(item.get("heading_boost", 0.0)), 4),
            "numeric_boost": round(float(item.get("numeric_boost", 0.0)), 4),
            "kv_boost": round(float(item.get("kv_boost", 0.0)), 4),
            "type_boost": round(float(item.get("type_boost", 0.0)), 4),
            "name_boost": round(float(item.get("name_boost", 0.0)), 4),
            "quality_boost": round(float(item.get("quality_boost", 0.0)), 4),
            "quality_reason": item.get("quality_reason"),
            "combined_score": round(float(item.get("combined", 0.0)), 4),
            "reranker_score": round(float(item.get("rerank", item.get("combined", 0.0))), 4),
            "record_type": payload.get("record_type"),
            "content_type": payload.get("content_type"),
            "label": payload.get("label"),
            "value": payload.get("value"),
            "section_title": payload.get("section_title"),
            "subsection_title": payload.get("subsection_title"),
            "filters": {
                "company_id": company_id,
                "preferred_types": sorted(preferred),
                "hard_type_filter": False,
            },
        }
    return RetrievedChunk(
        content=str(payload.get("content") or ""),
        document_name=str(payload.get("document_name") or "document"),
        page_number=payload.get("page_number"),
        source_url=payload.get("source_url"),
        score=float(item.get("rerank", item.get("combined", 0.0))),
        chunk_id=payload.get("chunk_id"),
        section_title=payload.get("section_title"),
        subsection_title=payload.get("subsection_title"),
        record_id=payload.get("record_id"),
        record_type=payload.get("record_type"),
        person_name=payload.get("person_name"),
        title=payload.get("title"),
        organization=payload.get("organization"),
        dates=payload.get("dates"),
        location=payload.get("location"),
        content_type=payload.get("content_type"),
        label=payload.get("label"),
        value=payload.get("value"),
        diagnostics=diagnostics,
    )


def _force_token_matches(
    items: Sequence[Dict[str, Any]], question: str
) -> List[Dict[str, Any]]:
    """Prefer chunks containing distinctive content tokens from the question."""
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]{4,}", (question or "").lower())
        if token
        not in {
            "what",
            "when",
            "where",
            "which",
            "does",
            "have",
            "this",
            "that",
            "with",
            "from",
            "about",
            "policy",
            "private",
            "events",
            "event",
            "serve",
            "allowed",
            "items",
            "much",
            "many",
            "cost",
            "each",
            "year",
            "site",
            "hall",
            "market",
        }
    ]
    if not tokens:
        return []
    matched: List[Dict[str, Any]] = []
    for item in items:
        content = str((item.get("payload") or {}).get("content") or "").lower()
        heading = str((item.get("payload") or {}).get("section_title") or "").lower()
        blob = f"{heading} {content}"
        hits = sum(1 for token in tokens if token in blob)
        if hits >= 1:
            matched.append(item)
    matched.sort(
        key=lambda item: sum(
            1
            for token in tokens
            if token in str((item.get("payload") or {}).get("content") or "").lower()
        ),
        reverse=True,
    )
    return matched[:4]


def _dedupe_retrieval_items(
    items: Sequence[Dict[str, Any]], *, limit: int
) -> List[Dict[str, Any]]:
    """Drop duplicate overview / near-identical chunks before generation."""
    from app.retrieval.section_match import is_overview_chunk

    selected: List[Dict[str, Any]] = []
    seen_fps = set()
    overview_kept = 0
    for item in items:
        payload = item.get("payload") or {}
        content = str(payload.get("content") or "")
        fp = content_fingerprint(content)
        if fp in seen_fps:
            continue
        if is_overview_chunk(payload):
            overview_kept += 1
            if overview_kept > 1:
                continue
        seen_fps.add(fp)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _entities_from_payloads(payloads: Sequence[Dict[str, Any]]) -> List[str]:
    entities: List[str] = []
    seen = set()
    for payload in payloads:
        for key in ("person_name", "organization", "title"):
            value = payload.get(key)
            if not value:
                continue
            text = str(value).strip()
            lower = text.lower()
            if text and lower not in seen and len(text.split()) <= 6:
                seen.add(lower)
                entities.append(text)
    return entities


def re_search_interest(content: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(music|musician|ensemble|band|performance|choir|orchestra)\b",
            content,
        )
    )


def label_family_of(payload: Dict[str, Any]) -> Optional[str]:
    label = str(payload.get("label") or payload.get("title") or "")
    if not label:
        content = re.sub(r"^\[[^\]]+\]\s*", "", str(payload.get("content") or "").strip())
        match = re.match(r"^([^:\n]{1,80}):\s*", content)
        label = match.group(1).strip() if match else ""
    return label_family(label)
