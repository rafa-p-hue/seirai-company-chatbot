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
from app.retrieval.hybrid_search import (
    hybrid_score,
    label_boost,
    phrase_score,
    query_term_scores,
)
from app.retrieval.label_match import (
    collect_key_value_fields,
    education_research_penalty,
    exact_label_match_score,
    find_exact_label_payloads,
    label_family,
)
from app.ingestion.document_status import version_rank_boost
from app.ingestion.service_domain import domain_match_score, title_match_boost
from app.retrieval.numeric_facts import numeric_answer_boost
from app.retrieval.procedure_context import (
    ProcedureContext,
    continuity_score,
    is_elliptical_followup,
    reject_insurance_evidence_for_move_in,
    MOVE_IN_HEADING_RE,
    INSURANCE_CONFLICT_RE,
)
from app.retrieval.query_understanding import (
    EXPERIENCE_QUERY_TYPES,
    PREFERRED_TYPES,
    SUMMARY_QUERY_TYPES,
    QueryUnderstanding,
    understand_query,
)
from app.retrieval.reranker import simple_rerank
from app.retrieval.section_match import (
    overview_penalty,
    procedural_noise_penalty,
    section_heading_boost,
    summary_section_boost,
)
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
    extracted_table_rows: List[Dict[str, Any]] = field(default_factory=list)
    table_chunks: List[RetrievedChunk] = field(default_factory=list)
    context_question: Optional[str] = None
    active_section: Optional[str] = None
    service_domain: Optional[str] = None
    procedure_context: Optional[Dict[str, Any]] = None
    continuity_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    rejected_cross_domain: List[Dict[str, Any]] = field(default_factory=list)
    candidates_before_rerank: List[RetrievedChunk] = field(default_factory=list)
    candidates_after_rerank: List[RetrievedChunk] = field(default_factory=list)
    checklist_assembly: Dict[str, Any] = field(default_factory=dict)


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
        session_id: Optional[str] = None,
        include_company_docs: bool = True,
        service_domain: Optional[str] = None,
        domain_source_question: Optional[str] = None,
        procedure_context: Optional[ProcedureContext] = None,
    ) -> Tuple[List[RetrievedChunk], QueryUnderstanding, RetrievalInspection]:
        entities, doc_name, headings = await self._document_context(
            company_id,
            session_id=session_id,
            include_company_docs=include_company_docs,
        )
        understanding = understand_query(
            question,
            history=history,
            subject_name=subject_name,
            document_entities=entities,
            document_name=doc_name,
            document_headings=headings,
            service_domain=service_domain
            or (procedure_context.active_domain if procedure_context else None),
            domain_source_question=domain_source_question,
        )
        if procedure_context and procedure_context.active_domain != "general":
            understanding.service_domain = procedure_context.active_domain
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
            context_question=understanding.context_question,
            service_domain=understanding.service_domain,
            procedure_context=procedure_context.to_diagnostics()
            if procedure_context
            else None,
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
                domain_source_question or "",
            ]
        ).strip()

        query_vector = await self.embeddings.embed_query(retrieval_query)
        dense_hits = await self.store.search(
            company_id=company_id,
            query_vector=query_vector,
            top_k=candidate_k,
            session_id=session_id,
            include_company_docs=include_company_docs,
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

        payloads = await self.store.list_payloads(
            company_id,
            session_id=session_id,
            include_company_docs=include_company_docs,
        )
        active_section = _infer_active_section(
            understanding.context_question, payloads
        )
        if procedure_context and procedure_context.active_section:
            active_section = procedure_context.active_section
        extracted_table_rows = [
            dict(payload.get("table_data") or {})
            for payload in payloads
            if payload.get("table_data")
            and str(payload.get("record_type") or "") == "universal"
        ]
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
        rejected_cross_domain: List[Dict[str, Any]] = []
        continuity_diagnostics: List[Dict[str, Any]] = []
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
            noise_penalty = procedural_noise_penalty(
                label_question, understanding.query_type, payload
            )
            summary_boost = (
                summary_section_boost(payload)
                if understanding.query_type in SUMMARY_QUERY_TYPES
                else 0.0
            )
            action_boost = _action_phrase_boost(label_question, content, payload)
            version_boost = version_rank_boost(
                question=label_question,
                document_status=str(payload.get("document_status") or "unknown"),
                effective_date=payload.get("effective_date"),
            )
            domain_boost = domain_match_score(
                understanding.service_domain,
                payload.get("service_domain"),
                document_name=str(payload.get("document_name") or ""),
                section_title=str(payload.get("section_title") or ""),
                content=content,
            )
            doc_title_boost = title_match_boost(
                label_question, str(payload.get("document_name") or "")
            )
            continuity_boost = 0.0
            cross_domain_penalty = 0.0
            continuity_reason = "n/a"
            domain_source = domain_source_question or understanding.original_question
            if procedure_context is not None:
                continuity_boost, cross_domain_penalty, continuity_reason = (
                    continuity_score(
                        context=procedure_context,
                        payload=payload,
                        content=content,
                    )
                )
                # Locked single-procedure questions: drop conflicting-domain bleed.
                if (
                    procedure_context.locked
                    and not procedure_context.allow_domain_switch
                    and cross_domain_penalty >= 3.0
                ):
                    rejected_cross_domain.append(
                        {
                            "document_name": payload.get("document_name"),
                            "section_title": payload.get("section_title"),
                            "service_domain": payload.get("service_domain"),
                            "continuity_reason": continuity_reason,
                            "cross_domain_penalty": cross_domain_penalty,
                            "content_preview": content[:160],
                            "rejection_stage": "continuity_hard_reject",
                        }
                    )
                    continue
            # Hard generic rule: move-in/address registration rejects insurance-dominated evidence.
            if reject_insurance_evidence_for_move_in(
                question=domain_source,
                content=content,
                section_title=str(payload.get("section_title") or ""),
                service_domain=str(payload.get("service_domain") or ""),
            ):
                rejected_cross_domain.append(
                    {
                        "document_name": payload.get("document_name"),
                        "section_title": payload.get("section_title"),
                        "service_domain": payload.get("service_domain"),
                        "continuity_reason": "move_in_rejects_insurance_evidence",
                        "cross_domain_penalty": 4.0,
                        "content_preview": content[:160],
                        "rejection_stage": "move_in_hard_validation",
                    }
                )
                continue
            # Strong procedure heading boost / insurance conflict lexical penalty.
            section_blob = f"{payload.get('section_title') or ''}\n{payload.get('document_name') or ''}"
            if MOVE_IN_HEADING_RE.search(section_blob) and re.search(
                r"(?i)\b(moved?|moving|register|registration|bring|new address)\b",
                domain_source,
            ):
                heading_boost = max(heading_boost, heading_boost + 0.85)
                continuity_boost += 0.45
            if INSURANCE_CONFLICT_RE.search(content) or INSURANCE_CONFLICT_RE.search(
                section_blob
            ):
                if re.search(
                    r"(?i)\b(moved?|moving|register|registration|new address)\b",
                    domain_source,
                ) and not re.search(
                    r"(?i)\b(employer|insurance|nhi|enroll|premium)\b", domain_source
                ):
                    cross_domain_penalty += 2.4
                    continuity_reason = (
                        f"{continuity_reason}+insurance_term_penalty"
                        if continuity_reason != "n/a"
                        else "insurance_term_penalty"
                    )
            num_boost = numeric_answer_boost(label_question, content)
            term_scores = query_term_scores(understanding.original_question, content)
            exact_term_boost = 0.0
            if understanding.query_type == "price" and num_boost > 0 and term_scores:
                exact_term_boost = max(term_scores.values()) * 0.55
            conv_boost = _conversation_context_boost(history, content)
            entity_boost = _entity_match_boost(understanding, payload, content)
            active_section_boost = _active_section_boost(
                active_section,
                payload,
                procedural=understanding.query_type
                in {"checklist", "date", "policy"},
            )
            requirement_boost = (
                _checklist_requirement_score(payload)
                if understanding.query_type in {"checklist", "date"}
                else 0.0
            )
            quality_boost += ov_penalty + noise_penalty + summary_boost
            if ov_penalty < 0:
                quality_reason = "overview_downrank"
            if noise_penalty < 0:
                quality_reason = "procedural_noise_downrank"
            if summary_boost > 0:
                quality_reason = "summary_section_coverage"
            if heading_boost > 0:
                quality_reason = "section_heading_match"
            if num_boost > 0:
                quality_reason = "numeric_fact_match"
            if action_boost > 0:
                quality_reason = "action_phrase_match"
            if conv_boost > 0:
                quality_reason = "conversation_context_match"

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
                "checklist",
            }:
                ctype = str(payload.get("content_type") or "")
                if ctype in {"list", "numbered_list", "paragraph", "key_value", "table"}:
                    type_compat += 0.08
                if understanding.query_type == "price" and ctype in {
                    "table",
                    "structured_table_row",
                }:
                    type_compat += 0.28
                if understanding.query_type == "checklist" and ctype in {
                    "list",
                    "numbered_list",
                }:
                    type_compat += 0.35
                # Heading-only chunks are metadata anchors — never prefer them as answers.
                if ctype == "heading":
                    type_compat -= 0.55
                    if len(content.split()) <= 6:
                        quality_boost -= 0.35
                        quality_reason = "heading_anchor_only"

            # Soft duplicate penalty for near-identical fingerprints already scored higher.
            dup_penalty = 0.0

            # Semantic similarity alone must not override a procedure/domain mismatch.
            dense_weight = 0.28
            if domain_boost <= -1.0:
                dense_weight = 0.08
            elif domain_boost < 0:
                dense_weight = 0.16

            combined = (
                dense * dense_weight
                + lexical * 0.16
                + phrase * 0.12
                + label * 0.08
                + heading_boost * 0.55
                + num_boost
                + exact_term_boost
                + conv_boost
                + entity_boost
                + active_section_boost
                + requirement_boost
                + action_boost
                + version_boost
                + domain_boost
                + doc_title_boost
                + continuity_boost
                - cross_domain_penalty
                + type_boost
                + name_boost
                + quality_boost
                + kv_boost
                + type_compat
                + dup_penalty
            )
            if domain_boost >= 1.0:
                quality_reason = "service_domain_match"
            elif domain_boost <= -1.0:
                quality_reason = "service_domain_conflict"
            if continuity_boost > 0:
                quality_reason = f"procedure_continuity:{continuity_reason}"
            if cross_domain_penalty >= 3.0:
                quality_reason = f"cross_domain_reject:{continuity_reason}"
            if version_boost:
                quality_reason = (
                    "prefer_current_document"
                    if version_boost > 0
                    else "downrank_archived_document"
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
                    "exact_term_boost": exact_term_boost,
                    "query_term_scores": term_scores,
                    "conversation_boost": conv_boost,
                    "entity_boost": entity_boost,
                    "active_section_boost": active_section_boost,
                    "requirement_boost": requirement_boost,
                    "action_boost": action_boost,
                    "version_boost": version_boost,
                    "domain_boost": domain_boost,
                    "title_boost": doc_title_boost,
                    "continuity_boost": continuity_boost,
                    "cross_domain_penalty": -cross_domain_penalty,
                    "continuity_reason": continuity_reason,
                    "type_boost": type_boost,
                    "name_boost": name_boost,
                    "quality_boost": quality_boost,
                    "quality_reason": quality_reason,
                    "kv_boost": kv_boost,
                    "combined": combined,
                }
            )
            if procedure_context is not None:
                continuity_diagnostics.append(
                    {
                        "document_name": payload.get("document_name"),
                        "section_title": payload.get("section_title"),
                        "service_domain": payload.get("service_domain"),
                        "continuity_score": round(continuity_boost - cross_domain_penalty, 4),
                        "continuity_boost": round(continuity_boost, 4),
                        "cross_domain_penalty": round(cross_domain_penalty, 4),
                        "continuity_reason": continuity_reason,
                        "final_score": round(combined, 4),
                    }
                )

        # Apply duplicate penalties against higher-ranked near-identical content.
        seen_fps: Dict[str, int] = {}
        for item in sorted(scored, key=lambda row: row["combined"], reverse=True):
            fp = content_fingerprint(str(item["payload"].get("content") or ""))
            count = seen_fps.get(fp, 0)
            if count:
                item["combined"] -= 0.25 * count
                item["duplicate_penalty"] = -0.25 * count
            seen_fps[fp] = count + 1
        scored.sort(key=lambda item: item["combined"], reverse=True)

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
                # Never re-inject cross-domain / insurance-dominated chunks via token match.
                if procedure_context is not None:
                    _b, penalty, reason = continuity_score(
                        context=procedure_context, payload=payload, content=content
                    )
                    if (
                        procedure_context.locked
                        and not procedure_context.allow_domain_switch
                        and penalty >= 3.0
                    ):
                        rejected_cross_domain.append(
                            {
                                "document_name": payload.get("document_name"),
                                "section_title": payload.get("section_title"),
                                "service_domain": payload.get("service_domain"),
                                "continuity_reason": f"exact_token_blocked:{reason}",
                                "cross_domain_penalty": penalty,
                                "content_preview": content[:160],
                                "rejection_stage": "exact_query_token_blocked",
                            }
                        )
                        continue
                if reject_insurance_evidence_for_move_in(
                    question=domain_source_question or understanding.original_question,
                    content=content,
                    section_title=str(payload.get("section_title") or ""),
                    service_domain=str(payload.get("service_domain") or ""),
                ):
                    rejected_cross_domain.append(
                        {
                            "document_name": payload.get("document_name"),
                            "section_title": payload.get("section_title"),
                            "service_domain": payload.get("service_domain"),
                            "continuity_reason": "exact_token_blocked:move_in_rejects_insurance",
                            "cross_domain_penalty": 4.0,
                            "content_preview": content[:160],
                            "rejection_stage": "exact_query_token_blocked",
                        }
                    )
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
                domain_boost = domain_match_score(
                    understanding.service_domain,
                    payload.get("service_domain"),
                    document_name=str(payload.get("document_name") or ""),
                    section_title=str(payload.get("section_title") or ""),
                    content=content,
                )
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
                        "domain_boost": domain_boost,
                        "continuity_boost": 0.0,
                        "cross_domain_penalty": 0.0,
                        "combined": 0.5 + 0.35 * hits + max(0.0, domain_boost),
                    }
                )
                existing_keys.add(key)
            scored.sort(key=lambda item: item["combined"], reverse=True)

        if understanding.query_type in {"checklist", "date", "policy"}:
            scored = _inject_procedural_lexical_fallback(
                scored,
                payloads=payloads,
                question=label_question,
                retrieval_query=retrieval_query,
                rerank_question=rerank_question,
                active_section=active_section,
                service_domain=understanding.service_domain,
                procedure_context=procedure_context,
            )

        initial = scored[:candidate_k]
        candidates_before_rerank = [
            _to_retrieved(item, company_id=company_id, preferred=preferred, development=True)
            for item in initial
        ]

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
                extra += float(item.get("domain_boost") or 0.0) * 0.85
                extra += float(item.get("version_boost") or 0.0) * 0.5
                extra += float(item.get("continuity_boost") or 0.0) * 0.9
                extra += float(item.get("cross_domain_penalty") or 0.0) * 0.9
                item["rerank"] = float(item.get("rerank", item["combined"])) + extra

        initial.sort(key=lambda item: item.get("rerank", item["combined"]), reverse=True)
        initial = _prefer_matching_domain(initial, understanding.service_domain)
        candidates_after_rerank = [
            _to_retrieved(item, company_id=company_id, preferred=preferred, development=True)
            for item in initial
        ]

        initial_chunks = [
            _to_retrieved(item, company_id=company_id, preferred=preferred, development=True)
            for item in initial[:candidate_k]
        ]
        reranked_chunks = list(candidates_after_rerank)

        final_items = mmr_select(
            initial,
            limit=final_k,
            prefer_topic_diversity=is_broad or understanding.query_type == "interest",
        )
        checklist_assembly_diag: Dict[str, Any] = {}
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
            final_items = (lead_items or final_items)[:final_k]
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
                # Soft guide: keep ranked items rather than wiping evidence entirely.
                final_items = final_items[:final_k]
        elif is_broad:
            filtered = [
                item
                for item in final_items
                if not is_weak_broad_evidence(str(item["payload"].get("content") or ""))
                or float(item.get("kv_boost") or 0.0) >= 0.35
                or str(item["payload"].get("content_type") or "") == "heading"
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
            elif understanding.query_type in SUMMARY_QUERY_TYPES:
                # Cover purpose + major sections via MMR; never a single fragment.
                diverse_pool = [
                    item
                    for item in initial
                    if not is_weak_broad_evidence(
                        str(item["payload"].get("content") or "")
                    )
                    or str(item["payload"].get("content_type") or "") == "heading"
                    or float(item.get("quality_boost") or 0.0) > 0
                ]
                final_items = mmr_select(
                    diverse_pool or initial,
                    limit=max(final_k, 5),
                    prefer_topic_diversity=True,
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
        elif understanding.query_type == "checklist":
            requirement_items = [
                item
                for item in initial
                if _is_requirement_payload(item["payload"])
            ]
            # Same-section / same-document follow-up path for elliptical bring/go.
            anchor_doc = (
                procedure_context.active_document if procedure_context else None
            )
            anchor_section = active_section or (
                procedure_context.active_section if procedure_context else None
            )
            same_doc = [
                item
                for item in (requirement_items or initial)
                if anchor_doc
                and str(item["payload"].get("document_name") or "").lower()
                == anchor_doc.lower()
                and str(item["payload"].get("content_type") or "") != "heading"
            ]
            same_section = [
                item
                for item in (same_doc or requirement_items)
                if anchor_section
                and str(item["payload"].get("section_title") or "").strip().lower()
                == anchor_section.strip().lower()
            ]
            if is_elliptical_followup(understanding.original_question) or (
                procedure_context and procedure_context.active_document
            ):
                # Prefer nearby/same-section chunks before broad corpus evidence.
                final_items = (same_section or same_doc or requirement_items)[:final_k]
            else:
                final_items = (same_section or requirement_items)[:final_k]
            if not final_items:
                section_pool = [
                    item
                    for item in initial
                    if anchor_section
                    and str(item["payload"].get("section_title") or "")
                    .strip()
                    .lower()
                    == anchor_section.strip().lower()
                    and str(item["payload"].get("content_type") or "") != "heading"
                ]
                procedural_pool = [
                    item
                    for item in (section_pool or same_doc or initial)
                    if REQUIREMENT_LANGUAGE_RE.search(
                        f"{item['payload'].get('section_title') or ''}\n"
                        f"{item['payload'].get('content') or ''}"
                    )
                ]
                final_items = (procedural_pool or section_pool or same_doc or initial)[
                    :final_k
                ]
            # Pull additional same-document requirement prose so checklists stay complete.
            if anchor_doc:
                extras = [
                    item
                    for item in initial
                    if str(item["payload"].get("document_name") or "").lower()
                    == anchor_doc.lower()
                    and item not in final_items
                    and REQUIREMENT_LANGUAGE_RE.search(
                        f"{item['payload'].get('section_title') or ''}\n"
                        f"{item['payload'].get('content') or ''}"
                    )
                ]
                merged = list(final_items)
                for item in extras:
                    merged.append(item)
                    if len(merged) >= final_k:
                        break
                final_items = merged[:final_k]
            # Load same-section / adjacent requirement siblings so trailing
            # checklist bullets are not dropped after the first fragment.
            from app.retrieval.checklist_assembly import expand_checklist_siblings

            final_items, checklist_assembly_diag = expand_checklist_siblings(
                final_items,
                payloads,
                limit=max(final_k, 8),
            )
            if checklist_assembly_diag.get("siblings_added"):
                logger.info(
                    "checklist_assembly siblings=%s",
                    checklist_assembly_diag.get("siblings_added"),
                )
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
                    float(item.get("domain_boost") or 0.0)
                    + float(item.get("heading_boost") or 0.0)
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
                if (
                    not is_overview_chunk(item["payload"])
                    or float(item.get("heading_boost") or 0.0) >= 0.3
                )
                and str(item["payload"].get("content_type") or "") != "heading"
            ]
            # If filtering removed everything, fall back to non-overview ranked items.
            if not filtered:
                filtered = [
                    item
                    for item in ranked
                    if str(item["payload"].get("content_type") or "") != "heading"
                ] or ranked
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
        final_items = _finalize_domain_evidence(
            final_items,
            pool=initial,
            service_domain=understanding.service_domain,
            limit=final_k,
        )

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
            extracted_table_rows=extracted_table_rows,
            table_chunks=[
                chunk
                for chunk in initial_chunks
                if chunk.content_type in {"table", "structured_table_row"}
            ],
            context_question=understanding.context_question,
            active_section=active_section,
            service_domain=understanding.service_domain,
            procedure_context=procedure_context.to_diagnostics()
            if procedure_context
            else None,
            continuity_diagnostics=sorted(
                continuity_diagnostics,
                key=lambda row: float(row.get("final_score") or 0.0),
                reverse=True,
            )[:20],
            rejected_cross_domain=rejected_cross_domain[:20],
            candidates_before_rerank=candidates_before_rerank[:20],
            candidates_after_rerank=candidates_after_rerank[:20],
            checklist_assembly=checklist_assembly_diag,
        )
        return results, understanding, inspection

    async def _document_context(
        self,
        company_id: str,
        *,
        session_id: Optional[str] = None,
        include_company_docs: bool = True,
    ) -> Tuple[List[str], Optional[str], List[str]]:
        try:
            docs = await self.store.list_documents(
                company_id,
                session_id=session_id,
                document_scope=None,
                include_company_docs=include_company_docs,
            )
        except Exception:  # noqa: BLE001
            return [], None, []
        if not docs:
            return [], None, []
        primary = sorted(docs, key=lambda d: d.uploaded_at, reverse=True)[0]
        entities = list(primary.primary_entities or [])
        headings = list(primary.document_headings or [])
        if not entities:
            try:
                payloads = await self.store.list_payloads(
                    company_id,
                    session_id=session_id,
                    include_company_docs=include_company_docs,
                )
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
            "numeric_currency_score": round(
                float(item.get("numeric_boost", 0.0)), 4
            ),
            "exact_term_boost": round(float(item.get("exact_term_boost", 0.0)), 4),
            "query_term_scores": item.get("query_term_scores") or {},
            "active_section_boost": round(
                float(item.get("active_section_boost", 0.0)), 4
            ),
            "requirement_boost": round(
                float(item.get("requirement_boost", 0.0)), 4
            ),
            "action_boost": round(float(item.get("action_boost", 0.0)), 4),
            "version_boost": round(float(item.get("version_boost", 0.0)), 4),
            "domain_match_score": round(float(item.get("domain_boost", 0.0)), 4),
            "domain_boost": round(float(item.get("domain_boost", 0.0)), 4),
            "title_boost": round(float(item.get("title_boost", 0.0)), 4),
            "continuity_score": round(
                float(item.get("continuity_boost", 0.0))
                + float(item.get("cross_domain_penalty", 0.0)),
                4,
            ),
            "continuity_boost": round(float(item.get("continuity_boost", 0.0)), 4),
            "cross_domain_penalty": round(
                float(item.get("cross_domain_penalty", 0.0)), 4
            ),
            "continuity_reason": item.get("continuity_reason"),
            "archive_penalty": round(
                min(0.0, float(item.get("version_boost", 0.0))), 4
            ),
            "kv_boost": round(float(item.get("kv_boost", 0.0)), 4),
            "type_boost": round(float(item.get("type_boost", 0.0)), 4),
            "name_boost": round(float(item.get("name_boost", 0.0)), 4),
            "quality_boost": round(float(item.get("quality_boost", 0.0)), 4),
            "quality_reason": item.get("quality_reason"),
            "combined_score": round(float(item.get("combined", 0.0)), 4),
            "final_score": round(float(item.get("rerank", item.get("combined", 0.0))), 4),
            "reranker_score": round(float(item.get("rerank", item.get("combined", 0.0))), 4),
            "record_type": payload.get("record_type"),
            "content_type": payload.get("content_type"),
            "label": payload.get("label"),
            "value": payload.get("value"),
            "section_title": payload.get("section_title"),
            "subsection_title": payload.get("subsection_title"),
            "service_domain": payload.get("service_domain"),
            "document_status": payload.get("document_status"),
            "filters": {
                "company_id": company_id,
                "preferred_types": sorted(preferred),
                "hard_type_filter": False,
            },
        }
    source_type = None
    raw_source = payload.get("source_type")
    if raw_source:
        try:
            from app.models.api import SourceType

            source_type = SourceType(raw_source)
        except ValueError:
            source_type = None
    return RetrievedChunk(
        content=str(payload.get("content") or ""),
        document_name=str(payload.get("document_name") or "document"),
        page_number=payload.get("page_number"),
        slide_number=payload.get("slide_number"),
        row_number=payload.get("row_number"),
        source_url=payload.get("source_url"),
        score=float(item.get("rerank", item.get("combined", 0.0))),
        chunk_id=payload.get("chunk_id"),
        document_id=payload.get("document_id"),
        chunk_index=payload.get("chunk_index"),
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
        table_data=payload.get("table_data"),
        file_type=payload.get("file_type"),
        source_type=source_type,
        effective_date=payload.get("effective_date"),
        version=payload.get("version"),
        document_status=payload.get("document_status"),
        service_domain=payload.get("service_domain"),
        diagnostics=diagnostics,
    )


def _prefer_matching_domain(
    items: List[Dict[str, Any]],
    service_domain: Optional[str],
) -> List[Dict[str, Any]]:
    """Keep conflicting domains searchable but rank matching procedures first."""
    from app.ingestion.service_domain import GENERAL

    domain = (service_domain or "").strip()
    if not domain or domain == GENERAL or not items:
        return items

    def _matches(item: Dict[str, Any]) -> bool:
        if float(item.get("domain_boost") or 0.0) >= 1.0:
            return True
        payload = item.get("payload") or {}
        return str(payload.get("service_domain") or "") == domain

    matching = [item for item in items if _matches(item)]
    if not matching:
        return items
    others = [item for item in items if item not in matching]
    return matching + others


def _finalize_domain_evidence(
    final_items: List[Dict[str, Any]],
    *,
    pool: List[Dict[str, Any]],
    service_domain: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Prefer same-domain evidence for generation when the query domain is known."""
    from app.ingestion.service_domain import GENERAL

    domain = (service_domain or "").strip()
    if not domain or domain == GENERAL:
        return final_items[:limit]

    def _matches(item: Dict[str, Any]) -> bool:
        if float(item.get("domain_boost") or 0.0) >= 1.0:
            return True
        payload = item.get("payload") or {}
        return str(payload.get("service_domain") or "") == domain

    matched_final = [item for item in final_items if _matches(item)]
    if len(matched_final) >= min(2, limit):
        return matched_final[:limit]
    matched_pool = [item for item in pool if _matches(item)]
    if matched_pool:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in matched_final + matched_pool:
            key = id(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged
    return final_items[:limit]


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


def _conversation_context_boost(
    history: Sequence[ChatMessage] | None, content: str
) -> float:
    """Soft boost when chunk overlaps recent conversation tokens."""
    if not history:
        return 0.0
    recent = " ".join(
        msg.content for msg in list(history)[-4:] if msg.role in {"user", "assistant"}
    )
    if not recent.strip():
        return 0.0
    from app.retrieval.hybrid_search import tokenize

    q_tokens = tokenize(recent)
    c_tokens = tokenize(content or "")
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    if overlap <= 0:
        return 0.0
    return min(0.35, 0.08 * overlap)


REQUIREMENT_LANGUAGE_RE = re.compile(
    r"(?i)\b("
    r"bring(?: the following| your| a| an| the)?|"
    r"required (?:documents?|items?|materials?)|"
    r"must (?:bring|provide|submit|include|complete)|"
    r"(?:documents?|items?|materials?) (?:are )?(?:needed|required)|"
    r"(?:is|are) required|"
    r"need to (?:bring|provide|submit)|submit the following|"
    r"provide(?: the following| your| a| an| the)?"
    r")\b"
)
MOVE_REGISTER_PHRASE_RE = re.compile(
    r"(?i)\b("
    r"moving\s+in|move-?in(?:\s+notification)?|register(?:ation)?|"
    r"notification|within\s+\d+\s+(?:business\s+)?days?|"
    r"bring|required documents?|submit|service\s+window"
    r")\b"
)
UNRELATED_CHECKLIST_RE = re.compile(
    r"(?i)\b("
    r"office hours?|opening hours?|business hours?|closures?|closed|"
    r"contact information|phone numbers?|email addresses?|directions?"
    r")\b"
)


def _infer_active_section(
    context_question: Optional[str], payloads: Sequence[Dict[str, Any]]
) -> Optional[str]:
    """Infer the prior topic's section from heading+content lexical evidence."""
    if not context_question:
        return None
    section_scores: Dict[str, float] = {}
    section_names: Dict[str, str] = {}
    for payload in payloads:
        heading = str(payload.get("section_title") or "").strip()
        if not heading:
            continue
        content = str(payload.get("content") or "")
        score = max(
            hybrid_score(context_question, heading),
            hybrid_score(context_question, f"{heading} {content}"),
        )
        # Procedural cues in the heading itself are strong section anchors.
        if re.search(
            r"(?i)\b(moving\s+in|move-?in|registration|register|requirements?)\b",
            heading,
        ) and re.search(
            r"(?i)\b(register|moving|move-?in|bring|deadline|within|required)\b",
            context_question,
        ):
            score += 0.35
        key = heading.lower()
        section_scores[key] = max(section_scores.get(key, 0.0), score)
        section_names[key] = heading
    if not section_scores:
        return None
    best_key, best_score = max(section_scores.items(), key=lambda item: item[1])
    return section_names[best_key] if best_score >= 0.12 else None


def _inject_procedural_lexical_fallback(
    scored: List[Dict[str, Any]],
    *,
    payloads: Sequence[Dict[str, Any]],
    question: str,
    retrieval_query: str,
    rerank_question: str,
    active_section: Optional[str],
    service_domain: Optional[str] = None,
    procedure_context: Optional[ProcedureContext] = None,
) -> List[Dict[str, Any]]:
    """Ensure procedure chunks enter the pool via lexical/heading match."""
    if not payloads:
        return scored
    existing = {
        str(item["payload"].get("content_hash") or item["payload"].get("chunk_id") or id(item["payload"]))
        for item in scored
    }
    q = (question or "").lower()
    wants_procedure = bool(
        re.search(
            r"\b(register|registration|moving|move-?in|bring|required documents?|"
            r"notification|within|deadline|submit|enroll|enrollment|apply|"
            r"allowance|garbage|certificate|emergency)\b",
            q,
        )
    )
    if not wants_procedure:
        return scored
    extras: List[Dict[str, Any]] = []
    for payload in payloads:
        content = str(payload.get("content") or "")
        if not content or str(payload.get("content_type") or "") == "heading":
            if str(payload.get("content_type") or "") == "heading" and len(content.split()) <= 6:
                continue
        if procedure_context is not None:
            _boost, penalty, reason = continuity_score(
                context=procedure_context, payload=payload, content=content
            )
            if (
                procedure_context.locked
                and not procedure_context.allow_domain_switch
                and penalty >= 3.0
            ):
                continue
        heading = str(payload.get("section_title") or "")
        blob = f"{heading}\n{content}"
        lexical = hybrid_score(retrieval_query, blob)
        phrase = phrase_score(rerank_question, blob)
        heading_hit = section_heading_boost(question, payload)
        procedural_hit = len(MOVE_REGISTER_PHRASE_RE.findall(blob.lower()))
        if procedural_hit <= 0 and lexical < 0.08 and phrase < 0.12 and heading_hit < 0.2:
            continue
        domain_boost = domain_match_score(
            service_domain,
            payload.get("service_domain"),
            document_name=str(payload.get("document_name") or ""),
            section_title=heading,
            content=content,
        )
        if domain_boost <= -1.0 and procedural_hit <= 0 and heading_hit < 0.35:
            continue
        key = str(payload.get("content_hash") or payload.get("chunk_id") or id(payload))
        section_bonus = 0.0
        if active_section and heading.strip().lower() == active_section.strip().lower():
            section_bonus = 0.6
        move_deadline_bonus = 0.0
        if re.search(r"(?i)\bmove-?in(?:\s+notification)?\b", blob) and re.search(
            r"(?i)\bwithin\s+\d+\s+(?:business\s+)?days?\b", blob
        ):
            move_deadline_bonus = 0.7
        version_boost = version_rank_boost(
            question=question,
            document_status=str(payload.get("document_status") or "unknown"),
            effective_date=payload.get("effective_date"),
        )
        continuity_boost = 0.0
        cross_domain_penalty = 0.0
        continuity_reason = "n/a"
        if procedure_context is not None:
            continuity_boost, cross_domain_penalty, continuity_reason = continuity_score(
                context=procedure_context, payload=payload, content=content
            )
        combined = (
            lexical * 0.45
            + phrase * 0.25
            + heading_hit * 0.55
            + 0.15 * procedural_hit
            + section_bonus
            + move_deadline_bonus
            + domain_boost
            + version_boost
            + continuity_boost
            - cross_domain_penalty
            + 0.35
        )
        if key in existing:
            for item in scored:
                pk = str(
                    item["payload"].get("content_hash")
                    or item["payload"].get("chunk_id")
                    or id(item["payload"])
                )
                if pk == key:
                    item["combined"] = max(float(item.get("combined") or 0.0), combined)
                    item["lexical"] = max(float(item.get("lexical") or 0.0), lexical)
                    item["phrase"] = max(float(item.get("phrase") or 0.0), phrase)
                    item["heading_boost"] = max(
                        float(item.get("heading_boost") or 0.0), heading_hit
                    )
                    item["action_boost"] = max(
                        float(item.get("action_boost") or 0.0),
                        move_deadline_bonus + 0.15 * procedural_hit,
                    )
                    item["domain_boost"] = domain_boost
                    item["version_boost"] = version_boost
                    item["continuity_boost"] = continuity_boost
                    item["cross_domain_penalty"] = -cross_domain_penalty
                    item["continuity_reason"] = continuity_reason
                    item["quality_reason"] = "procedural_lexical_fallback"
                    break
            continue
        extras.append(
            {
                "payload": payload,
                "dense": 0.0,
                "lexical": lexical,
                "phrase": phrase,
                "label_boost": 0.05,
                "heading_boost": heading_hit,
                "numeric_boost": numeric_answer_boost(question, content),
                "exact_term_boost": 0.0,
                "query_term_scores": query_term_scores(question, content),
                "conversation_boost": 0.0,
                "entity_boost": 0.0,
                "active_section_boost": section_bonus,
                "requirement_boost": _checklist_requirement_score(payload),
                "action_boost": move_deadline_bonus + 0.15 * procedural_hit,
                "version_boost": version_boost,
                "domain_boost": domain_boost,
                "continuity_boost": continuity_boost,
                "cross_domain_penalty": -cross_domain_penalty,
                "continuity_reason": continuity_reason,
                "title_boost": title_match_boost(
                    question, str(payload.get("document_name") or "")
                ),
                "type_boost": 0.05,
                "name_boost": 0.0,
                "quality_boost": 0.25,
                "quality_reason": "procedural_lexical_fallback",
                "kv_boost": 0.0,
                "combined": combined,
            }
        )
        existing.add(key)
    if not extras:
        scored.sort(key=lambda item: item["combined"], reverse=True)
        return scored
    merged = list(scored) + extras
    merged.sort(key=lambda item: item["combined"], reverse=True)
    return merged


def _active_section_boost(
    active_section: Optional[str],
    payload: Dict[str, Any],
    *,
    procedural: bool,
) -> float:
    if not active_section or not procedural:
        return 0.0
    heading = str(payload.get("section_title") or "").strip().lower()
    subsection = str(payload.get("subsection_title") or "").strip().lower()
    active = active_section.strip().lower()
    if heading == active:
        return 1.25
    if subsection == active:
        return 0.9
    return -0.45 if heading else -0.15


def _action_phrase_boost(
    question: str, content: str, payload: Dict[str, Any]
) -> float:
    q = (question or "").lower()
    text = f"{payload.get('section_title') or ''}\n{content or ''}".lower()
    boost = 0.0
    if re.search(
        r"\b(register|registration|moving|move-?in|notification|apply|submit|enroll|"
        r"within|days|bring|required documents?)\b",
        q,
    ):
        hits = len(MOVE_REGISTER_PHRASE_RE.findall(text))
        if hits:
            boost += min(0.85, 0.22 * hits)
        if re.search(r"\bmove-?in(?:\s+notification)?\b", text) and re.search(
            r"\bwithin\s+\d+\s+(?:business\s+)?days?\b", text
        ):
            # Complete move-in procedure units outrank title/hours/fee fragments.
            boost += 0.55
        if re.search(r"\b(within|before|after)\b", text):
            boost += 0.2
    if re.search(r"\b(bring|submit|provide|required)\b", q):
        if REQUIREMENT_LANGUAGE_RE.search(text):
            boost += 0.45
    return min(1.35, boost)


def _is_requirement_payload(payload: Dict[str, Any]) -> bool:
    content = str(payload.get("content") or "")
    heading = str(payload.get("section_title") or "")
    subsection = str(payload.get("subsection_title") or "")
    blob = f"{heading}\n{subsection}\n{content}"
    if not REQUIREMENT_LANGUAGE_RE.search(blob):
        return False
    if UNRELATED_CHECKLIST_RE.search(f"{heading}\n{subsection}"):
        return False
    ctype = str(payload.get("content_type") or "")
    # Never treat bare heading anchors as checklist evidence; otherwise accept
    # any body unit with procedural requirement language (including universal).
    if ctype == "heading" and len(content.split()) <= 6:
        return False
    return True


def _checklist_requirement_score(payload: Dict[str, Any]) -> float:
    content = str(payload.get("content") or "")
    heading = str(payload.get("section_title") or "")
    subsection = str(payload.get("subsection_title") or "")
    blob = f"{heading}\n{subsection}\n{content}"
    ctype = str(payload.get("content_type") or "")
    score = 0.0
    if REQUIREMENT_LANGUAGE_RE.search(blob):
        score += 0.9
    elif ctype in {"list", "numbered_list"}:
        score -= 0.85
    if UNRELATED_CHECKLIST_RE.search(blob):
        score -= 0.7
    return score


def _entity_match_boost(
    understanding: QueryUnderstanding, payload: dict, content: str
) -> float:
    """Soft boost for person/org/title entities present in the chunk."""
    boost = 0.0
    subject = (understanding.subject_name or "").strip()
    blob = " ".join(
        part
        for part in (
            content,
            str(payload.get("person_name") or ""),
            str(payload.get("organization") or ""),
            str(payload.get("title") or ""),
        )
        if part
    ).lower()
    if subject and subject.lower() in blob:
        boost += 0.12
    for key in ("organization", "title", "person_name"):
        value = str(payload.get(key) or "").strip()
        if value and value.lower() in (content or "").lower():
            boost += 0.05
    return min(0.25, boost)


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
