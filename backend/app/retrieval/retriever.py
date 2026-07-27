from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.models.api import ChatMessage, RetrievedChunk
from app.retrieval.hybrid_search import hybrid_score
from app.retrieval.query_understanding import (
    DEPRIORITIZED_TYPES,
    PREFERRED_TYPES,
    QueryUnderstanding,
    understand_query,
)
from app.retrieval.reranker import simple_rerank
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


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
    ) -> Tuple[List[RetrievedChunk], QueryUnderstanding]:
        understanding = understand_query(
            question, history=history, subject_name=subject_name
        )
        top_k = top_k or min(5, self.settings.retrieval_top_k)

        if understanding.query_type == "greeting":
            return [], understanding
        if understanding.query_type == "unsupported":
            return [], understanding

        candidate_k = max(top_k, min(20, self.settings.retrieval_candidate_k))
        retrieval_query = " ".join(
            [understanding.expanded_question, *understanding.expanded_terms]
        ).strip()
        query_vector = await self.embeddings.embed_query(retrieval_query)
        dense_hits = await self.store.search(
            company_id=company_id,
            query_vector=query_vector,
            top_k=candidate_k,
        )

        scored: List[Dict[str, Any]] = []
        seen_hashes = set()
        preferred = set(PREFERRED_TYPES.get(understanding.query_type, []))
        deprioritized = set(DEPRIORITIZED_TYPES.get(understanding.query_type, []))

        for hit in dense_hits:
            payload = hit.get("payload") or {}
            content = str(payload.get("content") or "")
            content_hash = str(payload.get("content_hash") or content[:64])
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            record_type = str(payload.get("record_type") or "")
            dense = float(hit.get("score") or 0.0)
            lexical = hybrid_score(retrieval_query, content)
            heading_boost = heading_phrase_boost(retrieval_query, payload)
            type_boost = 0.0
            if preferred and record_type in preferred:
                type_boost = 0.35
            elif deprioritized and record_type in deprioritized:
                type_boost = -0.25
            # Education queries: penalize experience that only matches school acronyms
            if understanding.query_type == "education" and record_type in {
                "experience",
                "internship",
            }:
                type_boost -= 0.3
            name_boost = 0.0
            person = str(payload.get("person_name") or "")
            if understanding.subject_name and person:
                if understanding.subject_name.lower() in person.lower():
                    name_boost = 0.15
            combined = (
                dense * 0.55
                + lexical * 0.25
                + heading_boost * 0.1
                + type_boost
                + name_boost
            )
            scored.append(
                {
                    "payload": payload,
                    "dense": dense,
                    "lexical": lexical,
                    "heading_boost": heading_boost,
                    "type_boost": type_boost,
                    "name_boost": name_boost,
                    "combined": combined,
                }
            )

        if preferred:
            preferred_hits = [
                item
                for item in scored
                if str(item["payload"].get("record_type") or "") in preferred
            ]
            if preferred_hits and understanding.query_type in {
                "education",
                "identity",
                "skills",
            }:
                # Keep preferred records first for precision-critical queries.
                other = [item for item in scored if item not in preferred_hits]
                scored = preferred_hits + other

        if self.settings.enable_reranker:
            scored = simple_rerank(retrieval_query, scored)

        scored.sort(key=lambda item: item.get("rerank", item["combined"]), reverse=True)
        results: List[RetrievedChunk] = []
        for item in scored[:top_k]:
            payload = item["payload"]
            diagnostics = None
            if self.settings.is_development:
                diagnostics = {
                    "dense_score": round(item["dense"], 4),
                    "lexical_score": round(item["lexical"], 4),
                    "heading_boost": round(item["heading_boost"], 4),
                    "type_boost": round(item["type_boost"], 4),
                    "name_boost": round(item["name_boost"], 4),
                    "combined_score": round(item["combined"], 4),
                    "reranker_score": round(item.get("rerank", item["combined"]), 4),
                    "record_type": payload.get("record_type"),
                    "title": payload.get("title"),
                    "organization": payload.get("organization"),
                    "filters": {
                        "company_id": company_id,
                        "preferred_types": sorted(preferred),
                    },
                }
            results.append(
                RetrievedChunk(
                    content=str(payload.get("content") or ""),
                    document_name=str(payload.get("document_name") or "document"),
                    page_number=payload.get("page_number"),
                    source_url=payload.get("source_url"),
                    score=float(item.get("rerank", item["combined"])),
                    chunk_id=payload.get("chunk_id"),
                    section_title=payload.get("section_title"),
                    record_id=payload.get("record_id"),
                    record_type=payload.get("record_type"),
                    person_name=payload.get("person_name"),
                    title=payload.get("title"),
                    organization=payload.get("organization"),
                    dates=payload.get("dates"),
                    location=payload.get("location"),
                    diagnostics=diagnostics,
                )
            )
        return results, understanding


def heading_phrase_boost(question: str, payload: Dict[str, Any]) -> float:
    heading = str(payload.get("section_title") or "").lower()
    title = str(payload.get("title") or "").lower()
    organization = str(payload.get("organization") or "").lower()
    content = str(payload.get("content") or "").lower()
    q = question.lower().strip()
    score = 0.0
    tokens = [token for token in q.split() if len(token) > 3]
    haystack = f"{heading} {title} {organization}"
    if heading and any(token in haystack for token in tokens):
        score += 0.45
    if len(q) >= 6 and q in content:
        score += 0.4
    # Exact education keyword boosts
    for term in ("education", "university", "college", "school", "degree", "major"):
        if term in q and term in content:
            score += 0.08
    return min(1.0, score)
