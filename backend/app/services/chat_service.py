from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.generation.answer_composer import compose_answer, UNSUPPORTED_STATUS_RE
from app.generation.base import LLMProvider
from app.generation.citations import cited_evidence, sources_from_answer
from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import ChatRequest, ChatResponse, CitationSource, RetrievedChunk
from app.retrieval.fact_types import (
    answer_matches_fact_type,
    detect_fact_type,
    filter_evidence_for_fact,
)
from app.retrieval.multi_question import split_questions
from app.retrieval.query_understanding import (
    EXPERIENCE_QUERY_TYPES,
    SUMMARY_QUERY_TYPES,
    QueryUnderstanding,
    classify_query,
)
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, *, retriever: Retriever, llm: LLMProvider) -> None:
        self.retriever = retriever
        self.llm = llm

    async def chat(self, request: ChatRequest) -> ChatResponse:
        settings = get_settings()
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Conversational intents never trigger document retrieval.
        early_intent = classify_query(request.question)
        if early_intent in {"greeting", "help"}:
            answer, sources = compose_answer(
                understanding=_minimal_understanding(request.question, early_intent),
                evidence=[],
            )
            diagnostics = None
            if settings.is_development:
                diagnostics = {
                    "original_question": request.question,
                    "query_intent": early_intent,
                    "retrieval_skipped": True,
                    "sub_questions": [],
                    "detected_key_value_fields": [],
                    "exact_label_matches": [],
                    "evidence_sent_to_llm": [],
                    "final_chunks_sent_to_llm": [],
                }
            return ChatResponse(
                answer=answer,
                sources=sources,
                conversation_id=conversation_id,
                diagnostics=diagnostics,
            )

        sub_questions = split_questions(request.question)
        if len(sub_questions) > 1:
            return await self._chat_multi(
                request=request,
                sub_questions=sub_questions,
                conversation_id=conversation_id,
            )

        return await self._chat_single(
            request=request,
            question=request.question,
            conversation_id=conversation_id,
            history=request.history,
        )

    async def _chat_multi(
        self,
        *,
        request: ChatRequest,
        sub_questions: Sequence[str],
        conversation_id: str,
    ) -> ChatResponse:
        settings = get_settings()
        parts: List[str] = []
        all_sources: List[CitationSource] = []
        sub_diags: List[Dict[str, Any]] = []
        seen_source_keys = set()
        citation_offset = 0

        for index, question in enumerate(sub_questions, start=1):
            response = await self._chat_single(
                request=request,
                question=question,
                conversation_id=conversation_id,
                history=request.history,
                renumber_citations_from=citation_offset + 1,
            )
            answer = response.answer.strip()
            if len(sub_questions) > 1:
                parts.append(f"{index}. {question}\n{answer}")
            else:
                parts.append(answer)
            for source in response.sources or []:
                key = (source.document_name, source.page_number)
                if key in seen_source_keys:
                    continue
                seen_source_keys.add(key)
                citation_offset += 1
                all_sources.append(
                    source.model_copy(update={"number": citation_offset})
                )
            if settings.is_development and response.diagnostics:
                sub_diags.append(
                    {
                        "question": question,
                        "query_intent": response.diagnostics.get("query_intent"),
                        "fact_type": response.diagnostics.get("fact_type"),
                        "answer": answer,
                        "evidence_sent_to_llm": response.diagnostics.get(
                            "evidence_sent_to_llm"
                        ),
                        "exact_label_matches": response.diagnostics.get(
                            "exact_label_matches"
                        ),
                    }
                )

        diagnostics = None
        if settings.is_development:
            diagnostics = {
                "original_question": request.question,
                "sub_questions": list(sub_questions),
                "sub_question_results": sub_diags,
                "query_intent": "multi",
            }

        return ChatResponse(
            answer="\n\n".join(parts),
            sources=all_sources,
            conversation_id=conversation_id,
            diagnostics=diagnostics,
        )

    async def _chat_single(
        self,
        *,
        request: ChatRequest,
        question: str,
        conversation_id: str,
        history: Sequence | None,
        renumber_citations_from: int = 1,
    ) -> ChatResponse:
        settings = get_settings()
        evidence, understanding, inspection = await self.retriever.retrieve(
            company_id=request.company_id,
            question=question,
            top_k=request.top_k or 5,
            history=history,
        )

        fact_type = detect_fact_type(question, understanding.query_type)
        evidence = filter_evidence_for_fact(
            evidence, fact_type, question=question
        )

        diagnostics = None
        if settings.is_development:
            diagnostics = _build_diagnostics(inspection)
            diagnostics["fact_type"] = fact_type
            diagnostics["evidence_after_fact_filter"] = [
                _chunk_diag(chunk) for chunk in evidence
            ]

        if understanding.query_type == "unsupported":
            return ChatResponse(
                answer=FALLBACK_ANSWER,
                sources=[],
                conversation_id=conversation_id,
                diagnostics=diagnostics,
            )

        if understanding.query_type in EXPERIENCE_QUERY_TYPES and fact_type != "leadership":
            from app.retrieval.chunk_quality import is_strong_experience_evidence

            if not any(is_strong_experience_evidence(item.content) for item in evidence):
                evidence = []

        answer = FALLBACK_ANSWER
        sources: List[CitationSource] = []
        cited: List[RetrievedChunk] = []

        if evidence:
            from app.generation.text_scrub import scrub_internal_metadata

            evidence = [
                item.model_copy(
                    update={"content": scrub_internal_metadata(item.content or "")}
                )
                for item in evidence
                if scrub_internal_metadata(item.content or "").strip()
            ]

        if evidence:
            try:
                answer, _raw_sources = await self.llm.generate(
                    question=question,
                    evidence=evidence,
                    history=history,
                )
                answer = _post_validate_answer(
                    answer, understanding.query_type, fact_type, evidence
                )
                from app.generation.answer_composer import (
                    _finalize_answer,
                    _looks_like_raw_chunk_dump,
                )

                if answer and answer != FALLBACK_ANSWER:
                    answer = _finalize_answer(answer)
                    if _looks_like_raw_chunk_dump(answer, evidence):
                        answer = FALLBACK_ANSWER
                if answer and answer != FALLBACK_ANSWER:
                    sources = sources_from_answer(
                        answer,
                        evidence,
                        query_type=understanding.query_type,
                    )
                    cited = cited_evidence(answer, evidence)
                    if understanding.query_type not in SUMMARY_QUERY_TYPES | {"general"}:
                        limit = 4 if understanding.query_type in {
                            "experience",
                            "leadership",
                            "summary",
                        } else 2
                        sources = sources[:limit]
                        cited = cited[:limit]
            except Exception:  # noqa: BLE001
                logger.exception("LLM generation failed; trying deterministic compose")
                answer, sources = FALLBACK_ANSWER, []
                cited = []

        if answer == FALLBACK_ANSWER:
            answer, sources = compose_answer(
                understanding=understanding, evidence=evidence
            )
            if answer != FALLBACK_ANSWER and not answer_matches_fact_type(
                answer, fact_type, evidence
            ):
                answer, sources = FALLBACK_ANSWER, []
            cited = cited_evidence(answer, evidence) if sources else []

        if answer == FALLBACK_ANSWER:
            sources = []
            cited = []

        if renumber_citations_from != 1 and sources:
            sources = [
                source.model_copy(update={"number": renumber_citations_from + i})
                for i, source in enumerate(sources)
            ]

        if diagnostics is not None:
            diagnostics["evidence_sent_to_llm"] = [
                _chunk_diag(chunk) for chunk in evidence
            ]
            diagnostics["final_chunks_sent_to_llm"] = diagnostics["evidence_sent_to_llm"]
            diagnostics["evidence_actually_cited"] = [
                _chunk_diag(chunk) for chunk in cited
            ]

        logger.info(
            "Chat company=%s type=%s fact=%s evidence=%s fallback=%s",
            request.company_id,
            understanding.query_type,
            fact_type,
            len(evidence),
            answer == FALLBACK_ANSWER,
        )
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            diagnostics=diagnostics,
        )


def _minimal_understanding(question: str, query_type: str) -> QueryUnderstanding:
    return QueryUnderstanding(
        original_question=question,
        normalized_question=question,
        resolved_question=question,
        expanded_question=question,
        query_type=query_type,
        expanded_terms=[],
        subject_name=None,
    )


def _post_validate_answer(
    answer: str,
    query_type: str,
    fact_type: str,
    evidence: Sequence[RetrievedChunk],
) -> str:
    if not answer:
        return FALLBACK_ANSWER
    if "could not find that information" in answer.lower():
        # Prefer explicit unannounced statements over generic fallback.
        from app.generation.answer_composer import _unannounced_from_evidence

        unannounced = _unannounced_from_evidence(evidence)
        if unannounced:
            return unannounced
        return FALLBACK_ANSWER
    if UNSUPPORTED_STATUS_RE.search(answer) and not any(
        UNSUPPORTED_STATUS_RE.search(item.content or "") for item in evidence
    ):
        return FALLBACK_ANSWER
    from app.generation.answer_composer import _looks_like_raw_chunk_dump
    from app.generation.text_scrub import looks_like_internal_metadata, scrub_internal_metadata

    answer = scrub_internal_metadata(answer)
    if looks_like_internal_metadata(answer) or re.search(
        r"(?i)\b(record\s*type|profile\s*description)\s*:", answer
    ):
        return FALLBACK_ANSWER
    if _looks_like_raw_chunk_dump(answer, evidence):
        return FALLBACK_ANSWER
    if not answer_matches_fact_type(answer, fact_type, evidence):
        return FALLBACK_ANSWER
    return answer


def _build_diagnostics(inspection) -> Dict[str, Any]:
    return {
        "original_question": inspection.original_question,
        "normalized_question": inspection.normalized_question,
        "resolved_question": inspection.resolved_question,
        "query_intent": inspection.query_type,
        "expanded_question": inspection.expanded_question,
        "expanded_terms": inspection.expanded_terms,
        "subject_name": inspection.subject_name,
        "detected_key_value_fields": inspection.detected_key_value_fields,
        "exact_label_matches": inspection.exact_label_matches,
        "initial_candidates": [
            _chunk_diag(chunk) for chunk in inspection.initial_candidates
        ],
        "initial_retrieved_chunks": [
            _chunk_diag(chunk) for chunk in inspection.initial_candidates
        ],
        "reranked_candidates": [
            _chunk_diag(chunk) for chunk in inspection.reranked_candidates
        ],
        "diversity_selected_evidence": [
            _chunk_diag(chunk) for chunk in inspection.diversity_selected
        ],
        "evidence_sent_to_llm": [
            _chunk_diag(chunk) for chunk in inspection.final_chunks
        ],
        "final_chunks_sent_to_llm": [
            _chunk_diag(chunk) for chunk in inspection.final_chunks
        ],
        "evidence_actually_cited": [],
    }


def _chunk_diag(chunk: RetrievedChunk) -> Dict[str, Any]:
    diag = chunk.diagnostics or {}
    return {
        "content": chunk.content,
        "document_name": chunk.document_name,
        "page_number": chunk.page_number,
        "record_type": chunk.record_type,
        "content_type": chunk.content_type,
        "label": chunk.label,
        "value": chunk.value,
        "section_title": chunk.section_title,
        "vector_similarity_score": diag.get("dense_score"),
        "semantic_score": diag.get("semantic_score", diag.get("dense_score")),
        "lexical_score": diag.get("lexical_score"),
        "exact_label_match_boost": diag.get("kv_boost"),
        "metadata_boosts": {
            "phrase_score": diag.get("phrase_score"),
            "label_boost": diag.get("label_boost"),
            "kv_boost": diag.get("kv_boost"),
            "type_boost": diag.get("type_boost"),
            "name_boost": diag.get("name_boost"),
            "quality_boost": diag.get("quality_boost"),
            "quality_reason": diag.get("quality_reason"),
        },
        "reranker_score": diag.get("reranker_score"),
        "combined_score": diag.get("combined_score"),
    }
