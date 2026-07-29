from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.generation.answer_composer import compose_answer, UNSUPPORTED_STATUS_RE
from app.generation.base import LLMProvider
from app.generation.citations import (
    cited_evidence,
    reconcile_answer_citations,
    remap_answer_citations,
    sources_from_answer,
)
from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CitationSource,
    RetrievedChunk,
)
from app.retrieval.fact_types import (
    answer_matches_fact_type,
    detect_fact_type,
    filter_evidence_for_fact,
)
from app.retrieval.multi_question import split_questions
from app.ingestion.service_domain import extract_query_service_domain
from app.retrieval.procedure_context import (
    assign_subquestion_domains,
    build_procedure_context,
    filter_evidence_to_procedure,
)
from app.retrieval.query_understanding import (
    EXPERIENCE_QUERY_TYPES,
    SUMMARY_QUERY_TYPES,
    QueryUnderstanding,
    classify_query,
    extract_procedural_topic,
    resolve_compound_subquestions,
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
            service_domain=extract_query_service_domain(request.question),
            domain_source_question=request.question,
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
        working_history: List[ChatMessage] = list(request.history)
        procedure_context = build_procedure_context(request.question)
        query_service_domain = procedure_context.active_domain
        resolved_subquestions = resolve_compound_subquestions(
            sub_questions,
            original_question=request.question,
            active_procedure=procedure_context.active_procedure,
            active_domain=procedure_context.active_domain,
        )
        sub_domains = assign_subquestion_domains(
            original_question=request.question,
            sub_questions=sub_questions,
            context=procedure_context,
        )
        active_topic = procedure_context.active_procedure or extract_procedural_topic(
            request.question
        )
        # Seed the full compound question so later elliptical sub-questions inherit
        # the same procedure/topic/section instead of searching bare follow-ups.
        if active_topic and request.question.strip():
            working_history = [
                ChatMessage(role="user", content=request.question.strip()),
                *working_history,
            ]

        for index, question in enumerate(sub_questions, start=1):
            resolved_question = (
                resolved_subquestions[index - 1]
                if index - 1 < len(resolved_subquestions)
                else question
            )
            sub_domain = (
                sub_domains[index - 1]
                if index - 1 < len(sub_domains)
                else query_service_domain
            )
            # Per-sub domain for legitimate multi-domain compounds; otherwise lock.
            if procedure_context.allow_domain_switch:
                procedure_context.active_domain = sub_domain
                procedure_context.active_procedure = (
                    "health-insurance enrollment"
                    if sub_domain == "health_insurance"
                    else "move-in resident-registration"
                    if sub_domain == "resident_registration"
                    else procedure_context.active_procedure
                )
                # Clear document anchor when switching procedures mid-compound.
                procedure_context.active_document = None
                procedure_context.active_section = None
                procedure_context.active_action = (
                    "enroll"
                    if sub_domain == "health_insurance"
                    else "register"
                    if sub_domain == "resident_registration"
                    else procedure_context.active_action
                )
            else:
                procedure_context.active_domain = query_service_domain
            response = await self._chat_single(
                request=request,
                question=resolved_question,
                conversation_id=conversation_id,
                history=working_history,
                renumber_citations_from=1,
                display_question=question,
                service_domain=procedure_context.active_domain,
                domain_source_question=request.question,
                procedure_context=procedure_context,
            )
            answer = response.answer.strip()
            working_history.extend(
                [
                    ChatMessage(role="user", content=question),
                    ChatMessage(role="assistant", content=answer),
                ]
            )
            if not active_topic:
                active_topic = extract_procedural_topic(question)
            # Remap local [1],[2] markers onto the global, deduped source list.
            local_to_global: Dict[int, int] = {}
            for local_num, source in enumerate(response.sources or [], start=1):
                key = (source.document_name, source.page_number)
                if key not in seen_source_keys:
                    citation_offset += 1
                    seen_source_keys.add(key)
                    # Track mapping from key → global number via parallel list length.
                    all_sources.append(
                        source.model_copy(update={"number": citation_offset})
                    )
                    local_to_global[local_num] = citation_offset
                else:
                    # Find existing global number for this source key.
                    for existing in all_sources:
                        if (
                            existing.document_name,
                            existing.page_number,
                        ) == key:
                            local_to_global[local_num] = existing.number
                            break
            if local_to_global:
                answer = remap_answer_citations(answer, local_to_global)
            if len(sub_questions) > 1:
                parts.append(f"{index}. {question}\n{answer}")
            else:
                parts.append(answer)
            if settings.is_development and response.diagnostics:
                candidates = (
                    response.diagnostics.get("initial_candidates")
                    or response.diagnostics.get("reranked_candidates")
                    or []
                )
                sub_diags.append(
                    {
                        "question": question,
                        "resolved_question": resolved_question,
                        "service_domain": procedure_context.active_domain,
                        "active_domain": procedure_context.active_domain,
                        "active_topic": active_topic
                        or response.diagnostics.get("active_section"),
                        "active_procedure": procedure_context.active_procedure,
                        "anchor_document": procedure_context.active_document,
                        "anchor_section": procedure_context.active_section,
                        "active_section": response.diagnostics.get("active_section"),
                        "query_intent": response.diagnostics.get("query_intent"),
                        "fact_type": response.diagnostics.get("fact_type"),
                        "answer": answer,
                        "top_candidates": candidates[:20],
                        "candidates_before_rerank": response.diagnostics.get(
                            "candidates_before_rerank"
                        ),
                        "candidates_after_rerank": response.diagnostics.get(
                            "candidates_after_rerank"
                        ),
                        "continuity_diagnostics": response.diagnostics.get(
                            "continuity_diagnostics"
                        ),
                        "rejected_cross_domain": response.diagnostics.get(
                            "rejected_cross_domain"
                        ),
                        "section_names": sorted(
                            {
                                str(item.get("section_title") or "").strip()
                                for item in candidates[:20]
                                if item.get("section_title")
                            }
                        ),
                        "final_evidence": response.diagnostics.get(
                            "evidence_sent_to_llm"
                        ),
                        "evidence_sent_to_llm": response.diagnostics.get(
                            "evidence_sent_to_llm"
                        ),
                        "exact_label_matches": response.diagnostics.get(
                            "exact_label_matches"
                        ),
                        "checklist_assembly": response.diagnostics.get(
                            "checklist_assembly"
                        ),
                        "answer_synthesis": response.diagnostics.get(
                            "answer_synthesis"
                        ),
                    }
                )

        diagnostics = None
        if settings.is_development:
            diagnostics = {
                "original_question": request.question,
                "service_domain": query_service_domain,
                "active_domain": procedure_context.active_domain,
                "procedure_context": procedure_context.to_diagnostics(),
                "sub_questions": list(sub_questions),
                "resolved_sub_questions": list(resolved_subquestions),
                "sub_question_domains": list(sub_domains),
                "active_topic": active_topic,
                "sub_question_results": sub_diags,
                "query_intent": "multi",
                "top_candidates_across_files": _flatten_top_candidates(sub_diags),
            }
            if re.search(
                r"(?i)\b(register|moving|move-?in|bring|notification|enroll|insurance)\b",
                request.question or "",
            ):
                diagnostics.update(
                    await self._procedural_query_diagnostics(
                        request=request,
                        sub_questions=sub_questions,
                        resolved_subquestions=resolved_subquestions,
                        active_topic=active_topic,
                    )
                )

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
        display_question: Optional[str] = None,
        service_domain: Optional[str] = None,
        domain_source_question: Optional[str] = None,
        procedure_context=None,
    ) -> ChatResponse:
        settings = get_settings()
        query_domain = service_domain or extract_query_service_domain(
            domain_source_question or question
        )
        evidence, understanding, inspection = await self.retriever.retrieve(
            company_id=request.company_id,
            question=question,
            top_k=request.top_k or 5,
            history=history,
            session_id=request.session_id,
            include_company_docs=request.include_company_docs,
            service_domain=query_domain,
            domain_source_question=domain_source_question or request.question,
            procedure_context=procedure_context,
        )

        if procedure_context is not None:
            evidence = filter_evidence_to_procedure(evidence, procedure_context)
            # Mutate shared continuity anchor for later sub-questions.
            updated = procedure_context.updated_from_evidence(evidence)
            procedure_context.active_document = updated.active_document
            procedure_context.active_section = updated.active_section
            procedure_context.active_entity = updated.active_entity
            procedure_context.active_action = updated.active_action
            if not procedure_context.locked or procedure_context.allow_domain_switch:
                procedure_context.active_domain = updated.active_domain

        fact_type = detect_fact_type(question, understanding.query_type)
        evidence = filter_evidence_for_fact(
            evidence, fact_type, question=question
        )
        if procedure_context is not None and fact_type == "checklist":
            evidence = filter_evidence_to_procedure(evidence, procedure_context)

        diagnostics = None
        if settings.is_development:
            diagnostics = _build_diagnostics(inspection)
            diagnostics["fact_type"] = fact_type
            diagnostics["service_domain"] = understanding.service_domain or query_domain
            diagnostics["procedure_context"] = (
                procedure_context.to_diagnostics() if procedure_context else None
            )
            diagnostics["candidates_before_rerank"] = [
                _chunk_diag(chunk)
                for chunk in getattr(inspection, "candidates_before_rerank", [])[:20]
            ]
            diagnostics["candidates_after_rerank"] = [
                _chunk_diag(chunk)
                for chunk in getattr(inspection, "candidates_after_rerank", [])[:20]
            ]
            diagnostics["continuity_diagnostics"] = getattr(
                inspection, "continuity_diagnostics", []
            )
            diagnostics["rejected_cross_domain"] = getattr(
                inspection, "rejected_cross_domain", []
            )
            diagnostics["top_candidates_across_files"] = [
                _candidate_rank_diag(chunk) for chunk in inspection.initial_candidates[:20]
            ]
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

            # Soft guide: prefer strong experience evidence, but do not wipe all
            # candidates here — fact validation remains the hard gate.
            strong = [
                item for item in evidence if is_strong_experience_evidence(item.content)
            ]
            if strong:
                evidence = strong

        answer = FALLBACK_ANSWER
        sources: List[CitationSource] = []
        cited: List[RetrievedChunk] = []
        used_llm = False
        generation_function = "none"
        response_formatter = "none"

        evidence_diag: Dict[str, Any] = {}
        if evidence:
            from app.generation.answer_synthesis import normalize_evidence_bundle
            from app.generation.text_scrub import scrub_internal_metadata

            evidence, evidence_diag = normalize_evidence_bundle(
                evidence, question=question
            )
            evidence = [
                item.model_copy(
                    update={"content": scrub_internal_metadata(item.content or "")}
                )
                for item in evidence
                if scrub_internal_metadata(item.content or "").strip()
            ]
            if diagnostics is not None:
                diagnostics["answer_synthesis"] = evidence_diag
                q_l = (question or "").lower()
                if re.search(r"\b(mov(?:e|ing)|register|bring|checklist)\b", q_l):
                    logger.info(
                        "move_in_synthesis selected=%s boundaries=%s merged=%s "
                        "checklist=%s evidence=%s",
                        evidence_diag.get("selected_chunk_ids"),
                        evidence_diag.get("chunk_boundary_flags"),
                        evidence_diag.get("merged_siblings"),
                        evidence_diag.get("checklist_structure"),
                        evidence_diag.get("final_clean_evidence"),
                    )
                if re.search(r"\b(fee|cost|price|juminhyo|¥|yen)\b", q_l):
                    logger.info(
                        "fee_synthesis rows=%s normalized=%s",
                        evidence_diag.get("selected_chunk_ids"),
                        evidence_diag.get("fee_rows_normalized"),
                    )

        logger.info(
            "[NEW_GENERATION_PATH] GENERATION FUNCTION: ChatService._chat_single "
            "provider=%s evidence=%s",
            type(self.llm).__name__,
            len(evidence),
        )
        logger.info(
            "[NEW_GENERATION_PATH] SELECTED EVIDENCE: %s",
            [
                {
                    "document_name": c.document_name,
                    "section_title": c.section_title,
                    "preview": (c.content or "")[:180],
                }
                for c in evidence[:6]
            ],
        )

        if evidence:
            try:
                answer, _raw_sources = await self.llm.generate(
                    question=question,
                    evidence=evidence,
                    history=history,
                )
                used_llm = type(self.llm).__name__ != "DeterministicFallbackProvider"
                generation_function = f"{type(self.llm).__name__}.generate"
                response_formatter = "_post_validate_answer"
                if fact_type == "checklist":
                    answer = await self._ensure_checklist_completeness(
                        question=question,
                        answer=answer,
                        evidence=evidence,
                        history=history,
                    )
                answer = _post_validate_answer(
                    answer,
                    question,
                    understanding.query_type,
                    fact_type,
                    evidence,
                )
                if diagnostics is not None:
                    diagnostics["generated_answer_pre_cite"] = answer
                    diagnostics["validation_result"] = {
                        "accepted": answer != FALLBACK_ANSWER,
                        "fact_type": fact_type,
                        "is_fallback": answer == FALLBACK_ANSWER,
                    }
                    logger.info(
                        "generation_validation fact=%s accepted=%s answer_preview=%s",
                        fact_type,
                        answer != FALLBACK_ANSWER,
                        (answer or "")[:240],
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
                    answer, sources, cited = reconcile_answer_citations(
                        answer, evidence
                    )
                    if not sources:
                        sources = sources_from_answer(
                            answer,
                            evidence,
                            query_type=understanding.query_type,
                            allow_uncited_fallback=understanding.query_type
                            in SUMMARY_QUERY_TYPES,
                        )
            except Exception:  # noqa: BLE001
                logger.exception("LLM generation failed; trying deterministic compose")
                answer, sources = FALLBACK_ANSWER, []
                cited = []
                used_llm = False
                generation_function = "compose_answer_fallback_after_llm_error"
                response_formatter = "compose_answer"

        if answer == FALLBACK_ANSWER:
            answer, sources = compose_answer(
                understanding=understanding, evidence=evidence
            )
            if generation_function == "none" or generation_function.startswith(
                "compose_answer"
            ):
                generation_function = "compose_answer"
                response_formatter = "_post_validate_answer"
                used_llm = False
            if answer != FALLBACK_ANSWER:
                answer = _post_validate_answer(
                    answer,
                    question,
                    understanding.query_type,
                    fact_type,
                    evidence,
                )
                if answer == FALLBACK_ANSWER:
                    sources = []
            if answer != FALLBACK_ANSWER and not answer_matches_fact_type(
                answer, fact_type, evidence
            ):
                answer, sources = FALLBACK_ANSWER, []
            cited = cited_evidence(answer, evidence) if sources else []
            if answer != FALLBACK_ANSWER:
                answer, reconciled_sources, reconciled_cited = (
                    reconcile_answer_citations(answer, evidence)
                )
                if reconciled_sources:
                    sources = reconciled_sources
                    cited = reconciled_cited
            # Deterministic compose often omits [n] markers — attach sources from used evidence.
            if answer != FALLBACK_ANSWER and not sources and evidence:
                sources = sources_from_answer(
                    answer,
                    evidence,
                    query_type=understanding.query_type,
                    allow_uncited_fallback=True,
                )

        logger.info(
            "[NEW_GENERATION_PATH] USING LLM: %s | GENERATION FUNCTION: %s | "
            "RESPONSE FORMATTER: %s | FINAL TEXT BEFORE RESPONSE: %s",
            used_llm,
            generation_function,
            response_formatter,
            (answer or "")[:400],
        )
        if diagnostics is not None:
            diagnostics["live_generation_path"] = {
                "marker": "[NEW_GENERATION_PATH]",
                "using_llm": used_llm,
                "generation_function": generation_function,
                "response_formatter": response_formatter,
                "provider": type(self.llm).__name__,
            }

        if answer == FALLBACK_ANSWER:
            sources = []
            cited = []

        if renumber_citations_from != 1 and sources:
            mapping = {
                i + 1: renumber_citations_from + i for i in range(len(sources))
            }
            answer = remap_answer_citations(answer, mapping)
            sources = [
                source.model_copy(update={"number": renumber_citations_from + i})
                for i, source in enumerate(sources)
            ]

        if diagnostics is not None:
            if display_question:
                diagnostics["display_question"] = display_question
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

    async def _ensure_checklist_completeness(
        self,
        *,
        question: str,
        answer: str,
        evidence: Sequence[RetrievedChunk],
        history: Sequence | None,
    ) -> str:
        """Validate checklist coverage; regenerate once; else synthesize from structure."""
        from app.generation.answer_synthesis import (
            checklist_regeneration_instruction,
            extract_checklist_structure,
            format_checklist_item_phrase,
            missing_checklist_items,
            synthesize_checklist_answer,
        )
        from app.generation.evidence_validation import (
            checklist_answer_is_complete,
            strip_checklist_deadline_padding,
        )

        cleaned = strip_checklist_deadline_padding(_strip_citation_analysis_framing(answer))
        complete_before = checklist_answer_is_complete(cleaned, evidence)
        structure = extract_checklist_structure(evidence)
        missing = missing_checklist_items(cleaned, evidence)
        logger.info(
            "checklist_validation_start complete=%s items=%s missing=%s "
            "structure=%s answer_preview=%s",
            complete_before,
            len(structure),
            [format_checklist_item_phrase(m) for m in missing],
            [
                {
                    "type": e.get("type"),
                    "alternatives": e.get("alternatives"),
                    "condition": e.get("condition"),
                    "scope": e.get("scope"),
                    "critical": True,
                }
                for e in structure
            ],
            (cleaned or "")[:240],
        )
        if complete_before:
            return cleaned

        instruction = checklist_regeneration_instruction(evidence, cleaned)
        try:
            regenerated, _ = await self.llm.generate(
                question=f"{question}\n\n{instruction}",
                evidence=evidence,
                history=history,
            )
            regenerated = strip_checklist_deadline_padding(
                _strip_citation_analysis_framing(regenerated)
            )
            if checklist_answer_is_complete(regenerated, evidence):
                logger.info("checklist_regeneration succeeded after one retry")
                return regenerated
            logger.info(
                "checklist_regeneration still incomplete missing=%s",
                [
                    format_checklist_item_phrase(m)
                    for m in missing_checklist_items(regenerated, evidence)
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception("checklist regeneration failed; using structured fallback")

        synthesized = synthesize_checklist_answer(evidence)
        logger.info(
            "checklist_deterministic_fallback triggered items=%s answer_preview=%s",
            len(structure),
            (synthesized or "")[:240],
        )
        return synthesized if synthesized != FALLBACK_ANSWER else cleaned

    async def _procedural_query_diagnostics(
        self,
        *,
        request: ChatRequest,
        sub_questions: Sequence[str],
        resolved_subquestions: Sequence[str],
        active_topic: Optional[str],
    ) -> Dict[str, Any]:
        """Development diagnostics for move/register procedural queries."""
        payloads = []
        try:
            payloads = await self.retriever.store.list_payloads(
                request.company_id,
                session_id=request.session_id,
                include_company_docs=request.include_company_docs,
            )
        except Exception:  # noqa: BLE001
            payloads = []

        moving_chunks = []
        source_snippets = []
        for payload in payloads:
            content = str(payload.get("content") or "")
            heading = str(payload.get("section_title") or "")
            blob = f"{heading}\n{content}".lower()
            if re.search(
                r"(?i)\b(moving\s+in|move-?in|register(?:ation)?|within\s+\d+\s+days?)\b",
                blob,
            ):
                moving_chunks.append(
                    {
                        "content": content,
                        "section_title": heading,
                        "subsection_title": payload.get("subsection_title"),
                        "content_type": payload.get("content_type"),
                        "page_number": payload.get("page_number"),
                        "document_name": payload.get("document_name"),
                    }
                )
            if content:
                source_snippets.append(content[:400])

        return {
            "extracted_source_text_samples": source_snippets[:12],
            "moving_in_chunks": moving_chunks,
            "split_sub_questions": list(sub_questions),
            "resolved_sub_questions": list(resolved_subquestions),
            "active_topic": active_topic,
        }


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
    question: str,
    query_type: str,
    fact_type: str,
    evidence: Sequence[RetrievedChunk],
) -> str:
    if not answer:
        return FALLBACK_ANSWER
    answer = _strip_citation_analysis_framing(answer)
    from app.generation.answer_synthesis import (
        checklist_structure_is_covered,
        extract_checklist_structure,
        is_retrieval_dump_answer,
        synthesize_checklist_answer,
        synthesize_fee_answer,
    )
    from app.generation.evidence_presentation import (
        answer_exposes_internal_field_keys,
        looks_like_answer_fragment,
    )
    from app.generation.evidence_validation import (
        checklist_answer_is_complete,
        price_answer_is_complete,
    )

    # Hard reject retrieval dumps / field leaks before any soft pass-through.
    if (
        is_retrieval_dump_answer(answer)
        or answer_exposes_internal_field_keys(answer)
        or looks_like_answer_fragment(answer)
    ):
        if fact_type == "checklist" or query_type == "checklist":
            rewritten = synthesize_checklist_answer(evidence)
            if rewritten and rewritten != FALLBACK_ANSWER:
                return rewritten
        if fact_type == "price" or query_type == "price":
            rewritten = synthesize_fee_answer(question, evidence)
            if rewritten and rewritten != FALLBACK_ANSWER:
                return rewritten
        return FALLBACK_ANSWER

    if fact_type == "checklist":
        from app.generation.evidence_validation import strip_checklist_deadline_padding

        answer = strip_checklist_deadline_padding(answer)
        structure = extract_checklist_structure(evidence)
        if not checklist_answer_is_complete(answer, evidence) or (
            structure and not checklist_structure_is_covered(answer, structure)
        ):
            rewritten = synthesize_checklist_answer(evidence)
            logger.info(
                "checklist_validation failed; regenerated items=%s",
                len(structure),
            )
            return rewritten if rewritten != FALLBACK_ANSWER else FALLBACK_ANSWER
    if fact_type == "price":
        if not price_answer_is_complete(answer, question, evidence):
            rewritten = synthesize_fee_answer(question, evidence)
            logger.info("fee_validation failed; regenerated from normalized rows")
            return rewritten if rewritten != FALLBACK_ANSWER else FALLBACK_ANSWER
    if fact_type == "date":
        from app.generation.answer_composer import _date_from_evidence
        from app.retrieval.query_understanding import QueryUnderstanding

        if not answer_matches_fact_type(answer, fact_type, evidence):
            composed = _date_from_evidence(
                QueryUnderstanding(
                    original_question=question,
                    normalized_question=question,
                    resolved_question=question,
                    expanded_question=question,
                    query_type=query_type or "date",
                ),
                evidence,
            )
            if composed and composed != FALLBACK_ANSWER:
                if looks_like_answer_fragment(composed) or is_retrieval_dump_answer(
                    composed
                ):
                    return FALLBACK_ANSWER
                return composed
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
    if answer_exposes_internal_field_keys(answer) or is_retrieval_dump_answer(answer):
        if fact_type == "price":
            rewritten = synthesize_fee_answer(question, evidence)
            if rewritten and rewritten != FALLBACK_ANSWER:
                return rewritten
        if fact_type == "checklist":
            rewritten = synthesize_checklist_answer(evidence)
            if rewritten and rewritten != FALLBACK_ANSWER:
                return rewritten
        return FALLBACK_ANSWER
    if looks_like_internal_metadata(answer) or re.search(
        r"(?i)\b(record\s*type|profile\s*description)\s*:", answer
    ):
        return FALLBACK_ANSWER
    if looks_like_answer_fragment(answer) or _looks_like_raw_chunk_dump(answer, evidence):
        if fact_type == "checklist":
            rewritten = synthesize_checklist_answer(evidence)
            if rewritten and rewritten != FALLBACK_ANSWER:
                return rewritten
        if fact_type == "price":
            rewritten = synthesize_fee_answer(question, evidence)
            if rewritten and rewritten != FALLBACK_ANSWER:
                return rewritten
        return FALLBACK_ANSWER
    if not answer_matches_fact_type(answer, fact_type, evidence):
        return FALLBACK_ANSWER
    return answer


def _strip_citation_analysis_framing(answer: str) -> str:
    """Turn citation commentary into a direct answer while retaining its marker."""
    text = (answer or "").strip()
    match = re.match(
        r"(?is)^(?:according to\s+)?(?:the\s+)?"
        r"(?:(?:cited\s+)?(?:source|citation|evidence|document)\s*)?"
        r"(\[\d+\])\s+"
        r"(?:confirms?|states?|shows?|indicates?|says?|notes?)"
        r"(?:\s+that)?\s+",
        text,
    )
    if not match:
        return text
    marker = match.group(1)
    direct = text[match.end() :].strip()
    if marker not in direct:
        direct = f"{direct.rstrip()} {marker}".strip()
    return direct


def _build_diagnostics(inspection) -> Dict[str, Any]:
    return {
        "original_question": inspection.original_question,
        "normalized_question": inspection.normalized_question,
        "resolved_question": inspection.resolved_question,
        "context_question": inspection.context_question,
        "active_section": inspection.active_section,
        "service_domain": getattr(inspection, "service_domain", None),
        "query_intent": inspection.query_type,
        "expanded_question": inspection.expanded_question,
        "expanded_terms": inspection.expanded_terms,
        "subject_name": inspection.subject_name,
        "detected_key_value_fields": inspection.detected_key_value_fields,
        "exact_label_matches": inspection.exact_label_matches,
        "extracted_table_rows": inspection.extracted_table_rows,
        "table_chunks_created": [
            _chunk_diag(chunk) for chunk in inspection.table_chunks
        ],
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
        "checklist_assembly": getattr(inspection, "checklist_assembly", {}) or {},
    }


def _candidate_rank_diag(chunk: RetrievedChunk) -> Dict[str, Any]:
    diag = chunk.diagnostics or {}
    return {
        "filename": chunk.document_name,
        "document_name": chunk.document_name,
        "service_domain": chunk.service_domain or diag.get("service_domain"),
        "heading": chunk.section_title,
        "section_title": chunk.section_title,
        "document_status": chunk.document_status or diag.get("document_status"),
        "domain_match_score": diag.get("domain_match_score", diag.get("domain_boost")),
        "lexical_score": diag.get("lexical_score"),
        "vector_score": diag.get("dense_score", diag.get("vector_score")),
        "archive_penalty": diag.get("archive_penalty"),
        "final_score": diag.get(
            "final_score", diag.get("combined_score", chunk.score)
        ),
        "combined_score": diag.get("combined_score"),
        "content": chunk.content[:240],
    }


def _flatten_top_candidates(sub_diags: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for block in sub_diags:
        for item in block.get("top_candidates") or []:
            key = (
                item.get("document_name"),
                item.get("section_title"),
                (item.get("content") or "")[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            diag = item.get("metadata_boosts") or {}
            rows.append(
                {
                    "filename": item.get("document_name"),
                    "service_domain": item.get("service_domain"),
                    "heading": item.get("section_title"),
                    "domain_match_score": item.get(
                        "domain_match_score", diag.get("domain_boost")
                    ),
                    "lexical_score": item.get("lexical_score"),
                    "vector_score": item.get(
                        "vector_score", item.get("vector_similarity_score")
                    ),
                    "archive_penalty": item.get("archive_penalty"),
                    "final_score": item.get(
                        "final_score", item.get("combined_score")
                    ),
                }
            )
    rows.sort(key=lambda row: float(row.get("final_score") or 0.0), reverse=True)
    return rows[:20]


def _chunk_diag(chunk: RetrievedChunk) -> Dict[str, Any]:
    diag = chunk.diagnostics or {}
    return {
        "content": chunk.content,
        "document_name": chunk.document_name,
        "filename": chunk.document_name,
        "document_id": getattr(chunk, "document_id", None),
        "chunk_id": chunk.chunk_id,
        "chunk_index": getattr(chunk, "chunk_index", None),
        "page_number": chunk.page_number,
        "record_type": chunk.record_type,
        "content_type": chunk.content_type,
        "label": chunk.label,
        "value": chunk.value,
        "section_title": chunk.section_title,
        "heading": chunk.section_title,
        "subsection_title": chunk.subsection_title,
        "service_domain": chunk.service_domain or diag.get("service_domain"),
        "document_status": chunk.document_status or diag.get("document_status"),
        "vector_similarity_score": diag.get("dense_score"),
        "vector_score": diag.get("dense_score"),
        "semantic_score": diag.get("semantic_score", diag.get("dense_score")),
        "lexical_score": diag.get("lexical_score"),
        "heading_score": diag.get("heading_boost"),
        "domain_match_score": diag.get("domain_match_score", diag.get("domain_boost")),
        "continuity_score": diag.get("continuity_score"),
        "continuity_boost": diag.get("continuity_boost"),
        "cross_domain_penalty": diag.get("cross_domain_penalty"),
        "continuity_reason": diag.get("continuity_reason"),
        "archive_penalty": diag.get("archive_penalty"),
        "final_score": diag.get("final_score", diag.get("combined_score")),
        "exact_lexical_scores": diag.get("query_term_scores", {}),
        "numeric_currency_score": diag.get(
            "numeric_currency_score", diag.get("numeric_boost")
        ),
        "exact_label_match_boost": diag.get("kv_boost"),
        "metadata_boosts": {
            "phrase_score": diag.get("phrase_score"),
            "label_boost": diag.get("label_boost"),
            "kv_boost": diag.get("kv_boost"),
            "type_boost": diag.get("type_boost"),
            "name_boost": diag.get("name_boost"),
            "quality_boost": diag.get("quality_boost"),
            "quality_reason": diag.get("quality_reason"),
            "active_section_boost": diag.get("active_section_boost"),
            "requirement_boost": diag.get("requirement_boost"),
            "action_boost": diag.get("action_boost"),
            "heading_boost": diag.get("heading_boost"),
            "numeric_boost": diag.get("numeric_boost"),
            "domain_boost": diag.get("domain_boost"),
            "title_boost": diag.get("title_boost"),
            "version_boost": diag.get("version_boost"),
            "continuity_boost": diag.get("continuity_boost"),
            "cross_domain_penalty": diag.get("cross_domain_penalty"),
            "continuity_reason": diag.get("continuity_reason"),
        },
        "reranker_score": diag.get("reranker_score"),
        "combined_score": diag.get("combined_score"),
    }
