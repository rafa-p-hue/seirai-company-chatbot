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

        from app.retrieval.conversation_context import (
            build_context_from_history,
            is_elliptical_followup_question,
            resolve_followup_question,
        )

        procedure_context = None
        conversation_anchors = None
        retrieval_question = request.question
        elliptical = is_elliptical_followup_question(request.question)
        if request.history and elliptical:
            procedure_context, conversation_anchors = build_context_from_history(
                question=request.question,
                history=request.history,
            )
            retrieval_question = resolve_followup_question(
                request.question,
                history=request.history,
                anchors=conversation_anchors,
            )
            if conversation_anchors and conversation_anchors.program:
                procedure_context.active_program = conversation_anchors.program
            if conversation_anchors and conversation_anchors.benefit:
                procedure_context.active_benefit = conversation_anchors.benefit
            if conversation_anchors and conversation_anchors.organization:
                procedure_context.active_entity = conversation_anchors.organization
        elif request.history:
            # Fully specified questions override the previous topic.
            # Keep soft anchors for diagnostics, but rebuild domain from CURRENT Q.
            _, conversation_anchors = build_context_from_history(
                question=request.question,
                history=request.history,
            )
            procedure_context = build_procedure_context(request.question)
            procedure_context.locked = False
            procedure_context.allow_domain_switch = True
            procedure_context.original_question = request.question

        # Domain always comes from the current question unless it is elliptical.
        domain_source = request.question
        if elliptical and conversation_anchors and conversation_anchors.prior_user_question:
            domain_source = conversation_anchors.prior_user_question
        current_domain = extract_query_service_domain(
            retrieval_question if elliptical else request.question
        )
        return await self._chat_single(
            request=request,
            question=retrieval_question,
            conversation_id=conversation_id,
            history=request.history,
            display_question=request.question,
            service_domain=current_domain,
            domain_source_question=domain_source,
            procedure_context=procedure_context,
            conversation_anchors=conversation_anchors,
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
        conversation_anchors=None,
    ) -> ChatResponse:
        settings = get_settings()
        original_user_question = display_question or question
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
            procedure_context.active_entity = (
                procedure_context.active_entity or updated.active_entity
            )
            procedure_context.active_action = updated.active_action
            if not procedure_context.locked or procedure_context.allow_domain_switch:
                procedure_context.active_domain = updated.active_domain

        from app.retrieval.conversation_context import (
            filter_chunks_to_active_entity,
            is_elliptical_followup_question,
        )
        from app.retrieval.entity_validation import (
            answer_matches_requested_entity,
            filter_evidence_for_requested_entity,
            format_not_found_answer,
            infer_not_found_contact_name,
        )

        singular_followup = bool(
            conversation_anchors
            and is_elliptical_followup_question(original_user_question)
        )
        if singular_followup and conversation_anchors is not None:
            evidence = filter_chunks_to_active_entity(
                evidence,
                conversation_anchors,
                singular_followup=True,
            )

        fact_type = detect_fact_type(question, understanding.query_type)
        evidence = filter_evidence_for_fact(
            evidence, fact_type, question=question
        )
        evidence = _penalize_conflicting_topic_evidence(
            evidence, original_user_question
        )
        evidence = _promote_exact_attribute_evidence(
            evidence,
            question=original_user_question,
            fact_type=fact_type,
            inspection=inspection,
        )
        evidence = _complete_truncated_evidence(
            evidence,
            question=original_user_question,
            inspection=inspection,
        )
        # Truncation repair can reattach same-doc siblings; re-apply topic cleanup.
        evidence = _penalize_conflicting_topic_evidence(
            evidence, original_user_question
        )
        if procedure_context is not None and fact_type in {"checklist", "procedure"}:
            evidence = filter_evidence_to_procedure(evidence, procedure_context)

        entity_diag: Dict[str, Any] = {}
        if fact_type == "price" or understanding.query_type == "price":
            evidence, entity_diag = filter_evidence_for_requested_entity(
                evidence, original_user_question
            )

        rescue_diag: Dict[str, Any] = {}
        store_payloads: List[Dict[str, Any]] = []
        try:
            store_payloads = await self.retriever.store.list_payloads(
                request.company_id,
                session_id=request.session_id,
                include_company_docs=request.include_company_docs,
            )
        except Exception:  # noqa: BLE001
            store_payloads = []
        # Validation emptied everything — attempt exact lexical / same-doc rescue
        # before returning not-found.
        if not evidence and understanding.query_type not in {
            "greeting",
            "unsupported",
            "help",
        }:
            rescued, rescue_diag = _rescue_evidence_from_candidates(
                question=original_user_question,
                fact_type=fact_type,
                inspection=inspection,
                prior_evidence=getattr(inspection, "final_chunks", None) or [],
                store_payloads=store_payloads,
            )
            if rescued:
                logger.info(
                    "validation_rescue original=%r fact=%s rescued=%s reasons=%s",
                    original_user_question,
                    fact_type,
                    [c.document_name for c in rescued],
                    rescue_diag.get("reasons"),
                )
                evidence = rescued
                if entity_diag.get("rejection_reason"):
                    entity_diag["rejection_reason"] = None
                    entity_diag["rescued_after_rejection"] = True
            elif entity_diag.get("rejection_reason"):
                logger.info(
                    "validation_rejected_all original=%r reason=%s rescue=%s",
                    original_user_question,
                    entity_diag.get("rejection_reason"),
                    rescue_diag,
                )

        from app.retrieval.answer_grounding import (
            detect_source_intent,
            filter_evidence_for_answer_grounding,
            format_office_not_found_answer,
            extract_requested_place,
        )

        grounding_diag: Dict[str, Any] = {}
        if understanding.query_type not in {"greeting", "unsupported", "help"}:
            evidence, grounding_diag = filter_evidence_for_answer_grounding(
                evidence,
                original_user_question,
                fact_type=fact_type,
            )
            if grounding_diag.get("rejected"):
                logger.info(
                    "GROUNDING_FILTER q=%r intent=%s kept=%s rejected=%s",
                    original_user_question,
                    grounding_diag.get("source_intent"),
                    len(grounding_diag.get("kept") or []),
                    [
                        r.get("rejection_reason")
                        for r in grounding_diag.get("rejected") or []
                    ][:8],
                )
            # If grounding wiped top-k, try rescue pool then re-ground.
            if not evidence and grounding_diag.get("rejection_reason"):
                rescued, rescue_diag = _rescue_evidence_from_candidates(
                    question=original_user_question,
                    fact_type=fact_type,
                    inspection=inspection,
                    prior_evidence=getattr(inspection, "initial_candidates", None)
                    or getattr(inspection, "final_chunks", None)
                    or [],
                    store_payloads=store_payloads,
                )
                if rescued:
                    evidence, grounding_diag = filter_evidence_for_answer_grounding(
                        rescued,
                        original_user_question,
                        fact_type=fact_type,
                    )
                    if evidence:
                        rescue_diag["reasons"] = list(rescue_diag.get("reasons") or []) + [
                            "grounding_rescue"
                        ]
                        logger.info(
                            "grounding_rescue original=%r intent=%s kept=%s",
                            original_user_question,
                            grounding_diag.get("source_intent"),
                            [c.document_name for c in evidence],
                        )

        diagnostics = None
        if settings.is_development:
            diagnostics = _build_diagnostics(inspection)
            diagnostics["fact_type"] = fact_type
            diagnostics["service_domain"] = understanding.service_domain or query_domain
            diagnostics["procedure_context"] = (
                procedure_context.to_diagnostics() if procedure_context else None
            )
            diagnostics["conversation_anchors"] = (
                conversation_anchors.to_diagnostics()
                if conversation_anchors is not None
                else None
            )
            diagnostics["original_question"] = original_user_question
            diagnostics["resolved_contextual_question"] = question
            diagnostics["active_entity_topic"] = (
                conversation_anchors.to_diagnostics()
                if conversation_anchors is not None
                else None
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
            diagnostics["entity_validation"] = entity_diag
            if grounding_diag:
                diagnostics["answer_grounding"] = grounding_diag
            if rescue_diag:
                diagnostics["validation_rescue"] = rescue_diag
            if entity_diag.get("rejection_reason"):
                diagnostics["rejection_reason"] = entity_diag.get("rejection_reason")
            elif grounding_diag.get("rejection_reason"):
                diagnostics["rejection_reason"] = grounding_diag.get("rejection_reason")

        logger.info(
            "VALIDATION_COMPLETE q=%r fact=%s evidence=%s rejection=%s rescue=%s",
            original_user_question,
            fact_type,
            len(evidence),
            (entity_diag or {}).get("rejection_reason"),
            bool(rescue_diag.get("kept")) if rescue_diag else False,
        )

        not_found_name = infer_not_found_contact_name(
            question=domain_source_question or original_user_question,
            configured_name=settings.not_found_contact_name,
            document_names=[
                chunk.document_name
                for chunk in (evidence or [])
                if getattr(chunk, "document_name", None)
            ],
        )
        source_intent = detect_source_intent(original_user_question)
        place = extract_requested_place(original_user_question)
        if source_intent == "office_location" and place:
            not_found = format_office_not_found_answer(
                place, contact_name=not_found_name
            )
        else:
            not_found = format_not_found_answer(not_found_name)


        if understanding.query_type == "unsupported":
            return ChatResponse(
                answer=not_found,
                sources=[],
                conversation_id=conversation_id,
                diagnostics=diagnostics,
            )

        if fact_type == "price" and entity_diag.get("rejection_reason") and not evidence:
            if diagnostics is not None:
                diagnostics["rejection_reason"] = entity_diag.get("rejection_reason")
                diagnostics["final_evidence"] = []
                logger.info(
                    "unsupported_entity original=%r resolved=%r anchors=%s "
                    "rejection=%s top=%s",
                    original_user_question,
                    question,
                    diagnostics.get("active_entity_topic"),
                    entity_diag.get("rejection_reason"),
                    diagnostics.get("top_candidates_across_files"),
                )
            return ChatResponse(
                answer=not_found,
                sources=[],
                conversation_id=conversation_id,
                diagnostics=diagnostics,
            )

        if not evidence and grounding_diag.get("rejection_reason"):
            if diagnostics is not None:
                diagnostics["rejection_reason"] = grounding_diag.get("rejection_reason")
                diagnostics["final_evidence"] = []
                logger.info(
                    "grounding_not_found original=%r intent=%s rejection=%s top=%s",
                    original_user_question,
                    grounding_diag.get("source_intent"),
                    grounding_diag.get("rejection_reason"),
                    (diagnostics or {}).get("top_candidates_across_files"),
                )
            return ChatResponse(
                answer=not_found,
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
                evidence, question=original_user_question
            )
            evidence = [
                item.model_copy(
                    update={"content": scrub_internal_metadata(item.content or "")}
                )
                for item in evidence
                if scrub_internal_metadata(item.content or "").strip()
            ]
            # Sibling merge can reintroduce conflicting move-out passages.
            evidence = _penalize_conflicting_topic_evidence(
                evidence, original_user_question
            )
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
            "LLM_GENERATION provider=%s evidence=%s q=%r",
            type(self.llm).__name__,
            len(evidence),
            original_user_question,
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

        # Generation uses the original user wording; retrieval used `question`.
        generation_question = original_user_question

        if evidence:
            try:
                answer, _raw_sources = await self.llm.generate(
                    question=generation_question,
                    evidence=evidence,
                    history=history,
                )
                used_llm = type(self.llm).__name__ != "DeterministicFallbackProvider"
                generation_function = f"{type(self.llm).__name__}.generate"
                response_formatter = "_post_validate_answer"
                if fact_type == "checklist":
                    answer = await self._ensure_checklist_completeness(
                        question=generation_question,
                        answer=answer,
                        evidence=evidence,
                        history=history,
                    )
                answer = _post_validate_answer(
                    answer,
                    generation_question,
                    understanding.query_type,
                    fact_type,
                    evidence,
                )
                if diagnostics is not None:
                    diagnostics["generated_answer_pre_cite"] = answer
                    diagnostics["validation_result"] = {
                        "accepted": answer != FALLBACK_ANSWER
                        and "could not find that information" not in answer.lower(),
                        "fact_type": fact_type,
                        "is_fallback": "could not find that information"
                        in (answer or "").lower(),
                    }
                    logger.info(
                        "generation_validation fact=%s accepted=%s answer_preview=%s",
                        fact_type,
                        diagnostics["validation_result"]["accepted"],
                        (answer or "")[:240],
                    )
                from app.generation.answer_composer import (
                    _finalize_answer,
                    _looks_like_raw_chunk_dump,
                )

                if answer and "could not find that information" not in answer.lower():
                    answer = _finalize_answer(answer)
                    if _looks_like_raw_chunk_dump(answer, evidence):
                        answer = FALLBACK_ANSWER
                if answer and "could not find that information" not in answer.lower():
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
            logger.info(
                "FALLBACK_FORMATTER compose_answer q=%r evidence=%s",
                original_user_question,
                len(evidence),
            )
            # Prefer original wording for compose_answer display semantics.
            understanding_for_compose = understanding
            if original_user_question != question:
                understanding_for_compose = understanding
                # Keep resolved_question for any compose heuristics that need it,
                # but route the user-facing question through original wording.
                understanding_for_compose = type(understanding)(
                    **{
                        **understanding.__dict__,
                        "original_question": original_user_question,
                    }
                )
            answer, sources = compose_answer(
                understanding=understanding_for_compose, evidence=evidence
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
                    generation_question,
                    understanding.query_type,
                    fact_type,
                    evidence,
                )
                if answer == FALLBACK_ANSWER:
                    sources = []
            if answer != FALLBACK_ANSWER and not answer_matches_fact_type(
                answer, fact_type, evidence, question=generation_question
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

        # Final relevance check: requested entity/service vs answer entity/service.
        if (
            answer
            and answer != FALLBACK_ANSWER
            and "could not find that information" not in answer.lower()
            and (fact_type == "price" or understanding.query_type == "price")
            and not answer_matches_requested_entity(answer, generation_question)
        ):
            if diagnostics is not None:
                diagnostics["rejection_reason"] = "answer_entity_mismatch"
                diagnostics["final_evidence"] = [
                    _chunk_diag(chunk) for chunk in evidence
                ]
                logger.info(
                    "answer_entity_mismatch original=%r resolved=%r answer=%r "
                    "top=%s",
                    original_user_question,
                    question,
                    (answer or "")[:240],
                    (diagnostics or {}).get("top_candidates_across_files"),
                )
            answer, sources, cited = not_found, [], []

        if answer == FALLBACK_ANSWER:
            answer = not_found
            sources = []
            cited = []

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
            diagnostics["final_evidence"] = [
                _chunk_diag(chunk) for chunk in evidence
            ]

        if "could not find that information" in (answer or "").lower():
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
    from app.retrieval.entity_validation import answer_matches_requested_category

    if not answer_matches_requested_category(answer, question):
        logger.info(
            "category_mismatch question=%r answer=%r",
            (question or "")[:160],
            (answer or "")[:240],
        )
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


def _looks_truncated_passage(text: str) -> bool:
    body = (text or "").rstrip()
    if not body:
        return False
    if body.endswith((".", "!", "?", ":", ";", "•")):
        return False
    if body.endswith("(") or body.count("(") > body.count(")"):
        return True
    if re.search(
        r"(?i)\b(?:and|or|with|for|to|of|the|a|an|bring|your|landing|passport|permission)\s*$",
        body,
    ):
        return True
    # Mid-word truncation (e.g. "car" from "card", "landing" cut before permission)
    if re.search(r"(?i)\b(?:landing|residenc|certificat|passpor|notif)\s*$", body):
        return True
    return False


def _complete_truncated_evidence(
    evidence: Sequence[RetrievedChunk],
    *,
    question: str,
    inspection,
) -> List[RetrievedChunk]:
    """Merge adjacent same-document candidates when selected evidence ends mid-phrase."""
    if not evidence:
        return list(evidence)

    pool: List[RetrievedChunk] = []
    seen = set()
    for group in (
        evidence,
        getattr(inspection, "final_chunks", None) or [],
        getattr(inspection, "reranked_candidates", None) or [],
        getattr(inspection, "initial_candidates", None) or [],
        getattr(inspection, "candidates_after_rerank", None) or [],
    ):
        for chunk in group or []:
            key = (chunk.document_name, chunk.chunk_id or (chunk.content or "")[:80])
            if key in seen:
                continue
            seen.add(key)
            pool.append(chunk)

    completed: List[RetrievedChunk] = []
    used_ids = set()
    for chunk in evidence:
        text = (chunk.content or "").rstrip()
        if not _looks_truncated_passage(text):
            completed.append(chunk)
            used_ids.add(chunk.chunk_id)
            continue
        # Find a same-document continuation: starts mid-sentence / closes paren /
        # continues the bring checklist.
        best = None
        for cand in pool:
            if cand.chunk_id and cand.chunk_id == chunk.chunk_id:
                continue
            if (cand.document_name or "").lower() != (chunk.document_name or "").lower():
                continue
            nxt = (cand.content or "").lstrip()
            if not nxt:
                continue
            continues = (
                nxt[0].islower()
                or nxt.startswith((")", ",", ";", "and ", "or "))
                or (
                    re.search(r"(?i)\b(moving-out certificate|my number|lease|bring)\b", nxt)
                    and re.search(r"(?i)\b(bring|passport|landing|residence\s+card)\b", text)
                )
            )
            if not continues:
                continue
            # Prefer closer chunk_index when available.
            best = cand
            if (
                chunk.chunk_index is not None
                and cand.chunk_index is not None
                and cand.chunk_index == (chunk.chunk_index or 0) + 1
            ):
                break
        if best is None:
            completed.append(chunk)
            used_ids.add(chunk.chunk_id)
            continue
        fused = f"{text} {best.content or ''}".strip()
        from app.generation.evidence_presentation import repair_passage_text

        fused = repair_passage_text(fused, section_title=chunk.section_title)
        logger.info(
            "MOVEIN_TRUNCATION_REPAIR base=%s sibling=%s preview=%r",
            chunk.chunk_id,
            best.chunk_id,
            fused[:180],
        )
        completed.append(chunk.model_copy(update={"content": fused}))
        used_ids.add(chunk.chunk_id)
        used_ids.add(best.chunk_id)
    # Keep unused siblings that still add checklist documents.
    q = (question or "").lower()
    if re.search(r"(?i)\b(bring|required|documents?|checklist)\b", q):
        for cand in pool:
            if cand.chunk_id in used_ids:
                continue
            if any(
                (c.document_name or "").lower() == (cand.document_name or "").lower()
                for c in completed
            ) and re.search(
                r"(?i)\b(my\s+number|moving-out certificate|lease|national health|"
                r"residence\s+card|passport)\b",
                cand.content or "",
            ):
                completed.append(cand)
                used_ids.add(cand.chunk_id)
    return completed


def _promote_exact_attribute_evidence(
    evidence: Sequence[RetrievedChunk],
    *,
    question: str,
    fact_type: str,
    inspection,
) -> List[RetrievedChunk]:
    """Keep/prioritize chunks with explicit entity+attribute lexical hits.

    Validation is an acceptance/ranking step: exact matches survive even when
    sibling merge or weak classifiers bury them.
    """
    q = (question or "").lower()
    wants_pet_shelter = bool(
        re.search(r"(?i)\b(pets?|pet[- ]friendly)\b", q)
        and re.search(r"(?i)\b(shelter|evacuat)", q)
    )
    if not wants_pet_shelter:
        return list(evidence)

    def _is_hit(chunk: RetrievedChunk) -> bool:
        blob = f"{chunk.section_title or ''}\n{chunk.content or ''}"
        return bool(
            re.search(r"(?i)\bpet[- ]friendly\b", blob)
            and re.search(r"(?i)\b(shelter|community\s+center|evacuat)", blob)
        )

    current = list(evidence)
    hits = [chunk for chunk in current if _is_hit(chunk)]
    if hits:
        rest = [chunk for chunk in current if not _is_hit(chunk)]
        return hits + rest

    # Exact hit was ranked but dropped by a later gate — rescue from candidates.
    rescued, diag = _rescue_evidence_from_candidates(
        question=question,
        fact_type=fact_type or "policy",
        inspection=inspection,
        prior_evidence=current,
    )
    exact = [chunk for chunk in rescued if _is_hit(chunk)]
    if exact:
        logger.info(
            "exact_attribute_promote pet_shelter rescued=%s reasons=%s",
            [c.document_name for c in exact],
            diag.get("reasons"),
        )
        return exact + current
    return current


def _penalize_conflicting_topic_evidence(
    evidence: Sequence[RetrievedChunk],
    question: str,
) -> List[RetrievedChunk]:
    """Ranking-style topic cleanup: drop clearly conflicting domains when the
    question fully specifies another program (still keep evidence if nothing else).
    """
    q = (question or "").lower()
    if not evidence:
        return list(evidence)
    asks_nhi = bool(
        re.search(r"(?i)\bnational\s+health\s+insurance|medical\s+costs?|patient\s+share|co-?payment\b", q)
    )
    asks_child = bool(re.search(r"(?i)\bchild(?:ren)?|2-year-old|childcare\b", q))
    if asks_nhi and not asks_child:
        kept = [
            chunk
            for chunk in evidence
            if (chunk.service_domain or "") != "childcare_support"
            and not re.search(
                r"(?i)\bchild\s+allowance\b",
                f"{chunk.document_name or ''}\n{chunk.section_title or ''}",
            )
        ]
        if kept:
            return kept
    # Move-in / register questions must not prefer move-out notifications.
    asks_move_in = bool(
        re.search(
            r"(?i)\b(just\s+moved|moving\s+in|move-?in|tennyu|register|what\s+do\s+i\s+bring)\b",
            q,
        )
    ) and not re.search(r"(?i)\b(moving\s+out|leave\s+hikari|leave\s+the\s+city)\b", q)
    if asks_move_in:
        def _is_move_out(chunk: RetrievedChunk) -> bool:
            head = f"{chunk.section_title or ''}\n{(chunk.content or '')[:120]}"
            return bool(
                re.search(r"(?i)\bmoving\s+out\b", head)
                and not re.search(r"(?i)\bmoving\s+in|tennyu\b", head)
            )

        def _is_move_in(chunk: RetrievedChunk) -> bool:
            blob = f"{chunk.section_title or ''}\n{chunk.content or ''}"
            return bool(
                re.search(
                    r"(?i)\b(moving\s+in|move-?in|tennyu|within\s+14\s+days\s+of\s+moving|"
                    r"bring|residence\s+card|passport\s+with\s+landing|my\s+number)\b",
                    blob,
                )
            ) and not _is_move_out(chunk)

        move_in = [chunk for chunk in evidence if _is_move_in(chunk)]
        if move_in:
            return move_in
        # At least drop pure move-out chunks when anything else remains.
        without_out = [chunk for chunk in evidence if not _is_move_out(chunk)]
        if without_out:
            return without_out
        # Fused PDF passages often start with Moving Out then include Moving In.
        # Keep only the move-in portion rather than answering from move-out.
        trimmed: List[RetrievedChunk] = []
        for chunk in evidence:
            content = chunk.content or ""
            if not re.search(r"(?i)\bmoving\s+out\b", content):
                trimmed.append(chunk)
                continue
            if not re.search(r"(?i)\bmoving\s+in|tennyu\b", content):
                continue
            match = re.search(
                r"(?is)(?:^|\n)\s*((?:Moving In|Move-In|Tennyu Todoke)\b.*)$",
                content,
            )
            keep = (match.group(1).strip() if match else "").strip()
            if not keep:
                # Fallback: drop leading move-out sentence block only.
                keep = re.sub(
                    r"(?is)^.*?((?:Moving In|Move-In|Tennyu Todoke)\b.*)$",
                    r"\1",
                    content,
                ).strip()
            if keep and re.search(r"(?i)\b(14\s+days|bring|window|tennyu|moving\s+in)\b", keep):
                trimmed.append(chunk.model_copy(update={"content": keep}))
        if trimmed:
            return trimmed
    return list(evidence)


def _chunk_from_store_payload(payload: Dict[str, Any]) -> RetrievedChunk:
    """Build a RetrievedChunk from a raw vector-store payload for rescue scans."""
    return RetrievedChunk(
        content=str(payload.get("content") or ""),
        document_name=str(payload.get("document_name") or "document"),
        page_number=payload.get("page_number"),
        slide_number=payload.get("slide_number"),
        row_number=payload.get("row_number"),
        source_url=payload.get("source_url"),
        score=0.0,
        chunk_id=payload.get("chunk_id"),
        document_id=payload.get("document_id"),
        chunk_index=payload.get("chunk_index"),
        section_title=payload.get("section_title"),
        subsection_title=payload.get("subsection_title"),
        record_type=payload.get("record_type"),
        content_type=payload.get("content_type"),
        label=payload.get("label"),
        value=payload.get("value"),
        table_data=payload.get("table_data"),
        document_status=payload.get("document_status"),
        service_domain=payload.get("service_domain"),
        effective_date=payload.get("effective_date"),
        version=payload.get("version"),
    )


def _rescue_evidence_from_candidates(
    *,
    question: str,
    fact_type: str,
    inspection,
    prior_evidence: Sequence[RetrievedChunk],
    store_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    """Second-pass exact phrase / heading / structured-row rescue after validation wipe."""
    from app.retrieval.entity_validation import (
        collect_named_service_fee_evidence,
        entity_match_score,
        extract_requested_service_phrases,
    )
    from app.retrieval.fact_types import evidence_matches_fact_type

    phrases = extract_requested_service_phrases(question)
    for match in re.finditer(
        r"(?i)\b("
        r"national\s+health\s+insurance|child\s+allowance|residence\s+certificate|"
        r"family\s+register|burnable\s+garbage|oversized\s+garbage|"
        r"evacuation\s+shelter|accepts?\s+pets?|pet[- ]friendly|"
        r"parking\s+permit|residential\s+parking|"
        r"patient\s+share|co-?payment|"
        r"move-?in|citizen\s+services|"
        r"permanent\s+placement|executive\s+search|placement\s+fee|recruitment\s+fee"
        r")\b",
        question or "",
    ):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if phrase and phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(phrase)

    # Explicit attribute rescue: named entity + requested attribute in chunk text.
    q_l = (question or "").lower()
    wants_pet_shelter = bool(
        re.search(r"(?i)\b(pets?|pet[- ]friendly)\b", q_l)
        and re.search(r"(?i)\b(shelter|evacuat)", q_l)
    )

    diagnostics: Dict[str, Any] = {
        "requested_phrases": phrases,
        "reasons": [],
        "candidates_considered": 0,
        "kept": [],
    }
    pool: List[RetrievedChunk] = []
    seen = set()

    def _add_to_pool(chunk: RetrievedChunk) -> None:
        key = (
            chunk.document_name,
            chunk.page_number,
            (chunk.content or "")[:120],
        )
        if key in seen:
            return
        seen.add(key)
        pool.append(chunk)

    for group in (
        prior_evidence,
        getattr(inspection, "initial_candidates", None) or [],
        getattr(inspection, "reranked_candidates", None) or [],
        getattr(inspection, "final_chunks", None) or [],
    ):
        for chunk in group:
            _add_to_pool(chunk)

    # Store-wide expansion: top-k can bury the free FAQ answer or the 22% fee
    # row when a prior turn's section continuity crowds candidates.
    from app.retrieval.answer_grounding import (
        detect_source_intent,
        is_candidate_fee_evidence,
    )

    intent = detect_source_intent(question)
    if store_payloads and intent in {
        "candidate_fee",
        "employer_fee",
    }:
        for payload in store_payloads:
            content = str(payload.get("content") or "")
            if not content:
                continue
            if intent == "candidate_fee" and not is_candidate_fee_evidence(content):
                # Still pull FAQ siblings when the question heading is present.
                doc = str(payload.get("document_name") or "").lower()
                if not (
                    "candidate" in doc
                    or "faq" in doc
                    or re.search(r"(?i)\bhave\s+to\s+pay\b", content)
                ):
                    continue
            if intent == "employer_fee":
                if not (
                    re.search(r"(?i)^\s*service\s*:", content)
                    or re.search(r"(?i)^\s*(?:fee|minimum\s+fee)\b", content)
                    or (
                        re.search(r"\d{1,3}(?:\.\d+)?\s*%", content)
                        and re.search(r"(?i)\b(first[- ]year|base\s+salary|fee)\b", content)
                    )
                ):
                    continue
            _add_to_pool(_chunk_from_store_payload(payload))
        diagnostics["reasons"].append("store_payload_expansion")

    diagnostics["candidates_considered"] = len(pool)
    if not pool:
        diagnostics["reasons"].append("no_candidates")
        return [], diagnostics

    # Named fee/service questions: recover Service + Fee rows even when top-k
    # was crowded by unrelated currency rows (e.g. job salaries).
    if fact_type == "price" and phrases:
        from app.retrieval.answer_grounding import (
            filter_evidence_for_answer_grounding,
            is_fee_process_noise_only,
        )

        fee_block = collect_named_service_fee_evidence(pool, question)
        fee_block = [
            chunk
            for chunk in fee_block
            if not is_fee_process_noise_only(chunk.content or "")
            and (
                evidence_matches_fact_type(chunk.content or "", "price", chunk)
                or re.search(r"(?i)\bservice\s*:", chunk.content or "")
                or entity_match_score(chunk.content or "", phrases) >= 0.75
            )
        ]
        fee_block, _ = filter_evidence_for_answer_grounding(
            fee_block, question, fact_type=fact_type
        )
        # Keep blocks that include at least one real price signal.
        if any(
            evidence_matches_fact_type(chunk.content or "", "price", chunk)
            for chunk in fee_block
        ) and (
            not re.search(r"(?i)\bpermanent\s+placement\b", question or "")
            or any(
                re.search(r"\d{1,3}(?:\.\d+)?\s*%", chunk.content or "")
                for chunk in fee_block
            )
        ):
            diagnostics["reasons"].append("named_service_fee_block")
            diagnostics["kept"] = [
                {
                    "document_name": chunk.document_name,
                    "section_title": chunk.section_title,
                    "preview": (chunk.content or "")[:140],
                }
                for chunk in fee_block[:8]
            ]
            return fee_block[:8], diagnostics

    # Candidate / job-seeker free-of-charge policy rescue.
    if (
        fact_type == "policy"
        or intent == "candidate_fee"
        or re.search(
            r"(?i)\b(job\s+seeker|as\s+a\s+candidate|have\s+to\s+pay)\b",
            question or "",
        )
    ):
        candidate_hits = [
            chunk for chunk in pool if is_candidate_fee_evidence(chunk.content or "")
        ]
        if not candidate_hits and store_payloads:
            # Log why FAQ-looking rows were rejected (heading without free text).
            for payload in store_payloads:
                content = str(payload.get("content") or "")
                doc = str(payload.get("document_name") or "")
                if "candidate" not in doc.lower() and "faq" not in doc.lower():
                    if not re.search(r"(?i)\b(job\s+seeker|candidate|have\s+to\s+pay)\b", content):
                        continue
                if is_candidate_fee_evidence(content):
                    candidate_hits.append(_chunk_from_store_payload(payload))
                else:
                    logger.info(
                        "CANDIDATE_FEE_REJECTED doc=%s reason=missing_free_of_charge "
                        "preview=%r",
                        doc,
                        content[:160],
                    )
        if candidate_hits:
            diagnostics["reasons"].append("candidate_fee_free_evidence")
            diagnostics["kept"] = [
                {
                    "document_name": chunk.document_name,
                    "preview": (chunk.content or "")[:140],
                }
                for chunk in candidate_hits[:4]
            ]
            return candidate_hits[:4], diagnostics

    kept: List[RetrievedChunk] = []
    for chunk in pool:
        blob = "\n".join(
            part
            for part in (
                chunk.content or "",
                chunk.section_title or "",
                chunk.document_name or "",
            )
            if part
        )
        entity_score = entity_match_score(blob, phrases) if phrases else 0.0
        heading_hit = bool(
            phrases
            and chunk.section_title
            and any(
                p.lower() in (chunk.section_title or "").lower() for p in phrases if len(p) >= 4
            )
        )
        exact = bool(
            phrases
            and any(p.lower() in (chunk.content or "").lower() for p in phrases if len(p) >= 6)
        )
        fact_ok = evidence_matches_fact_type(chunk.content or "", fact_type, chunk) if fact_type else True
        if wants_pet_shelter:
            if not (
                re.search(r"(?i)\bpet[- ]friendly\b", blob)
                and re.search(r"(?i)\b(shelter|community\s+center|evacuat)\b", blob)
            ):
                continue
            kept.append(chunk)
            continue
        if phrases and (exact or heading_hit or entity_score >= 0.75) and fact_ok:
            kept.append(chunk)
        elif not phrases and fact_ok and exact:
            kept.append(chunk)

    if kept:
        diagnostics["reasons"].append("exact_phrase_or_heading")
        diagnostics["kept"] = [
            {
                "document_name": chunk.document_name,
                "section_title": chunk.section_title,
                "preview": (chunk.content or "")[:140],
            }
            for chunk in kept[:8]
        ]
        return kept[:8], diagnostics

    diagnostics["reasons"].append("no_exact_phrase_match")
    return [], diagnostics


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
