from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import get_chat_service, get_retriever
from app.models.api import ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse
from app.retrieval.retriever import Retriever
from app.services.chat_service import ChatService, _build_diagnostics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _validate_company_id(company_id: str) -> str:
    cleaned = company_id.strip().lower()
    if not cleaned or not re.fullmatch(r"[a-z0-9_-]{1,64}", cleaned):
        raise HTTPException(status_code=400, detail="Invalid company_id.")
    return cleaned


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    request: RetrieveRequest,
    retriever: Retriever = Depends(get_retriever),
) -> RetrieveResponse:
    company_id = _validate_company_id(request.company_id)
    results, understanding, inspection = await retriever.retrieve(
        company_id=company_id,
        question=request.question,
        top_k=request.top_k,
        session_id=request.session_id,
        include_company_docs=request.include_company_docs,
    )
    settings = get_settings()
    inspection_payload = None
    if settings.is_development:
        from app.retrieval.fact_types import detect_fact_type, filter_evidence_for_fact

        inspection_payload = _build_diagnostics(inspection)
        before = list(results)
        fact_type = detect_fact_type(request.question, understanding.query_type)
        after = before
        rejected: list = []
        if not request.skip_evidence_validation:
            after = filter_evidence_for_fact(
                before, fact_type, question=request.question
            )
            kept = {
                (c.document_name, (c.content or "")[:80]) for c in after
            }
            for chunk in before:
                key = (chunk.document_name, (chunk.content or "")[:80])
                if key not in kept:
                    diag = chunk.diagnostics or {}
                    rejected.append(
                        {
                            "document_name": chunk.document_name,
                            "section_title": chunk.section_title,
                            "content": (chunk.content or "")[:240],
                            "vector_score": diag.get("dense_score"),
                            "lexical_score": diag.get("lexical_score"),
                            "domain_score": diag.get(
                                "domain_match_score", diag.get("domain_boost")
                            ),
                            "rejection_reason": "fact_filter",
                        }
                    )
            results = after
        else:
            inspection_payload["skip_evidence_validation"] = True
        inspection_payload["candidates_before_validation"] = [
            {
                "document_name": c.document_name,
                "section_title": c.section_title,
                "content": (c.content or "")[:280],
                "vector_score": (c.diagnostics or {}).get("dense_score"),
                "lexical_score": (c.diagnostics or {}).get("lexical_score"),
                "domain_score": (c.diagnostics or {}).get(
                    "domain_match_score", (c.diagnostics or {}).get("domain_boost")
                ),
                "final_score": (c.diagnostics or {}).get(
                    "final_score", (c.diagnostics or {}).get("combined_score", c.score)
                ),
            }
            for c in before[:20]
        ]
        inspection_payload["candidates_after_validation"] = [
            {
                "document_name": c.document_name,
                "section_title": c.section_title,
                "content": (c.content or "")[:280],
            }
            for c in after[:20]
        ]
        inspection_payload["rejected_by_validation"] = rejected[:20]
        inspection_payload["candidate_count_before_filter"] = len(
            getattr(inspection, "initial_candidates", []) or before
        )
        inspection_payload["candidate_count_after_validation"] = len(after)
        inspection_payload["fallback_reason"] = (
            None
            if after
            else (
                "no_indexed_candidates"
                if not (getattr(inspection, "initial_candidates", None) or before)
                else "all_candidates_rejected_by_validation"
            )
        )
    return RetrieveResponse(
        results=results,
        original_query=understanding.original_question,
        expanded_query=understanding.expanded_question,
        query_type=understanding.query_type,
        expanded_terms=understanding.expanded_terms,
        resolved_query=understanding.resolved_question,
        subject_name=understanding.subject_name,
        inspection=inspection_payload,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    logger.info(
        "LIVE_ROUTE POST /api/chat [NEW_GENERATION_PATH] company=%s q=%r",
        request.company_id,
        (request.question or "")[:160],
    )
    request.company_id = _validate_company_id(request.company_id)
    try:
        return await service.chat(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Chat request failed.") from exc
