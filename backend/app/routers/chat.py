from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chat_service, get_retriever
from app.models.api import ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse
from app.retrieval.retriever import Retriever
from app.services.chat_service import ChatService

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
    results, understanding = await retriever.retrieve(
        company_id=company_id,
        question=request.question,
        top_k=request.top_k,
    )
    return RetrieveResponse(
        results=results,
        original_query=understanding.original_question,
        expanded_query=understanding.expanded_question,
        query_type=understanding.query_type,
        expanded_terms=understanding.expanded_terms,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    request.company_id = _validate_company_id(request.company_id)
    try:
        return await service.chat(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Chat request failed.") from exc
