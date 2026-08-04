"""Persistent chat history endpoints."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status

from app.config import get_settings
from app.dependencies import get_chat_history_service
from app.ingestion.formats import (
    EmptyDocumentError,
    PasswordProtectedError,
    UnsupportedFormatError,
)
from app.ingestion.pdf_loader import PdfExtractionError
from app.models.api import UploadDocumentResponse
from app.schemas.chat import (
    ChatAttachmentRead,
    ChatMessageCreate,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionRead,
    ChatSessionRename,
    ChatTurnRead,
)
from app.services.chat_history_service import (
    ChatAttachmentNotFoundError,
    ChatHistoryService,
    ChatSessionNotFoundError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat/sessions", tags=["chat-history"])


def _validate_company_id(company_id: str) -> str:
    cleaned = company_id.strip().lower()
    if not cleaned or not re.fullmatch(r"[a-z0-9_-]{1,64}", cleaned):
        raise HTTPException(status_code=400, detail="Invalid company_id.")
    return cleaned


@router.post("", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
def create_session(
    request: Optional[ChatSessionCreate] = None,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatSessionRead:
    return service.create_session(request or ChatSessionCreate())


@router.get("", response_model=List[ChatSessionRead])
def list_sessions(
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> List[ChatSessionRead]:
    return service.list_sessions()


@router.get("/{session_id}", response_model=ChatSessionDetail)
def get_session(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatSessionDetail:
    try:
        return service.get_session(session_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc


@router.patch("/{session_id}", response_model=ChatSessionRead)
def rename_session(
    session_id: str,
    request: ChatSessionRename,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatSessionRead:
    try:
        return service.rename_session(session_id, request)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> Response:
    try:
        await service.delete_session(session_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{session_id}/messages",
    response_model=ChatTurnRead,
    status_code=status.HTTP_201_CREATED,
)
async def add_message(
    session_id: str,
    request: ChatMessageCreate,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatTurnRead:
    logger.info(
        "LIVE_ROUTE POST /api/chat/sessions/%s/messages "
        "[NEW_GENERATION_PATH] company=%s q=%r",
        session_id,
        request.company_id,
        (request.content or "")[:160],
    )
    try:
        return await service.add_message(session_id, request)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/{session_id}/attachments",
    response_model=List[ChatAttachmentRead],
)
def list_attachments(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> List[ChatAttachmentRead]:
    try:
        return service.list_attachments(session_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc


@router.post(
    "/{session_id}/attachments",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(
    session_id: str,
    company_id: str = Form(...),
    file: UploadFile = File(...),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> UploadDocumentResponse:
    settings = get_settings()
    company_id = _validate_company_id(company_id)
    filename = file.filename or "document"

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    try:
        return await service.upload_attachment(
            session_id,
            company_id=company_id,
            filename=filename,
            data=data,
            content_type=file.content_type,
        )
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc
    except (
        ValueError,
        PdfExtractionError,
        UnsupportedFormatError,
        EmptyDocumentError,
        PasswordProtectedError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Session attachment upload failed")
        raise HTTPException(
            status_code=500,
            detail="Upload failed. Please try again or use a different file.",
        ) from exc


@router.delete(
    "/{session_id}/attachments/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_attachment(
    session_id: str,
    document_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> Response:
    try:
        await service.delete_attachment(session_id, document_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found.") from exc
    except ChatAttachmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Attachment not found.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
