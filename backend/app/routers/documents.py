from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_chat_history_service, get_document_service
from app.ingestion.formats import (
    EmptyDocumentError,
    PasswordProtectedError,
    UnsupportedFormatError,
)
from app.ingestion.pdf_loader import PdfExtractionError
from app.models.api import (
    DocumentChunksResponse,
    DocumentListResponse,
    StoredChunkView,
    UploadDocumentResponse,
)
from app.repositories.chat_repository import ChatRepository
from app.services.chat_history_service import ChatHistoryService, ChatSessionNotFoundError
from app.services.document_service import DocumentService
from app.ingestion.cleaner import estimate_tokens
from app.generation.text_scrub import scrub_internal_metadata

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _validate_company_id(company_id: str) -> str:
    cleaned = company_id.strip().lower()
    if not cleaned or not re.fullmatch(r"[a-z0-9_-]{1,64}", cleaned):
        raise HTTPException(status_code=400, detail="Invalid company_id.")
    return cleaned


@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    company_id: str = Form(...),
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    document_scope: Optional[str] = Form(None),
    service: DocumentService = Depends(get_document_service),
    history_service: ChatHistoryService = Depends(get_chat_history_service),
    database: Session = Depends(get_db),
) -> UploadDocumentResponse:
    settings = get_settings()
    company_id = _validate_company_id(company_id)
    filename = file.filename or "document"

    scoped_session = (session_id or "").strip() or None
    scope = (document_scope or "").strip().lower() or None
    if scoped_session:
        scope = "chat"
        repo = ChatRepository(database)
        if repo.get_session(scoped_session) is None:
            raise HTTPException(status_code=404, detail="Chat session not found.")
    if scope == "chat" and not scoped_session:
        raise HTTPException(
            status_code=400,
            detail="session_id is required when document_scope is 'chat'.",
        )
    if scope and scope not in {"company", "chat"}:
        raise HTTPException(
            status_code=400,
            detail="document_scope must be 'company' or 'chat'.",
        )

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")

    try:
        response = await service.upload_document(
            company_id=company_id,
            filename=filename,
            data=data,
            session_id=scoped_session,
            document_scope=scope,
            content_type=file.content_type,
        )
        if scoped_session:
            history_service.register_attachment(
                session_id=scoped_session,
                company_id=company_id,
                document_id=response.document.document_id,
                document_name=response.document.document_name,
                status=response.document.status,
            )
        return response
    except (
        ValueError,
        PdfExtractionError,
        UnsupportedFormatError,
        EmptyDocumentError,
        PasswordProtectedError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload failed")
        raise HTTPException(
            status_code=500,
            detail="Upload failed. Please try again or use a different file.",
        ) from exc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    company_id: str,
    session_id: Optional[str] = None,
    document_scope: Optional[str] = "company",
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List documents. Defaults to company-scope so admin is not polluted by chat uploads."""
    company_id = _validate_company_id(company_id)
    scope = (document_scope or "company").strip().lower()
    if scope not in {"company", "chat"}:
        raise HTTPException(
            status_code=400,
            detail="document_scope must be 'company' or 'chat'.",
        )
    scoped_session = (session_id or "").strip() or None
    if scope == "chat" and not scoped_session:
        raise HTTPException(
            status_code=400,
            detail="session_id is required when listing chat-scoped documents.",
        )
    documents = await service.list_documents(
        company_id,
        session_id=scoped_session,
        document_scope=scope,
        include_company_docs=False if scope == "chat" else True,
    )
    return DocumentListResponse(documents=documents)


@router.get("/debug/index-status")
async def debug_index_status(
    company_id: str,
    session_id: str,
    service: DocumentService = Depends(get_document_service),
    database: Session = Depends(get_db),
) -> dict:
    """List chat attachments vs indexed chunk/embedding counts (dev recovery)."""
    settings = get_settings()
    if not settings.is_development:
        raise HTTPException(status_code=404, detail="Not Found")
    company_id = _validate_company_id(company_id)
    repo = ChatRepository(database)
    if repo.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    attachments = repo.list_attachments(session_id)
    return await service.index_status_for_session(
        company_id=company_id,
        session_id=session_id,
        attachments=attachments,
    )


@router.post("/debug/rebuild-session")
async def debug_rebuild_session(
    company_id: str,
    session_id: str,
    service: DocumentService = Depends(get_document_service),
    history_service: ChatHistoryService = Depends(get_chat_history_service),
    database: Session = Depends(get_db),
) -> dict:
    """Re-ingest all chat attachments for a session from on-disk upload files."""
    settings = get_settings()
    if not settings.is_development:
        raise HTTPException(status_code=404, detail="Not Found")
    company_id = _validate_company_id(company_id)
    repo = ChatRepository(database)
    if repo.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    attachments = repo.list_attachments(session_id)
    result = await service.rebuild_session_index(
        company_id=company_id,
        session_id=session_id,
        attachments=attachments,
    )
    for item in result.get("rebuild_results") or []:
        if item.get("status") == "ready":
            history_service.register_attachment(
                session_id=session_id,
                company_id=company_id,
                document_id=item["document_id"],
                document_name=item.get("filename") or item["document_id"],
                status="ready",
            )
    return result


@router.get("/debug/phrase-scan")
async def debug_phrase_scan(
    company_id: str,
    session_id: Optional[str] = None,
    service: DocumentService = Depends(get_document_service),
) -> dict:
    """Confirm whether key phrases exist as searchable chunks."""
    settings = get_settings()
    if not settings.is_development:
        raise HTTPException(status_code=404, detail="Not Found")
    company_id = _validate_company_id(company_id)
    phrases = [
        "Residence certificate (Juminhyo)",
        "Moving In (Tennyu Todoke)",
        "Citizen Affairs Division Window 3",
        "¥350",
        "Moving-Out Certificate",
    ]
    return await service.scan_indexed_phrases(
        company_id=company_id,
        session_id=session_id,
        phrases=phrases,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    company_id: str,
    service: DocumentService = Depends(get_document_service),
) -> dict:
    company_id = _validate_company_id(company_id)
    deleted = await service.delete_document(company_id, document_id)
    return {"deleted_chunks": deleted, "document_id": document_id}


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def list_document_chunks(
    document_id: str,
    company_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentChunksResponse:
    """Development/admin view of stored chunks for a document."""
    company_id = _validate_company_id(company_id)
    chunks = await service.list_document_chunks(company_id, document_id)
    views = [
        StoredChunkView(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            subsection_title=chunk.subsection_title,
            content_type=chunk.content_type,
            token_count=estimate_tokens(chunk.content or ""),
            content=scrub_internal_metadata(chunk.content or ""),
            chunk_index=chunk.chunk_index,
        )
        for chunk in chunks
    ]
    return DocumentChunksResponse(
        document_id=document_id,
        company_id=company_id,
        chunks=views,
    )


@router.post("/{document_id}/reprocess", response_model=UploadDocumentResponse)
async def reprocess_document(
    document_id: str,
    company_id: str,
    session_id: Optional[str] = None,
    document_name: Optional[str] = None,
    service: DocumentService = Depends(get_document_service),
) -> UploadDocumentResponse:
    company_id = _validate_company_id(company_id)
    try:
        return await service.reprocess_document(
            company_id,
            document_id,
            session_id=session_id,
            document_name=document_name,
            document_scope="chat" if session_id else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
