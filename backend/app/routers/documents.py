from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.dependencies import get_document_service
from app.ingestion.pdf_loader import PdfExtractionError
from app.models.api import DocumentListResponse, UploadDocumentResponse
from app.services.document_service import DocumentService

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
    service: DocumentService = Depends(get_document_service),
) -> UploadDocumentResponse:
    settings = get_settings()
    company_id = _validate_company_id(company_id)
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF exceeds the 10 MB limit.")

    try:
        return await service.upload_pdf(
            company_id=company_id,
            filename=filename,
            data=data,
        )
    except (ValueError, PdfExtractionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    company_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    company_id = _validate_company_id(company_id)
    documents = await service.list_documents(company_id)
    return DocumentListResponse(documents=documents)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    company_id: str,
    service: DocumentService = Depends(get_document_service),
) -> dict:
    company_id = _validate_company_id(company_id)
    deleted = await service.delete_document(company_id, document_id)
    return {"deleted_chunks": deleted, "document_id": document_id}


@router.post("/{document_id}/reprocess", response_model=UploadDocumentResponse)
async def reprocess_document(
    document_id: str,
    company_id: str,
    service: DocumentService = Depends(get_document_service),
) -> UploadDocumentResponse:
    company_id = _validate_company_id(company_id)
    try:
        return await service.reprocess_document(company_id, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
