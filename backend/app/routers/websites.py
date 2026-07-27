from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_document_service
from app.ingestion.web_loader import UnsafeUrlError
from app.models.api import UploadDocumentResponse, WebsiteIngestRequest
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/websites", tags=["websites"])


@router.post("/ingest", response_model=UploadDocumentResponse)
async def ingest_website(
    request: WebsiteIngestRequest,
    service: DocumentService = Depends(get_document_service),
) -> UploadDocumentResponse:
    try:
        return await service.ingest_site(
            company_id=request.company_id,
            url=request.url,
            max_pages=request.max_pages,
        )
    except UnsafeUrlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Website ingest failed")
        raise HTTPException(status_code=500, detail=f"Website ingest failed: {exc}") from exc
