from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.ingestion.pipeline import ingest_pdf_bytes, ingest_website
from app.models.api import (
    DocumentChunk,
    DocumentSummary,
    ExtractedPage,
    UploadDocumentResponse,
)
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
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

    async def upload_pdf(
        self,
        *,
        company_id: str,
        filename: str,
        data: bytes,
        replace_document_id: Optional[str] = None,
    ) -> UploadDocumentResponse:
        if len(data) > self.settings.max_upload_bytes:
            raise ValueError(
                f"File exceeds maximum size of {self.settings.max_upload_bytes} bytes."
            )
        if not data.startswith(b"%PDF"):
            raise ValueError("Uploaded file does not look like a valid PDF.")

        if replace_document_id:
            await self.store.delete_document(company_id, replace_document_id)

        summary, pages, chunks = await ingest_pdf_bytes(
            data=data,
            filename=filename,
            company_id=company_id,
            document_id=replace_document_id,
        )
        embedded = await self._embed_and_store(summary, chunks)
        return UploadDocumentResponse(
            document=embedded,
            pages=pages,
            chunks=chunks,
            message="Document ingested successfully.",
        )

    async def ingest_site(
        self,
        *,
        company_id: str,
        url: str,
        max_pages: int,
    ) -> UploadDocumentResponse:
        summary, pages, chunks = await ingest_website(
            company_id=company_id,
            url=url,
            max_pages=min(max_pages, self.settings.max_crawl_pages),
            allowlist=self.settings.crawl_domain_allowlist(),
        )
        embedded = await self._embed_and_store(summary, chunks)
        return UploadDocumentResponse(
            document=embedded,
            pages=pages,
            chunks=chunks,
            message="Website ingested successfully.",
        )

    async def list_documents(self, company_id: str) -> List[DocumentSummary]:
        docs = await self.store.list_documents(company_id)
        for doc in docs:
            doc.embedding_model = self.embeddings.model_name
            doc.embedding_dimension = self.embeddings.dimensions
        return docs

    async def delete_document(self, company_id: str, document_id: str) -> int:
        return await self.store.delete_document(company_id, document_id)

    async def reprocess_document(
        self, company_id: str, document_id: str
    ) -> UploadDocumentResponse:
        existing = await self.store.get_document_chunks(company_id, document_id)
        if not existing:
            raise ValueError("Document not found for this company.")

        document_name = existing[0].document_name
        source_type = existing[0].source_type
        source_url = existing[0].source_url
        uploaded_at = existing[0].uploaded_at

        # Prefer full re-parse from the saved upload so structure-aware records replace
        # any older page-based chunks.
        from pathlib import Path

        upload_dir = Path(self.settings.upload_dir) / company_id
        saved = None
        if upload_dir.exists():
            matches = sorted(upload_dir.glob(f"{document_id}_*"))
            if matches:
                saved = matches[0]

        await self.store.delete_document(company_id, document_id)

        if saved and saved.exists() and source_type.value == "pdf":
            data = saved.read_bytes()
            return await self.upload_pdf(
                company_id=company_id,
                filename=document_name,
                data=data,
                replace_document_id=document_id,
            )

        if source_type.value == "website" and source_url:
            return await self.ingest_site(
                company_id=company_id,
                url=source_url,
                max_pages=self.settings.max_crawl_pages,
            )

        # Last resort: re-embed existing contents (legacy path).
        summary = DocumentSummary(
            document_id=document_id,
            company_id=company_id,
            document_name=document_name,
            source_type=source_type,
            source_url=source_url,
            page_count=max((chunk.page_number or 0) for chunk in existing),
            chunk_count=len(existing),
            embedding_count=0,
            status="reprocessing",
            uploaded_at=uploaded_at,
        )
        embedded = await self._embed_and_store(summary, existing)
        pages = [
            ExtractedPage(page_number=chunk.page_number or 1, text=chunk.content)
            for chunk in existing
        ]
        return UploadDocumentResponse(
            document=embedded,
            pages=pages,
            chunks=existing,
            message="Document re-embedded. Re-upload the PDF to rebuild structured records.",
        )

    async def _embed_and_store(
        self,
        summary: DocumentSummary,
        chunks: List[DocumentChunk],
    ) -> DocumentSummary:
        await self.store.ensure_collection(self.embeddings.dimensions)
        texts = [chunk.content for chunk in chunks]
        vectors = await self.embeddings.embed_texts(texts)
        count = await self.store.upsert_chunks(chunks, vectors)
        summary.embedding_count = count
        summary.chunk_count = len(chunks)
        summary.embedding_model = self.embeddings.model_name
        summary.embedding_dimension = self.embeddings.dimensions
        summary.status = "ready"
        logger.info(
            "Stored %s embeddings for document %s (%s)",
            count,
            summary.document_id,
            summary.document_name,
        )
        return summary
