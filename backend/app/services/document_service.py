from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.ingestion.formats import EmptyDocumentError, detect_format
from app.ingestion.pipeline import ingest_document_bytes, ingest_website
from app.models.api import (
    DocumentChunk,
    DocumentSummary,
    ExtractedPage,
    SourceType,
    UploadDocumentResponse,
)
from app.vector_store.base import VectorStore
from app.vector_store.scope import normalize_document_scope

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

    def _upload_path(self, company_id: str, document_id: str) -> Optional[Path]:
        upload_dir = Path(self.settings.upload_dir) / company_id
        if not upload_dir.exists():
            return None
        matches = sorted(upload_dir.glob(f"{document_id}_*"))
        return matches[0] if matches else None

    @staticmethod
    def _filename_from_upload(path: Path, document_id: str) -> str:
        prefix = f"{document_id}_"
        if path.name.startswith(prefix):
            return path.name[len(prefix) :]
        return path.name

    async def upload_document(
        self,
        *,
        company_id: str,
        filename: str,
        data: bytes,
        replace_document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        document_scope: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> UploadDocumentResponse:
        if len(data) > self.settings.max_upload_bytes:
            raise ValueError(
                f"File exceeds maximum size of {self.settings.max_upload_bytes} bytes."
            )
        if not data or not data.strip():
            raise EmptyDocumentError("Uploaded file is empty.")

        # Validate format early for clear errors.
        detect_format(filename, data, content_type)

        scope, scoped_session = self._resolve_scope(
            session_id=session_id, document_scope=document_scope
        )

        if replace_document_id:
            await self.store.delete_document(company_id, replace_document_id)

        summary, pages, chunks = await ingest_document_bytes(
            data=data,
            filename=filename,
            company_id=company_id,
            document_id=replace_document_id,
            document_scope=scope,
            session_id=scoped_session,
            content_type=content_type,
        )
        embedded = await self._embed_and_store(summary, chunks)
        return UploadDocumentResponse(
            document=embedded,
            pages=pages,
            chunks=chunks,
            message="Document ingested successfully.",
        )

    async def upload_pdf(
        self,
        *,
        company_id: str,
        filename: str,
        data: bytes,
        replace_document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        document_scope: Optional[str] = None,
    ) -> UploadDocumentResponse:
        """Backward-compatible PDF upload entry point."""
        return await self.upload_document(
            company_id=company_id,
            filename=filename,
            data=data,
            replace_document_id=replace_document_id,
            session_id=session_id,
            document_scope=document_scope,
            content_type="application/pdf",
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
            document_scope="company",
            session_id=None,
        )
        embedded = await self._embed_and_store(summary, chunks)
        return UploadDocumentResponse(
            document=embedded,
            pages=pages,
            chunks=chunks,
            message="Website ingested successfully.",
        )

    async def list_documents(
        self,
        company_id: str,
        *,
        session_id: Optional[str] = None,
        document_scope: Optional[str] = "company",
        include_company_docs: bool = True,
    ) -> List[DocumentSummary]:
        docs = await self.store.list_documents(
            company_id,
            session_id=session_id,
            document_scope=document_scope,
            include_company_docs=include_company_docs,
        )
        for doc in docs:
            doc.embedding_model = self.embeddings.model_name
            doc.embedding_dimension = self.embeddings.dimensions
        return docs

    async def delete_document(self, company_id: str, document_id: str) -> int:
        return await self.store.delete_document(company_id, document_id)

    async def reprocess_document(
        self,
        company_id: str,
        document_id: str,
        *,
        session_id: Optional[str] = None,
        document_name: Optional[str] = None,
        document_scope: Optional[str] = None,
    ) -> UploadDocumentResponse:
        """Re-ingest from the vector store and/or the on-disk upload file.

        Memory stores lose vectors on restart while upload files and chat
        attachment rows remain. When the store is empty, recover from disk
        using the attachment metadata (session_id / filename).
        """
        existing = await self.store.get_document_chunks(company_id, document_id)
        saved = self._upload_path(company_id, document_id)

        if not existing and not saved:
            raise ValueError(
                "Document not found in the vector index and no upload file on disk."
            )

        if existing:
            name = document_name or existing[0].document_name
            source_type = existing[0].source_type
            source_url = existing[0].source_url
            uploaded_at = existing[0].uploaded_at
            scope = normalize_document_scope(
                document_scope or existing[0].document_scope
            )
            scoped_session = (
                session_id
                if session_id is not None
                else (existing[0].session_id if scope == "chat" else None)
            )
        else:
            assert saved is not None
            name = document_name or self._filename_from_upload(saved, document_id)
            source_type = SourceType.pdf
            source_url = None
            uploaded_at = None
            scope = normalize_document_scope(
                document_scope or ("chat" if session_id else "company")
            )
            scoped_session = session_id if scope == "chat" else None

        if existing:
            await self.store.delete_document(company_id, document_id)

        if saved and saved.exists() and (
            not existing or source_type.value != "website"
        ):
            data = saved.read_bytes()
            return await self.upload_document(
                company_id=company_id,
                filename=name,
                data=data,
                replace_document_id=document_id,
                session_id=scoped_session,
                document_scope=scope,
            )

        if existing and source_type.value == "website" and source_url:
            return await self.ingest_site(
                company_id=company_id,
                url=source_url,
                max_pages=self.settings.max_crawl_pages,
            )

        if not existing:
            raise ValueError(
                f"Upload file missing for document {document_id}; cannot rebuild."
            )

        summary = DocumentSummary(
            document_id=document_id,
            company_id=company_id,
            document_name=name,
            source_type=source_type,
            source_url=source_url,
            page_count=max((chunk.page_number or 0) for chunk in existing),
            chunk_count=len(existing),
            embedding_count=0,
            status="reprocessing",
            uploaded_at=uploaded_at or existing[0].uploaded_at,
            document_scope=scope,  # type: ignore[arg-type]
            session_id=scoped_session,
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
            message="Document re-embedded. Re-upload the file to rebuild structured records.",
        )

    async def index_status_for_session(
        self,
        *,
        company_id: str,
        session_id: str,
        attachments: Sequence[Any],
    ) -> Dict[str, Any]:
        """Debug inventory: chat attachments vs searchable vector chunks."""
        rows: List[Dict[str, Any]] = []
        total_chunks = 0
        for attachment in attachments:
            document_id = getattr(attachment, "document_id", None) or attachment.get(
                "document_id"
            )
            document_name = getattr(attachment, "document_name", None) or attachment.get(
                "document_name"
            )
            status = getattr(attachment, "status", None) or attachment.get("status")
            deleted_at = getattr(attachment, "deleted_at", None)
            if isinstance(attachment, dict):
                deleted_at = attachment.get("deleted_at")
            chunks = await self.store.get_document_chunks(company_id, document_id)
            saved = self._upload_path(company_id, document_id)
            file_type = None
            if chunks:
                file_type = chunks[0].file_type or (
                    chunks[0].source_type.value if chunks[0].source_type else None
                )
            elif document_name and "." in document_name:
                file_type = document_name.rsplit(".", 1)[-1].lower()
            row = {
                "filename": document_name,
                "document_id": document_id,
                "session_id": session_id,
                "status": status,
                "deleted": deleted_at is not None,
                "document_scope": "chat",
                "chunk_count": len(chunks),
                "embedding_count": len(chunks),
                "file_type": file_type,
                "upload_file_present": bool(saved and saved.exists()),
                "upload_path": str(saved) if saved else None,
                "indexed": len(chunks) > 0,
            }
            total_chunks += len(chunks)
            rows.append(row)
        store_name = type(self.store).__name__
        collection = getattr(self.settings, "qdrant_collection", None) or "memory"
        return {
            "company_id": company_id,
            "session_id": session_id,
            "document_scope": "chat",
            "attachment_count": len(rows),
            "indexed_chunk_count": total_chunks,
            "embedding_model": self.embeddings.model_name,
            "embedding_dimension": self.embeddings.dimensions,
            "vector_store": self.settings.vector_store,
            "vector_collection": collection,
            "vector_store_class": store_name,
            "documents": rows,
        }

    async def rebuild_session_index(
        self,
        *,
        company_id: str,
        session_id: str,
        attachments: Sequence[Any],
    ) -> Dict[str, Any]:
        """Re-ingest every non-deleted chat attachment from disk into the vector store."""
        results: List[Dict[str, Any]] = []
        for attachment in attachments:
            document_id = getattr(attachment, "document_id", None) or attachment.get(
                "document_id"
            )
            document_name = getattr(attachment, "document_name", None) or attachment.get(
                "document_name"
            )
            deleted_at = getattr(attachment, "deleted_at", None)
            if isinstance(attachment, dict):
                deleted_at = attachment.get("deleted_at")
            if deleted_at is not None:
                continue
            try:
                response = await self.reprocess_document(
                    company_id,
                    document_id,
                    session_id=session_id,
                    document_name=document_name,
                    document_scope="chat",
                )
                results.append(
                    {
                        "document_id": document_id,
                        "filename": document_name,
                        "status": response.document.status,
                        "chunk_count": response.document.chunk_count,
                        "embedding_count": response.document.embedding_count,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Failed to rebuild %s (%s) for session %s",
                    document_name,
                    document_id,
                    session_id,
                )
                results.append(
                    {
                        "document_id": document_id,
                        "filename": document_name,
                        "status": "error",
                        "chunk_count": 0,
                        "embedding_count": 0,
                        "error": str(exc),
                    }
                )
        status = await self.index_status_for_session(
            company_id=company_id,
            session_id=session_id,
            attachments=attachments,
        )
        status["rebuild_results"] = results
        return status

    async def scan_indexed_phrases(
        self,
        *,
        company_id: str,
        session_id: Optional[str],
        phrases: Sequence[str],
    ) -> Dict[str, Any]:
        """Confirm whether exact phrases exist in searchable chunk text."""
        docs = await self.store.list_documents(
            company_id,
            session_id=session_id,
            document_scope="chat" if session_id else None,
            include_company_docs=not bool(session_id),
        )
        hits: Dict[str, List[Dict[str, str]]] = {phrase: [] for phrase in phrases}
        for doc in docs:
            chunks = await self.store.get_document_chunks(company_id, doc.document_id)
            for chunk in chunks:
                text = chunk.content or ""
                low = text.lower()
                for phrase in phrases:
                    if phrase.lower() in low:
                        hits[phrase].append(
                            {
                                "document_name": chunk.document_name,
                                "section_title": chunk.section_title or "",
                                "preview": " ".join(text.split())[:220],
                            }
                        )
        return {
            "company_id": company_id,
            "session_id": session_id,
            "phrases": {
                phrase: {"found": bool(matches), "matches": matches[:5]}
                for phrase, matches in hits.items()
            },
        }

    async def list_document_chunks(
        self, company_id: str, document_id: str
    ) -> List[DocumentChunk]:
        return await self.store.get_document_chunks(company_id, document_id)

    @staticmethod
    def _resolve_scope(
        *,
        session_id: Optional[str],
        document_scope: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        scoped_session = (session_id or "").strip() or None
        scope = (document_scope or "").strip().lower() or None
        if scoped_session:
            return "chat", scoped_session
        if scope == "chat":
            raise ValueError("session_id is required when document_scope is 'chat'.")
        return "company", None

    async def _embed_and_store(
        self, summary: DocumentSummary, chunks: List[DocumentChunk]
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
