"""In-memory vector store for local development when Qdrant is unavailable.

Labeled fallback — not for production. Enable with:
  VECTOR_STORE=memory
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.models.api import DocumentChunk, DocumentSummary, SourceType
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._dimension: Optional[int] = None
        self._points: Dict[str, Dict[str, Any]] = {}
        logger.warning(
            "Using in-memory vector store (VECTOR_STORE=memory). "
            "Data is lost on restart and is not shared across processes."
        )

    async def ensure_collection(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension
            return
        if self._dimension != dimension:
            raise ValueError(
                f"In-memory store expects dimension {self._dimension}, got {dimension}."
            )

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if self._dimension is None and vectors:
            await self.ensure_collection(len(vectors[0]))

        # Skip exact content-hash duplicates for the same company/document.
        existing_hashes = {
            (point["payload"]["company_id"], point["payload"]["content_hash"])
            for point in self._points.values()
        }
        written = 0
        for chunk, vector in zip(chunks, vectors):
            if self._dimension and len(vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch for chunk {chunk.chunk_id}: "
                    f"expected {self._dimension}, got {len(vector)}"
                )
            key = (chunk.company_id, chunk.content_hash)
            if key in existing_hashes and chunk.chunk_id not in self._points:
                continue
            self._points[chunk.chunk_id] = {
                "vector": np.asarray(vector, dtype=np.float32),
                "payload": {
                    "chunk_id": chunk.chunk_id,
                    "company_id": chunk.company_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "source_type": chunk.source_type.value,
                    "source_url": chunk.source_url,
                    "uploaded_at": chunk.uploaded_at.isoformat(),
                    "record_id": chunk.record_id,
                    "record_type": chunk.record_type,
                    "person_name": chunk.person_name,
                    "title": chunk.title,
                    "organization": chunk.organization,
                    "dates": chunk.dates,
                    "location": chunk.location,
                },
            }
            existing_hashes.add(key)
            written += 1
        return written

    async def search(
        self,
        *,
        company_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(query)) or 1.0
        scored: List[Dict[str, Any]] = []
        for point in self._points.values():
            payload = point["payload"]
            if payload["company_id"] != company_id:
                continue
            vector = point["vector"]
            v_norm = float(np.linalg.norm(vector)) or 1.0
            score = float(np.dot(query, vector) / (q_norm * v_norm))
            scored.append({"score": score, "payload": payload})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def list_documents(self, company_id: str) -> List[DocumentSummary]:
        documents: Dict[str, DocumentSummary] = {}
        for point in self._points.values():
            payload = point["payload"]
            if payload["company_id"] != company_id:
                continue
            document_id = str(payload["document_id"])
            if document_id not in documents:
                uploaded = payload.get("uploaded_at")
                documents[document_id] = DocumentSummary(
                    document_id=document_id,
                    company_id=company_id,
                    document_name=str(payload.get("document_name") or document_id),
                    source_type=SourceType(payload.get("source_type") or "pdf"),
                    source_url=payload.get("source_url"),
                    page_count=0,
                    chunk_count=0,
                    embedding_count=0,
                    status="ready",
                    uploaded_at=datetime.fromisoformat(uploaded)
                    if uploaded
                    else datetime.utcnow(),
                )
            summary = documents[document_id]
            summary.chunk_count += 1
            summary.embedding_count += 1
            page = payload.get("page_number")
            if isinstance(page, int):
                summary.page_count = max(summary.page_count, page)
        return sorted(documents.values(), key=lambda item: item.uploaded_at, reverse=True)

    async def delete_document(self, company_id: str, document_id: str) -> int:
        to_delete = [
            key
            for key, point in self._points.items()
            if point["payload"]["company_id"] == company_id
            and point["payload"]["document_id"] == document_id
        ]
        for key in to_delete:
            del self._points[key]
        return len(to_delete)

    async def get_document_chunks(
        self, company_id: str, document_id: str
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for point in self._points.values():
            payload = point["payload"]
            if (
                payload["company_id"] != company_id
                or payload["document_id"] != document_id
            ):
                continue
            uploaded = payload.get("uploaded_at")
            chunks.append(
                DocumentChunk(
                    chunk_id=str(payload.get("chunk_id")),
                    company_id=company_id,
                    document_id=document_id,
                    document_name=str(payload.get("document_name")),
                    page_number=payload.get("page_number"),
                    section_title=payload.get("section_title"),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    content=str(payload.get("content") or ""),
                    content_hash=str(payload.get("content_hash") or ""),
                    source_type=SourceType(payload.get("source_type") or "pdf"),
                    source_url=payload.get("source_url"),
                    uploaded_at=datetime.fromisoformat(uploaded)
                    if uploaded
                    else datetime.utcnow(),
                    record_id=payload.get("record_id"),
                    record_type=payload.get("record_type"),
                    person_name=payload.get("person_name"),
                    title=payload.get("title"),
                    organization=payload.get("organization"),
                    dates=payload.get("dates"),
                    location=payload.get("location"),
                )
            )
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)
