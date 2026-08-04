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
from app.vector_store.payloads import chunk_from_payload, chunk_to_payload
from app.vector_store.scope import (
    dedup_key,
    normalize_document_scope,
    payload_dedup_key,
    payload_in_scope,
)

logger = logging.getLogger(__name__)


def _chunk_from_payload(payload: Dict[str, Any], *, company_id: str, document_id: str) -> DocumentChunk:
    return chunk_from_payload(payload, company_id=company_id, document_id=document_id)


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

        existing_hashes = {
            payload_dedup_key(point["payload"]) for point in self._points.values()
        }
        written = 0
        for chunk, vector in zip(chunks, vectors):
            if self._dimension and len(vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch for chunk {chunk.chunk_id}: "
                    f"expected {self._dimension}, got {len(vector)}"
                )
            key = dedup_key(
                chunk.company_id,
                chunk.content_hash,
                document_scope=chunk.document_scope,
                session_id=chunk.session_id,
            )
            if key in existing_hashes and chunk.chunk_id not in self._points:
                continue
            self._points[chunk.chunk_id] = {
                "vector": np.asarray(vector, dtype=np.float32),
                "payload": chunk_to_payload(chunk),
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
        session_id: Optional[str] = None,
        include_company_docs: bool = True,
        document_scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(query)) or 1.0
        scored: List[Dict[str, Any]] = []
        for point in self._points.values():
            payload = point["payload"]
            if not payload_in_scope(
                payload,
                company_id=company_id,
                session_id=session_id,
                include_company_docs=include_company_docs,
                document_scope=document_scope,
            ):
                continue
            vector = point["vector"]
            v_norm = float(np.linalg.norm(vector)) or 1.0
            score = float(np.dot(query, vector) / (q_norm * v_norm))
            scored.append({"score": score, "payload": payload})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    async def list_documents(
        self,
        company_id: str,
        *,
        session_id: Optional[str] = None,
        document_scope: Optional[str] = "company",
        include_company_docs: bool = True,
    ) -> List[DocumentSummary]:
        documents: Dict[str, DocumentSummary] = {}
        for point in self._points.values():
            payload = point["payload"]
            if not payload_in_scope(
                payload,
                company_id=company_id,
                session_id=session_id,
                include_company_docs=include_company_docs,
                document_scope=document_scope,
            ):
                continue
            document_id = str(payload["document_id"])
            if document_id not in documents:
                uploaded = payload.get("uploaded_at")
                scope = normalize_document_scope(payload.get("document_scope"))
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
                    primary_entities=[],
                    universal_chunk_count=0,
                    structured_chunk_count=0,
                    document_headings=[],
                    document_scope=scope,  # type: ignore[arg-type]
                    session_id=payload.get("session_id") if scope == "chat" else None,
                )
            summary = documents[document_id]
            summary.chunk_count += 1
            summary.embedding_count += 1
            page = payload.get("page_number")
            if isinstance(page, int):
                summary.page_count = max(summary.page_count, page)
            record_type = str(payload.get("record_type") or "universal")
            if record_type == "universal":
                summary.universal_chunk_count += 1
            else:
                summary.structured_chunk_count += 1
            person = payload.get("person_name")
            if person and str(person) not in summary.primary_entities:
                summary.primary_entities.append(str(person))
            heading = payload.get("section_title")
            if heading and str(heading) not in summary.document_headings:
                if len(summary.document_headings) < 12:
                    summary.document_headings.append(str(heading))
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
            chunks.append(
                _chunk_from_payload(payload, company_id=company_id, document_id=document_id)
            )
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    async def list_payloads(
        self,
        company_id: str,
        *,
        session_id: Optional[str] = None,
        include_company_docs: bool = True,
        document_scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            point["payload"]
            for point in self._points.values()
            if payload_in_scope(
                point["payload"],
                company_id=company_id,
                session_id=session_id,
                include_company_docs=include_company_docs,
                document_scope=document_scope,
            )
        ]
