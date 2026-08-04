from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings
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


class QdrantVectorStore(VectorStore):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.request_timeout_seconds,
        )
        self._dimension: Optional[int] = None

    async def ensure_collection(self, dimension: int) -> None:
        self._dimension = dimension
        names = {collection.name for collection in self.client.get_collections().collections}
        if self.collection in names:
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors.size
            if existing != dimension:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' expects dimension {existing}, "
                    f"but embeddings are {dimension}."
                )
            self._ensure_scope_indexes()
            return

        logger.info("Creating Qdrant collection %s (dim=%s)", self.collection, dimension)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        for field_name in (
            "company_id",
            "document_id",
            "content_hash",
            "document_scope",
            "session_id",
        ):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def _ensure_scope_indexes(self) -> None:
        for field_name in ("document_scope", "session_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001
                # Index may already exist on older collections.
                logger.debug("Payload index %s may already exist", field_name)

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if self._dimension is None and vectors:
            await self.ensure_collection(len(vectors[0]))

        existing_hashes = self._existing_content_hashes(
            {chunk.company_id for chunk in chunks}
        )

        points = []
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
            if key in existing_hashes:
                logger.info(
                    "Skipping duplicate chunk hash %s for company %s scope=%s session=%s",
                    chunk.content_hash[:12],
                    chunk.company_id,
                    chunk.document_scope,
                    chunk.session_id,
                )
                continue
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=list(vector),
                    payload=chunk_to_payload(chunk),
                )
            )
            existing_hashes.add(key)

        if not points:
            return 0

        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def _existing_content_hashes(self, company_ids: set) -> set:
        hashes = set()
        for company_id in company_ids:
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="company_id",
                                match=qmodels.MatchValue(value=company_id),
                            )
                        ]
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=[
                        "company_id",
                        "content_hash",
                        "document_scope",
                        "session_id",
                    ],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    hashes.add(payload_dedup_key(payload))
                if offset is None:
                    break
        return hashes

    @staticmethod
    def _company_filter(company_id: str) -> qmodels.Filter:
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="company_id",
                    match=qmodels.MatchValue(value=company_id),
                )
            ]
        )

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
        # Over-fetch then apply scope filter so legacy payloads without
        # document_scope still resolve as company-scoped.
        fetch_k = max(top_k * 4, top_k, 32)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=list(query_vector),
            limit=fetch_k,
            query_filter=self._company_filter(company_id),
            with_payload=True,
        )
        output: List[Dict[str, Any]] = []
        for point in results:
            payload = point.payload or {}
            if not payload_in_scope(
                payload,
                company_id=company_id,
                session_id=session_id,
                include_company_docs=include_company_docs,
                document_scope=document_scope,
            ):
                continue
            output.append({"score": float(point.score), "payload": payload})
            if len(output) >= top_k:
                break
        return output

    async def list_documents(
        self,
        company_id: str,
        *,
        session_id: Optional[str] = None,
        document_scope: Optional[str] = "company",
        include_company_docs: bool = True,
    ) -> List[DocumentSummary]:
        documents: Dict[str, DocumentSummary] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._company_filter(company_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if not payload_in_scope(
                    payload,
                    company_id=company_id,
                    session_id=session_id,
                    include_company_docs=include_company_docs,
                    document_scope=document_scope,
                ):
                    continue
                document_id = str(payload.get("document_id"))
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
            if offset is None:
                break
        return sorted(documents.values(), key=lambda item: item.uploaded_at, reverse=True)

    async def delete_document(self, company_id: str, document_id: str) -> int:
        before = await self.get_document_chunks(company_id, document_id)
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="company_id",
                            match=qmodels.MatchValue(value=company_id),
                        ),
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        ),
                    ]
                )
            ),
        )
        return len(before)

    async def get_document_chunks(
        self, company_id: str, document_id: str
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="company_id",
                            match=qmodels.MatchValue(value=company_id),
                        ),
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        ),
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunks.append(
                    _chunk_from_payload(
                        payload, company_id=company_id, document_id=document_id
                    )
                )
            if offset is None:
                break
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    async def list_payloads(
        self,
        company_id: str,
        *,
        session_id: Optional[str] = None,
        include_company_docs: bool = True,
        document_scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._company_filter(company_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if payload_in_scope(
                    payload,
                    company_id=company_id,
                    session_id=session_id,
                    include_company_docs=include_company_docs,
                    document_scope=document_scope,
                ):
                    payloads.append(payload)
            if offset is None:
                break
        return payloads

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        hex_id = "".join(ch for ch in chunk_id.lower() if ch in "0123456789abcdef")
        hex_id = (hex_id + "0" * 32)[:32]
        return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:32]}"
