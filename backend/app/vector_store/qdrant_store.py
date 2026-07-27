from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings
from app.models.api import DocumentChunk, DocumentSummary, SourceType
from app.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


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
            return

        logger.info("Creating Qdrant collection %s (dim=%s)", self.collection, dimension)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="company_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="document_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="content_hash",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
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

        # Skip chunks whose content_hash already exists for this company.
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
            if (chunk.company_id, chunk.content_hash) in existing_hashes:
                logger.info(
                    "Skipping duplicate chunk hash %s for company %s",
                    chunk.content_hash[:12],
                    chunk.company_id,
                )
                continue
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=list(vector),
                    payload={
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
                )
            )
            existing_hashes.add((chunk.company_id, chunk.content_hash))

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
                    with_payload=["company_id", "content_hash"],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    hashes.add(
                        (
                            str(payload.get("company_id")),
                            str(payload.get("content_hash")),
                        )
                    )
                if offset is None:
                    break
        return hashes

    async def search(
        self,
        *,
        company_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=list(query_vector),
            limit=top_k,
            query_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="company_id",
                        match=qmodels.MatchValue(value=company_id),
                    )
                ]
            ),
            with_payload=True,
        )
        output: List[Dict[str, Any]] = []
        for point in results:
            payload = point.payload or {}
            output.append(
                {
                    "score": float(point.score),
                    "payload": payload,
                }
            )
        return output

    async def list_documents(self, company_id: str) -> List[DocumentSummary]:
        documents: Dict[str, DocumentSummary] = {}
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
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                document_id = str(payload.get("document_id"))
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
            if offset is None:
                break
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        hex_id = "".join(ch for ch in chunk_id.lower() if ch in "0123456789abcdef")
        hex_id = (hex_id + "0" * 32)[:32]
        return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:32]}"
