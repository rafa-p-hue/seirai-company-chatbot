from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from app.models.api import DocumentChunk, DocumentSummary


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self, dimension: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        *,
        company_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def list_documents(self, company_id: str) -> List[DocumentSummary]:
        raise NotImplementedError

    @abstractmethod
    async def delete_document(self, company_id: str, document_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    async def get_document_chunks(
        self, company_id: str, document_id: str
    ) -> List[DocumentChunk]:
        raise NotImplementedError
