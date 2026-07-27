from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingProvider(ABC):
    name: str
    model_name: str
    dimensions: int

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]
