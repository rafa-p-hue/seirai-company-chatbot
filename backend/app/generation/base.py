from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from app.models.api import ChatMessage, CitationSource, RetrievedChunk


class LLMProvider(ABC):
    name: str
    model_name: str

    @abstractmethod
    async def generate(
        self,
        *,
        question: str,
        evidence: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
    ) -> tuple[str, List[CitationSource]]:
        raise NotImplementedError
