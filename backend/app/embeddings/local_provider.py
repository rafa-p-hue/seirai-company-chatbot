from __future__ import annotations

import asyncio
import logging
from typing import List, Sequence

import numpy as np

from app.config import Settings
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Real local sentence-embedding provider via sentence-transformers.

    Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dims).
    First use downloads the model into the local Hugging Face cache.
    """

    def __init__(self, settings: Settings) -> None:
        self.name = "local-sentence-transformers"
        self.model_name = settings.embedding_model
        self.dimensions = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            import os
            from pathlib import Path

            from sentence_transformers import SentenceTransformer

            # Avoid broken local proxies from the IDE sandbox when loading cached models.
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
            ):
                os.environ.pop(key, None)

            logger.info("Loading embedding model: %s", self.model_name)
            model_ref = self.model_name
            cache_snapshots = (
                Path.home()
                / ".cache/huggingface/hub"
                / f"models--{self.model_name.replace('/', '--')}"
                / "snapshots"
            )
            if cache_snapshots.is_dir():
                snaps = sorted(p for p in cache_snapshots.iterdir() if p.is_dir())
                if snaps:
                    model_ref = str(snaps[-1])

            try:
                self._model = SentenceTransformer(
                    model_ref,
                    local_files_only=True,
                )
            except Exception as exc:
                logger.warning(
                    "Local cache load failed for %s (%s); retrying with download enabled",
                    self.model_name,
                    exc,
                )
                self._model = SentenceTransformer(self.model_name)

            actual_dim = int(self._model.get_sentence_embedding_dimension())
            if actual_dim != self.dimensions:
                logger.warning(
                    "Configured EMBEDDING_DIMENSION=%s but model reports %s; using model dimension",
                    self.dimensions,
                    actual_dim,
                )
                self.dimensions = actual_dim
        return self._model

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        def _encode() -> List[List[float]]:
            model = self._load_model()
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            array = np.asarray(vectors, dtype=np.float32)
            return [row.tolist() for row in array]

        return await asyncio.to_thread(_encode)


class HashingFallbackEmbeddingProvider(EmbeddingProvider):
    """Explicit fallback only — not the default.

    Used in tests or environments where sentence-transformers cannot load.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.name = "hashing-fallback"
        self.model_name = "hashing-fallback-v1"
        self.dimensions = dimensions

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = [token.lower() for token in text.split() if len(token) > 1]
        for index, token in enumerate(tokens):
            self._add(vector, token, 1.0)
            if index + 1 < len(tokens):
                self._add(vector, f"{token}_{tokens[index + 1]}", 1.4)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def _add(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = abs(hash(feature))
        vector[digest % self.dimensions] += weight


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()
    if provider in {"local", "sentence-transformers", "st"}:
        return LocalSentenceTransformerProvider(settings)
    if provider in {"hash", "hashing", "fallback"}:
        return HashingFallbackEmbeddingProvider(settings.embedding_dimension)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
