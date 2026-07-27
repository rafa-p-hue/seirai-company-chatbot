from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.local_provider import create_embedding_provider
from app.generation.base import LLMProvider
from app.generation.qwen_provider import create_llm_provider
from app.retrieval.retriever import Retriever
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.vector_store.base import VectorStore
from app.vector_store.memory_store import InMemoryVectorStore
from app.vector_store.qdrant_store import QdrantVectorStore


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return create_embedding_provider(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    provider = settings.vector_store.lower().strip()
    if provider in {"memory", "inmemory", "local"}:
        return InMemoryVectorStore()
    if provider in {"qdrant"}:
        return QdrantVectorStore(settings)
    raise ValueError(f"Unsupported VECTOR_STORE: {settings.vector_store}")


@lru_cache
def get_llm_provider() -> LLMProvider:
    return create_llm_provider(get_settings())


def get_document_service() -> DocumentService:
    settings = get_settings()
    return DocumentService(
        settings=settings,
        embeddings=get_embedding_provider(),
        store=get_vector_store(),
    )


def get_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(
        settings=settings,
        embeddings=get_embedding_provider(),
        store=get_vector_store(),
    )


def get_chat_service() -> ChatService:
    return ChatService(retriever=get_retriever(), llm=get_llm_provider())
