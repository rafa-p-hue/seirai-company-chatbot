"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    app_name: str = Field(default="seirai-rag-backend", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(
        default=(
            "http://localhost:3000,http://localhost:3001,"
            "http://127.0.0.1:3000,http://127.0.0.1:3001"
        ),
        alias="CORS_ORIGINS",
    )
    api_prefix: str = Field(default="/api", alias="API_PREFIX")
    database_url: str = Field(
        default="sqlite:///./chat_history.db",
        alias="DATABASE_URL",
    )
    chat_history_message_limit: int = Field(
        default=10,
        ge=8,
        le=12,
        alias="CHAT_HISTORY_MESSAGE_LIMIT",
    )
    request_timeout_seconds: int = Field(default=60, alias="REQUEST_TIMEOUT_SECONDS")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    default_company_id: str = Field(default="seirai", alias="DEFAULT_COMPANY_ID")
    not_found_contact_name: str = Field(
        default="",
        alias="NOT_FOUND_CONTACT_NAME",
    )
    allowed_crawl_domains: str = Field(default="", alias="ALLOWED_CRAWL_DOMAINS")
    max_crawl_pages: int = Field(default=20, alias="MAX_CRAWL_PAGES")

    vector_store: str = Field(default="qdrant", alias="VECTOR_STORE")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="company_chunks", alias="QDRANT_COLLECTION")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_base_url: str = Field(default="http://localhost:11434", alias="LLM_BASE_URL")
    llm_model: str = Field(default="qwen2.5:7b", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=800, alias="LLM_MAX_TOKENS")

    hf_token: str = Field(default="", alias="HF_TOKEN")

    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_candidate_k: int = Field(default=20, alias="RETRIEVAL_CANDIDATE_K")
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")

    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in {"development", "dev", "local", "test"}

    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def crawl_domain_allowlist(self) -> List[str]:
        return [
            domain.strip().lower()
            for domain in self.allowed_crawl_domains.split(",")
            if domain.strip()
        ]

    @field_validator("api_prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/") or "/api"


@lru_cache
def get_settings() -> Settings:
    return Settings()
