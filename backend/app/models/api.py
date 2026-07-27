from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourceType(str, Enum):
    pdf = "pdf"
    website = "website"


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class ExtractedPage(BaseModel):
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    chunk_id: str
    company_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    chunk_index: int
    content: str
    content_hash: str
    source_type: SourceType = SourceType.pdf
    source_url: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    # Structure-aware record metadata
    record_id: Optional[str] = None
    record_type: Optional[str] = None
    person_name: Optional[str] = None
    title: Optional[str] = None
    organization: Optional[str] = None
    dates: Optional[str] = None
    location: Optional[str] = None
    # Universal content typing (key_value, paragraph, list, heading, prompt, …)
    content_type: Optional[str] = None
    label: Optional[str] = None
    value: Optional[str] = None


class DocumentSummary(BaseModel):
    document_id: str
    company_id: str
    document_name: str
    source_type: SourceType
    source_url: Optional[str] = None
    page_count: int = 0
    chunk_count: int = 0
    embedding_count: int = 0
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    status: str = "ready"
    uploaded_at: datetime
    errors: List[str] = Field(default_factory=list)
    primary_entities: List[str] = Field(default_factory=list)
    universal_chunk_count: int = 0
    structured_chunk_count: int = 0
    document_headings: List[str] = Field(default_factory=list)


class UploadDocumentResponse(BaseModel):
    document: DocumentSummary
    pages: List[ExtractedPage]
    chunks: List[DocumentChunk]
    message: str


class WebsiteIngestRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=64)
    url: str
    max_pages: int = Field(default=20, ge=1, le=100)

    @field_validator("company_id")
    @classmethod
    def validate_company_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned.replace("-", "").replace("_", "").isalnum():
            raise ValueError("company_id must be alphanumeric with optional - or _")
        return cleaned


class RetrieveRequest(BaseModel):
    company_id: str
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    content: str
    document_name: str
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    score: float
    chunk_id: Optional[str] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    record_id: Optional[str] = None
    record_type: Optional[str] = None
    person_name: Optional[str] = None
    title: Optional[str] = None
    organization: Optional[str] = None
    dates: Optional[str] = None
    location: Optional[str] = None
    content_type: Optional[str] = None
    label: Optional[str] = None
    value: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class RetrieveResponse(BaseModel):
    results: List[RetrievedChunk]
    original_query: Optional[str] = None
    expanded_query: Optional[str] = None
    query_type: Optional[str] = None
    expanded_terms: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    company_id: str
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationSource(BaseModel):
    number: int
    document_name: str
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    source_type: Optional[SourceType] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[CitationSource] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
