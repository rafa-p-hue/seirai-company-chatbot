from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

DocumentScope = Literal["company", "chat"]
DocumentStatus = Literal["current", "archived", "draft", "unknown"]


class SourceType(str, Enum):
    pdf = "pdf"
    website = "website"
    html = "html"
    docx = "docx"
    markdown = "markdown"
    csv = "csv"
    pptx = "pptx"


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class ExtractedTableRow(BaseModel):
    table_index: int
    row_index: int
    heading: Optional[str] = None
    effective_date: Optional[str] = None
    column_labels: List[str] = Field(default_factory=list)
    cells: List[str] = Field(default_factory=list)
    row_label: Optional[str] = None
    human_text: str


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    table_rows: List[ExtractedTableRow] = Field(default_factory=list)
    slide_number: Optional[int] = None
    row_number: Optional[int] = None
    section_heading: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    company_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    row_number: Optional[int] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    chunk_index: int
    content: str
    content_hash: str
    source_type: SourceType = SourceType.pdf
    source_url: Optional[str] = None
    file_type: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: Optional[datetime] = None
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
    table_data: Optional[Dict[str, Any]] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    effective_date: Optional[str] = None
    version: Optional[str] = None
    document_status: DocumentStatus = "unknown"
    service_domain: Optional[str] = None
    document_scope: DocumentScope = "company"
    session_id: Optional[str] = None

    @property
    def filename(self) -> str:
        return self.document_name

    @property
    def section_heading(self) -> Optional[str]:
        return self.section_title

    @property
    def subsection_heading(self) -> Optional[str]:
        return self.subsection_title

    @property
    def clean_text(self) -> str:
        return self.content


class DocumentSummary(BaseModel):
    document_id: str
    company_id: str
    document_name: str
    source_type: SourceType
    source_url: Optional[str] = None
    file_type: Optional[str] = None
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
    document_scope: DocumentScope = "company"
    session_id: Optional[str] = None
    document_status: DocumentStatus = "unknown"
    service_domain: Optional[str] = None
    effective_date: Optional[str] = None
    version: Optional[str] = None


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
    session_id: Optional[str] = None
    include_company_docs: bool = True
    # Development-only: keep raw retrieval candidates without fact filters.
    skip_evidence_validation: bool = False


class RetrievedChunk(BaseModel):
    content: str
    document_name: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    row_number: Optional[int] = None
    source_url: Optional[str] = None
    score: float
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
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
    table_data: Optional[Dict[str, Any]] = None
    file_type: Optional[str] = None
    source_type: Optional[SourceType] = None
    effective_date: Optional[str] = None
    version: Optional[str] = None
    document_status: Optional[DocumentStatus] = None
    service_domain: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class RetrieveResponse(BaseModel):
    results: List[RetrievedChunk]
    original_query: Optional[str] = None
    expanded_query: Optional[str] = None
    query_type: Optional[str] = None
    expanded_terms: List[str] = Field(default_factory=list)
    resolved_query: Optional[str] = None
    subject_name: Optional[str] = None
    inspection: Optional[Dict[str, Any]] = None


class StoredChunkView(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    content_type: Optional[str] = None
    token_count: int = 0
    content: str
    chunk_index: int = 0


class DocumentChunksResponse(BaseModel):
    document_id: str
    company_id: str
    chunks: List[StoredChunkView] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    company_id: str
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = None
    include_company_docs: bool = True


class CitationSource(BaseModel):
    number: int
    document_name: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    row_number: Optional[int] = None
    section_title: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[SourceType] = None
    effective_date: Optional[str] = None
    document_status: Optional[DocumentStatus] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[CitationSource] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]
