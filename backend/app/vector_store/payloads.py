"""Serialize/deserialize DocumentChunk vector payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from app.models.api import DocumentChunk, SourceType
from app.vector_store.scope import chunk_payload_fields, normalize_document_scope


def chunk_to_payload(chunk: DocumentChunk) -> Dict[str, Any]:
    scope_fields = chunk_payload_fields(chunk)
    return {
        "chunk_id": chunk.chunk_id,
        "company_id": chunk.company_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "page_number": chunk.page_number,
        "slide_number": chunk.slide_number,
        "row_number": chunk.row_number,
        "section_title": chunk.section_title,
        "subsection_title": chunk.subsection_title,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "source_type": chunk.source_type.value,
        "source_url": chunk.source_url,
        "file_type": chunk.file_type or chunk.source_type.value,
        "uploaded_at": chunk.uploaded_at.isoformat(),
        "created_at": (chunk.created_at or chunk.uploaded_at).isoformat(),
        "record_id": chunk.record_id,
        "record_type": chunk.record_type,
        "person_name": chunk.person_name,
        "title": chunk.title,
        "organization": chunk.organization,
        "dates": chunk.dates,
        "location": chunk.location,
        "content_type": chunk.content_type,
        "label": chunk.label,
        "value": chunk.value,
        "table_data": chunk.table_data,
        "metadata_json": chunk.metadata_json or {},
        "effective_date": chunk.effective_date,
        "version": chunk.version,
        "document_status": chunk.document_status,
        "service_domain": chunk.service_domain,
        **scope_fields,
    }


def chunk_from_payload(
    payload: Dict[str, Any], *, company_id: str, document_id: str
) -> DocumentChunk:
    uploaded = payload.get("uploaded_at") or payload.get("created_at")
    created = payload.get("created_at") or uploaded
    scope = normalize_document_scope(payload.get("document_scope"))
    source = payload.get("source_type") or "pdf"
    try:
        source_type = SourceType(source)
    except ValueError:
        source_type = SourceType.pdf
    return DocumentChunk(
        chunk_id=str(payload.get("chunk_id")),
        company_id=company_id,
        document_id=document_id,
        document_name=str(payload.get("document_name")),
        page_number=payload.get("page_number"),
        slide_number=payload.get("slide_number"),
        row_number=payload.get("row_number"),
        section_title=payload.get("section_title"),
        subsection_title=payload.get("subsection_title"),
        chunk_index=int(payload.get("chunk_index") or 0),
        content=str(payload.get("content") or ""),
        content_hash=str(payload.get("content_hash") or ""),
        source_type=source_type,
        source_url=payload.get("source_url"),
        file_type=payload.get("file_type") or source_type.value,
        uploaded_at=datetime.fromisoformat(uploaded)
        if uploaded
        else datetime.utcnow(),
        created_at=datetime.fromisoformat(created) if created else None,
        record_id=payload.get("record_id"),
        record_type=payload.get("record_type"),
        person_name=payload.get("person_name"),
        title=payload.get("title"),
        organization=payload.get("organization"),
        dates=payload.get("dates"),
        location=payload.get("location"),
        content_type=payload.get("content_type"),
        label=payload.get("label"),
        value=payload.get("value"),
        table_data=payload.get("table_data"),
        metadata_json=payload.get("metadata_json") or {},
        effective_date=payload.get("effective_date"),
        version=payload.get("version"),
        document_status=payload.get("document_status") or "unknown",
        service_domain=payload.get("service_domain"),
        document_scope=scope,  # type: ignore[arg-type]
        session_id=payload.get("session_id") if scope == "chat" else None,
    )
