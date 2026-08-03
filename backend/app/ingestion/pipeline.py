from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.ingestion.chunker import new_document_id
from app.ingestion.cleaner import clean_pages
from app.ingestion.csv_loader import extract_csv_bytes
from app.ingestion.document_status import detect_document_lifecycle
from app.ingestion.docx_loader import extract_docx_bytes
from app.ingestion.entity_detect import detect_document_headings, detect_primary_entities
from app.ingestion.formats import (
    EmptyDocumentError,
    PasswordProtectedError,
    UnsupportedFormatError,
    detect_format,
)
from app.ingestion.html_loader import extract_html_bytes
from app.ingestion.linkedin_clean import clean_linkedin_pages
from app.ingestion.markdown_loader import extract_markdown_bytes
from app.ingestion.pdf_loader import PdfExtractionError, extract_pdf_pages
from app.ingestion.pptx_loader import extract_pptx_bytes
from app.ingestion.service_domain import (
    detect_service_domain,
    extract_chunk_procedure_metadata,
)
from app.ingestion.structured_records import parse_structured_records
from app.ingestion.universal_chunker import create_universal_chunks
from app.ingestion.web_loader import crawl_website
from app.models.api import DocumentChunk, DocumentSummary, ExtractedPage, SourceType

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w.\- ]+", "_", name).strip()
    return name[:180] or "document.pdf"


async def ingest_document_bytes(
    *,
    data: bytes,
    filename: str,
    company_id: str,
    document_id: Optional[str] = None,
    document_scope: str = "company",
    session_id: Optional[str] = None,
    content_type: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Tuple[DocumentSummary, List[ExtractedPage], List[DocumentChunk]]:
    """Parse any supported upload format into the shared chunk schema."""
    safe_name = sanitize_filename(filename)
    document_id = document_id or new_document_id()
    if session_id:
        document_scope = "chat"
    elif document_scope not in {"company", "chat"}:
        document_scope = "company"
    if document_scope != "chat":
        session_id = None

    fmt = detect_format(safe_name, data, content_type)
    temp_dir = Path("data/uploads") / company_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{document_id}_{safe_name}"
    path.write_bytes(data)

    try:
        pages, metadata = _extract_pages_for_format(
            fmt.file_type,
            data=data,
            path=path,
            filename=safe_name,
            source_url=source_url,
        )
    except (PdfExtractionError, EmptyDocumentError, PasswordProtectedError, UnsupportedFormatError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse {fmt.file_type.upper()} file: {exc}") from exc

    if fmt.file_type == "pdf":
        cleaned = clean_linkedin_pages(clean_pages(pages))
    else:
        cleaned = clean_pages(pages)

    lifecycle = detect_document_lifecycle(
        filename=safe_name, pages=cleaned, metadata=metadata
    )
    headings = detect_document_headings(cleaned)
    service_domain = detect_service_domain(
        filename=safe_name, pages=cleaned, headings=headings
    )
    chunks = _build_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=safe_name,
        source_type=fmt.source_type,
        source_url=source_url,
        file_type=fmt.file_type,
        lifecycle=lifecycle,
        service_domain=service_domain,
    )
    _apply_document_scope(chunks, document_scope=document_scope, session_id=session_id)
    _enrich_chunks_metadata(chunks, service_domain=service_domain)
    if not chunks:
        raise EmptyDocumentError(
            f"No useful text chunks could be created from this {fmt.file_type.upper()} file."
        )

    entities = detect_primary_entities(
        pages=cleaned, document_name=safe_name, chunks=chunks
    )
    logger.info(
        "Ingested %s (%s): %s chunks, status=%s domain=%s",
        safe_name,
        fmt.file_type,
        len(chunks),
        lifecycle.document_status,
        service_domain,
    )
    summary = DocumentSummary(
        document_id=document_id,
        company_id=company_id,
        document_name=safe_name,
        source_type=fmt.source_type,
        source_url=source_url,
        file_type=fmt.file_type,
        page_count=int(metadata.get("page_count") or len(cleaned)),
        chunk_count=len(chunks),
        embedding_count=0,
        status="chunked",
        uploaded_at=chunks[0].uploaded_at,
        errors=[],
        primary_entities=entities,
        universal_chunk_count=sum(
            1 for chunk in chunks if chunk.record_type == "universal"
        ),
        structured_chunk_count=sum(
            1 for chunk in chunks if chunk.record_type != "universal"
        ),
        document_headings=headings,
        document_scope=document_scope,  # type: ignore[arg-type]
        session_id=session_id,
        document_status=lifecycle.document_status,
        service_domain=service_domain,
        effective_date=lifecycle.effective_date,
        version=lifecycle.version,
    )
    return summary, cleaned, chunks


async def ingest_pdf_bytes(
    *,
    data: bytes,
    filename: str,
    company_id: str,
    document_id: Optional[str] = None,
    document_scope: str = "company",
    session_id: Optional[str] = None,
) -> Tuple[DocumentSummary, List[ExtractedPage], List[DocumentChunk]]:
    return await ingest_document_bytes(
        data=data,
        filename=filename if filename.lower().endswith(".pdf") else f"{filename}.pdf",
        company_id=company_id,
        document_id=document_id,
        document_scope=document_scope,
        session_id=session_id,
        content_type="application/pdf",
    )


async def ingest_website(
    *,
    company_id: str,
    url: str,
    max_pages: int,
    allowlist: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    document_scope: str = "company",
    session_id: Optional[str] = None,
) -> Tuple[DocumentSummary, List[ExtractedPage], List[DocumentChunk]]:
    document_id = document_id or new_document_id()
    if session_id:
        document_scope = "chat"
    elif document_scope not in {"company", "chat"}:
        document_scope = "company"
    if document_scope != "chat":
        session_id = None

    pages = await crawl_website(url, max_pages=max_pages, allowlist=allowlist)
    cleaned = clean_pages(pages)
    document_name = url.rstrip("/").split("/")[-1] or "website"
    lifecycle = detect_document_lifecycle(filename=document_name, pages=cleaned)
    headings = detect_document_headings(cleaned)
    service_domain = detect_service_domain(
        filename=document_name, pages=cleaned, headings=headings
    )

    chunks = _build_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=document_name,
        source_type=SourceType.website,
        source_url=url,
        file_type="website",
        lifecycle=lifecycle,
        service_domain=service_domain,
    )
    _apply_document_scope(chunks, document_scope=document_scope, session_id=session_id)
    _enrich_chunks_metadata(chunks, service_domain=service_domain)
    if not chunks:
        raise ValueError("No useful chunks could be created from this website.")

    entities = detect_primary_entities(
        pages=cleaned, document_name=document_name, chunks=chunks
    )
    summary = DocumentSummary(
        document_id=document_id,
        company_id=company_id,
        document_name=document_name,
        source_type=SourceType.website,
        source_url=url,
        file_type="website",
        page_count=len(cleaned),
        chunk_count=len(chunks),
        embedding_count=0,
        status="chunked",
        uploaded_at=chunks[0].uploaded_at,
        errors=[],
        primary_entities=entities,
        universal_chunk_count=sum(
            1 for chunk in chunks if chunk.record_type == "universal"
        ),
        structured_chunk_count=sum(
            1 for chunk in chunks if chunk.record_type != "universal"
        ),
        document_headings=headings,
        document_scope=document_scope,  # type: ignore[arg-type]
        session_id=session_id,
        document_status=lifecycle.document_status,
        service_domain=service_domain,
        effective_date=lifecycle.effective_date,
        version=lifecycle.version,
    )
    return summary, cleaned, chunks


def _extract_pages_for_format(
    file_type: str,
    *,
    data: bytes,
    path: Path,
    filename: str,
    source_url: Optional[str],
) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
    if file_type == "pdf":
        try:
            return extract_pdf_pages(path)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "password" in message or "encrypted" in message:
                raise PasswordProtectedError(
                    "This PDF appears password-protected and cannot be processed."
                ) from exc
            raise
    if file_type == "html":
        return extract_html_bytes(data, filename=filename, source_url=source_url)
    if file_type == "docx":
        return extract_docx_bytes(data, filename=filename)
    if file_type == "markdown":
        return extract_markdown_bytes(data, filename=filename)
    if file_type == "csv":
        return extract_csv_bytes(data, filename=filename)
    if file_type == "pptx":
        return extract_pptx_bytes(data, filename=filename)
    raise UnsupportedFormatError(
        "Unsupported file format. Supported formats: PDF, DOCX, HTML, MD, CSV, PPTX."
    )


def _build_chunks(
    *,
    pages: List[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType,
    source_url: Optional[str],
    file_type: str,
    lifecycle,
    service_domain: Optional[str] = None,
) -> List[DocumentChunk]:
    universal = create_universal_chunks(
        pages=pages,
        company_id=company_id,
        document_id=document_id,
        document_name=document_name,
        source_type=source_type,
        source_url=source_url,
        file_type=file_type,
        document_status=lifecycle.document_status,
        effective_date=lifecycle.effective_date,
        version=lifecycle.version,
        metadata_json=lifecycle.metadata or {},
        service_domain=service_domain,
    )
    table_structured = create_structured_table_chunks(
        pages=pages,
        company_id=company_id,
        document_id=document_id,
        document_name=document_name,
        source_type=source_type,
        source_url=source_url,
        file_type=file_type,
        document_status=lifecycle.document_status,
        effective_date=lifecycle.effective_date,
        version=lifecycle.version,
        chunk_index_offset=len(universal),
        service_domain=service_domain,
    )
    structured = table_structured
    if source_type in {SourceType.pdf, SourceType.website, SourceType.html, SourceType.docx}:
        structured = table_structured + optional_structured_chunks(
            pages=pages,
            company_id=company_id,
            document_id=document_id,
            document_name=document_name,
            source_type=source_type,
            source_url=source_url,
            file_type=file_type,
            document_status=lifecycle.document_status,
            chunk_index_offset=len(universal) + len(table_structured),
            service_domain=service_domain,
        )
    return merge_chunks(universal, structured)


def _apply_document_scope(
    chunks: List[DocumentChunk],
    *,
    document_scope: str,
    session_id: Optional[str],
) -> None:
    for chunk in chunks:
        chunk.document_scope = document_scope  # type: ignore[assignment]
        chunk.session_id = session_id if document_scope == "chat" else None


def _enrich_chunks_metadata(
    chunks: List[DocumentChunk], *, service_domain: Optional[str]
) -> None:
    for chunk in chunks:
        if service_domain and not chunk.service_domain:
            chunk.service_domain = service_domain
        meta = dict(chunk.metadata_json or {})
        if service_domain:
            meta.setdefault("service_domain", service_domain)
        procedure = extract_chunk_procedure_metadata(
            content=chunk.content,
            section_title=chunk.section_title,
            subsection_title=chunk.subsection_title,
        )
        for key, value in procedure.items():
            if value and key not in meta:
                meta[key] = value
        if procedure.get("responsible_office") and not chunk.organization:
            chunk.organization = procedure["responsible_office"]
        if procedure.get("deadline") and not chunk.dates:
            chunk.dates = procedure["deadline"]
        chunk.metadata_json = meta


def optional_structured_chunks(
    *,
    pages: List[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType = SourceType.pdf,
    source_url: Optional[str] = None,
    file_type: Optional[str] = None,
    document_status: str = "unknown",
    chunk_index_offset: int = 0,
    service_domain: Optional[str] = None,
) -> List[DocumentChunk]:
    """Optional enhancement — never the only retrieval source."""
    try:
        records = parse_structured_records(pages, document_id=document_id)
    except Exception:  # noqa: BLE001
        logger.exception("Structured parsing failed; continuing with universal chunks")
        return []

    uploaded_at = datetime.utcnow()
    chunks: List[DocumentChunk] = []
    for offset, record in enumerate(records):
        content = record.embedding_text()
        if len(content.strip()) < 20:
            continue
        if record.record_type in {"section", "unknown"}:
            continue
        content_hash = hashlib.sha256(
            f"{company_id}:{document_id}:structured:{record.record_id}:{content}".encode(
                "utf-8"
            )
        ).hexdigest()
        chunks.append(
            DocumentChunk(
                chunk_id=record.record_id,
                company_id=company_id,
                document_id=document_id,
                document_name=document_name,
                page_number=record.source_page,
                section_title=record.heading or record.record_type,
                subsection_title=None,
                chunk_index=chunk_index_offset + offset,
                content=content,
                content_hash=content_hash,
                source_type=source_type,
                source_url=source_url,
                file_type=file_type or source_type.value,
                uploaded_at=uploaded_at,
                created_at=uploaded_at,
                record_id=record.record_id,
                record_type=record.record_type,
                person_name=record.person_name,
                title=record.title,
                organization=record.organization,
                dates=record.dates,
                location=record.location,
                content_type="structured",
                document_status=document_status,  # type: ignore[arg-type]
                service_domain=service_domain,
            )
        )
    return chunks


def create_structured_table_chunks(
    *,
    pages: List[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType = SourceType.pdf,
    source_url: Optional[str] = None,
    file_type: Optional[str] = None,
    document_status: str = "unknown",
    effective_date: Optional[str] = None,
    version: Optional[str] = None,
    chunk_index_offset: int = 0,
    service_domain: Optional[str] = None,
) -> List[DocumentChunk]:
    """Index table rows as structured enhancements alongside universal rows."""
    uploaded_at = datetime.utcnow()
    chunks: List[DocumentChunk] = []
    for page in pages:
        for row in page.table_rows:
            content = row.human_text.strip()
            if not content:
                continue
            identity = f"{page.page_number}:{row.table_index}:{row.row_index}:{content}"
            content_hash = hashlib.sha256(
                f"{company_id}:{document_id}:table-structured:{identity}".encode("utf-8")
            ).hexdigest()
            chunk_id = hashlib.sha256(
                f"{company_id}:{document_id}:table-row:{identity}".encode("utf-8")
            ).hexdigest()[:32]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    company_id=company_id,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page.page_number,
                    slide_number=page.slide_number,
                    row_number=page.row_number or row.row_index,
                    section_title=row.heading,
                    chunk_index=chunk_index_offset + len(chunks),
                    content=content,
                    content_hash=content_hash,
                    source_type=source_type,
                    source_url=source_url,
                    file_type=file_type or source_type.value,
                    uploaded_at=uploaded_at,
                    created_at=uploaded_at,
                    record_id=chunk_id,
                    record_type="table_row",
                    title=row.row_label or row.heading,
                    dates=row.effective_date or effective_date,
                    content_type="structured_table_row",
                    label=row.column_labels[0] if row.column_labels else None,
                    value=row.row_label,
                    table_data=row.model_dump(),
                    metadata_json=dict(page.metadata or {}),
                    effective_date=row.effective_date or effective_date,
                    version=version,
                    document_status=document_status,  # type: ignore[arg-type]
                    service_domain=service_domain,
                )
            )
    return chunks


def merge_chunks(
    universal: List[DocumentChunk], structured: List[DocumentChunk]
) -> List[DocumentChunk]:
    """Keep all universal chunks; append structured enhancements when not exact dupes."""
    merged = list(universal)
    seen = {chunk.content_hash for chunk in universal}
    seen_ids = {chunk.chunk_id for chunk in universal}
    for chunk in structured:
        if chunk.content_hash in seen or chunk.chunk_id in seen_ids:
            continue
        merged.append(chunk)
        seen.add(chunk.content_hash)
        seen_ids.add(chunk.chunk_id)
    for index, chunk in enumerate(merged):
        chunk.chunk_index = index
    return merged
