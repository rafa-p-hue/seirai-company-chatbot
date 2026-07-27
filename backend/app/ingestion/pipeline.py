from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from app.ingestion.chunker import new_document_id
from app.ingestion.cleaner import clean_pages
from app.ingestion.entity_detect import detect_document_headings, detect_primary_entities
from app.ingestion.linkedin_clean import clean_linkedin_pages
from app.ingestion.pdf_loader import PdfExtractionError, extract_pdf_pages
from app.ingestion.structured_records import parse_structured_records
from app.ingestion.universal_chunker import create_universal_chunks
from app.ingestion.web_loader import crawl_website
from app.models.api import DocumentChunk, DocumentSummary, ExtractedPage, SourceType

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w.\- ]+", "_", name).strip()
    return name[:180] or "document.pdf"


async def ingest_pdf_bytes(
    *,
    data: bytes,
    filename: str,
    company_id: str,
    document_id: Optional[str] = None,
) -> Tuple[DocumentSummary, List[ExtractedPage], List[DocumentChunk]]:
    """Always create universal chunks; structured records are optional metadata."""
    safe_name = sanitize_filename(filename)
    document_id = document_id or new_document_id()

    temp_dir = Path("data/uploads") / company_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{document_id}_{safe_name}"
    path.write_bytes(data)

    try:
        pages, metadata = extract_pdf_pages(path)
    except PdfExtractionError:
        raise

    cleaned = clean_linkedin_pages(clean_pages(pages))

    universal = create_universal_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=safe_name,
        source_type=SourceType.pdf,
    )
    structured = optional_structured_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=safe_name,
        source_type=SourceType.pdf,
        chunk_index_offset=len(universal),
    )
    chunks = merge_chunks(universal, structured)
    if not chunks:
        raise ValueError(
            "No useful text chunks could be created from this PDF. "
            "If it is a scanned/image PDF, OCR is required."
        )

    entities = detect_primary_entities(
        pages=cleaned, document_name=safe_name, chunks=chunks
    )
    logger.info(
        "Ingested %s: %s universal, %s structured, entities=%s",
        safe_name,
        len(universal),
        len(structured),
        entities,
    )

    summary = DocumentSummary(
        document_id=document_id,
        company_id=company_id,
        document_name=safe_name,
        source_type=SourceType.pdf,
        page_count=metadata["page_count"],
        chunk_count=len(chunks),
        embedding_count=0,
        status="chunked",
        uploaded_at=chunks[0].uploaded_at,
        errors=[],
        primary_entities=entities,
        universal_chunk_count=len(universal),
        structured_chunk_count=len(structured),
        document_headings=detect_document_headings(cleaned),
    )
    return summary, cleaned, chunks


async def ingest_website(
    *,
    company_id: str,
    url: str,
    max_pages: int,
    allowlist: Optional[List[str]] = None,
    document_id: Optional[str] = None,
) -> Tuple[DocumentSummary, List[ExtractedPage], List[DocumentChunk]]:
    document_id = document_id or new_document_id()
    pages = await crawl_website(url, max_pages=max_pages, allowlist=allowlist)
    cleaned = clean_pages(pages)
    document_name = url.rstrip("/").split("/")[-1] or "website"

    universal = create_universal_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=document_name,
        source_type=SourceType.website,
        source_url=url,
    )
    structured = optional_structured_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=document_name,
        source_type=SourceType.website,
        source_url=url,
        chunk_index_offset=len(universal),
    )
    chunks = merge_chunks(universal, structured)
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
        page_count=len(cleaned),
        chunk_count=len(chunks),
        embedding_count=0,
        status="chunked",
        uploaded_at=chunks[0].uploaded_at,
        errors=[],
        primary_entities=entities,
        universal_chunk_count=len(universal),
        structured_chunk_count=len(structured),
        document_headings=detect_document_headings(cleaned),
    )
    return summary, cleaned, chunks


def optional_structured_chunks(
    *,
    pages: List[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType = SourceType.pdf,
    source_url: Optional[str] = None,
    chunk_index_offset: int = 0,
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
                uploaded_at=uploaded_at,
                record_id=record.record_id,
                record_type=record.record_type,
                person_name=record.person_name,
                title=record.title,
                organization=record.organization,
                dates=record.dates,
                location=record.location,
                content_type="structured",
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
