from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from app.ingestion.chunker import chunk_pages, new_document_id
from app.ingestion.cleaner import clean_pages
from app.ingestion.linkedin_clean import clean_linkedin_pages
from app.ingestion.pdf_loader import PdfExtractionError, extract_pdf_pages
from app.ingestion.structured_records import parse_structured_records
from app.ingestion.web_loader import crawl_website
from app.models.api import DocumentChunk, DocumentSummary, ExtractedPage, SourceType
from datetime import datetime
import hashlib

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
    chunks = records_to_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=safe_name,
        source_type=SourceType.pdf,
    )
    if not chunks:
        # Fallback to paragraph chunking if structure parsing yields nothing.
        chunks = chunk_pages(
            pages=cleaned,
            company_id=company_id,
            document_id=document_id,
            document_name=safe_name,
            source_type=SourceType.pdf,
        )
    if not chunks:
        raise ValueError(
            "No useful text chunks could be created from this PDF. "
            "If it is a scanned/image PDF, OCR is required."
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
    chunks = records_to_chunks(
        pages=cleaned,
        company_id=company_id,
        document_id=document_id,
        document_name=document_name,
        source_type=SourceType.website,
        source_url=url,
    )
    if not chunks:
        chunks = chunk_pages(
            pages=cleaned,
            company_id=company_id,
            document_id=document_id,
            document_name=document_name,
            source_type=SourceType.website,
            source_url=url,
        )
    if not chunks:
        raise ValueError("No useful chunks could be created from this website.")

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
    )
    return summary, cleaned, chunks


def records_to_chunks(
    *,
    pages: List[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType = SourceType.pdf,
    source_url: Optional[str] = None,
) -> List[DocumentChunk]:
    records = parse_structured_records(pages, document_id=document_id)
    structured_types = {
        "education",
        "experience",
        "internship",
        "research",
        "leadership",
        "skills",
        "project",
    }
    has_entries = any(record.record_type in structured_types for record in records)
    page_chars = sum(len(page.text) for page in pages)
    record_chars = sum(len(record.embedding_text()) for record in records)
    # Generic company docs often only yield a weak profile; prefer paragraph chunks.
    if not has_entries or (page_chars > 0 and record_chars < page_chars * 0.45):
        logger.info(
            "Structured parse weak for %s (entries=%s); using section/paragraph chunks",
            document_id,
            has_entries,
        )
        return []
    uploaded_at = datetime.utcnow()
    chunks: List[DocumentChunk] = []
    for index, record in enumerate(records):
        content = record.embedding_text()
        if len(content.strip()) < 20:
            continue
        content_hash = hashlib.sha256(
            f"{company_id}:{document_id}:{record.record_id}:{content}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            DocumentChunk(
                chunk_id=record.record_id,
                company_id=company_id,
                document_id=document_id,
                document_name=document_name,
                page_number=record.source_page,
                section_title=record.heading or record.record_type,
                chunk_index=index,
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
            )
        )
    logger.info(
        "Created %s structured records for document %s", len(chunks), document_id
    )
    return chunks
