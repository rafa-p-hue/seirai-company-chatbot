from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime
from typing import List, Optional, Sequence

from app.ingestion.cleaner import clean_text, estimate_tokens, split_sentences
from app.models.api import DocumentChunk, ExtractedPage, SourceType

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 /&-]{2,80}|[A-Z][A-Za-z0-9 /&-]{2,80})$"
)


def chunk_pages(
    *,
    pages: Sequence[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType = SourceType.pdf,
    source_url: Optional[str] = None,
    target_min: int = 400,
    target_max: int = 700,
    overlap: int = 80,
    min_useful: int = 12,
) -> List[DocumentChunk]:
    blocks: List[tuple[int, Optional[str], str]] = []
    current_heading: Optional[str] = None

    for page in pages:
        text = clean_text(page.text)
        if not text:
            continue
        for paragraph in re.split(r"\n{2,}", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            first_line = paragraph.split("\n", 1)[0].strip()
            if is_heading(first_line):
                current_heading = first_line
                body = paragraph[len(first_line) :].strip()
                if body:
                    blocks.append((page.page_number, current_heading, body))
                else:
                    blocks.append((page.page_number, current_heading, first_line))
            else:
                blocks.append((page.page_number, current_heading, paragraph))

    packed = pack_blocks(blocks, target_min=target_min, target_max=target_max, overlap=overlap)
    chunks: List[DocumentChunk] = []
    uploaded_at = datetime.utcnow()

    for index, (page_number, heading, content) in enumerate(packed):
        content = content.strip()
        if not is_useful_chunk(content, min_useful=min_useful):
            continue
        content_hash = hashlib.sha256(
            f"{company_id}:{document_id}:{index}:{content}".encode("utf-8")
        ).hexdigest()
        chunk_id = hashlib.sha256(
            f"{company_id}:{document_id}:{content_hash}".encode("utf-8")
        ).hexdigest()[:32]

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                company_id=company_id,
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                section_title=heading,
                chunk_index=index,
                content=content,
                content_hash=content_hash,
                source_type=source_type,
                source_url=source_url,
                uploaded_at=uploaded_at,
            )
        )

    # Prefer keeping a short but meaningful document over failing ingestion entirely.
    if not chunks and packed:
        page_number, heading, content = max(packed, key=lambda item: len(item[2]))
        content = content.strip()
        if len(content) >= 20:
            content_hash = hashlib.sha256(
                f"{company_id}:{document_id}:0:{content}".encode("utf-8")
            ).hexdigest()
            chunk_id = hashlib.sha256(
                f"{company_id}:{document_id}:{content_hash}".encode("utf-8")
            ).hexdigest()[:32]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    company_id=company_id,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    section_title=heading,
                    chunk_index=0,
                    content=content,
                    content_hash=content_hash,
                    source_type=source_type,
                    source_url=source_url,
                    uploaded_at=uploaded_at,
                )
            )

    logger.info("Created %s chunks for document %s", len(chunks), document_id)
    return chunks


def is_heading(line: str) -> bool:
    if not line or len(line) > 90 or line.endswith("."):
        return False
    if HEADING_RE.match(line):
        words = line.split()
        titled = sum(1 for word in words if word[:1].isupper())
        return titled / max(len(words), 1) >= 0.7
    return False


def looks_factual(text: str) -> bool:
    words = [word for word in text.split() if word]
    if len(words) >= 8:
        return True
    return bool(re.search(r"[:：]\s*\S+", text) or re.search(r"\b\d{4}\b", text))


def is_useful_chunk(text: str, *, min_useful: int) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    return estimate_tokens(text) >= min_useful or looks_factual(text)

def pack_blocks(
    blocks: Sequence[tuple[int, Optional[str], str]],
    *,
    target_min: int,
    target_max: int,
    overlap: int,
) -> List[tuple[int, Optional[str], str]]:
    if not blocks:
        return []

    packed: List[tuple[int, Optional[str], str]] = []
    current_parts: List[str] = []
    current_page = blocks[0][0]
    current_heading = blocks[0][1]
    current_tokens = 0

    def flush() -> None:
        nonlocal current_parts, current_tokens
        if not current_parts:
            return
        packed.append((current_page, current_heading, "\n\n".join(current_parts).strip()))

    for page_number, heading, text in blocks:
        token_count = estimate_tokens(text)
        if token_count > target_max:
            flush()
            for sentence_chunk in split_oversized(text, target_max=target_max, overlap=overlap):
                packed.append((page_number, heading, sentence_chunk))
            current_parts = []
            current_tokens = 0
            current_page = page_number
            current_heading = heading
            continue

        if current_parts and (
            current_tokens + token_count > target_max
            or (heading and heading != current_heading and current_tokens >= target_min)
        ):
            flush()
            # overlap: keep last part if useful
            if current_parts and overlap > 0:
                last = current_parts[-1]
                if estimate_tokens(last) <= overlap:
                    current_parts = [last]
                    current_tokens = estimate_tokens(last)
                else:
                    current_parts = []
                    current_tokens = 0
            else:
                current_parts = []
                current_tokens = 0
            current_page = page_number
            current_heading = heading

        if not current_parts:
            current_page = page_number
            current_heading = heading

        current_parts.append(text)
        current_tokens += token_count

    flush()
    return packed


def split_oversized(text: str, *, target_max: int, overlap: int) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        words = text.split()
        step = max(1, target_max - overlap)
        return [
            " ".join(words[i : i + target_max])
            for i in range(0, len(words), step)
            if words[i : i + target_max]
        ]

    chunks: List[str] = []
    current: List[str] = []
    tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and tokens + sentence_tokens > target_max:
            chunks.append(" ".join(current))
            if overlap > 0 and current:
                current = [current[-1]]
                tokens = estimate_tokens(current[0])
            else:
                current = []
                tokens = 0
        current.append(sentence)
        tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def new_document_id() -> str:
    return str(uuid.uuid4())
