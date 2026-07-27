"""Universal semantic chunking for document-agnostic RAG.

Every document produces raw chunks from paragraphs, headings, labeled fields,
and list/table rows. No document-type-specific facts are assumed.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import List, Optional, Sequence

from app.ingestion.cleaner import clean_text, estimate_tokens
from app.models.api import DocumentChunk, ExtractedPage, SourceType

LABEL_RE = re.compile(
    r"^(?P<label>"
    r"[A-Za-z][^:\n]{0,80}?"
    r")\s*:\s*(?P<value>.+)$"
)
INLINE_LABEL_SPLIT_RE = re.compile(
    r"(?=(?:[A-Z][A-Za-z][^:\n]{0,60})\s*:)"
)
HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 /&-]{2,90}|[A-Z][A-Za-z0-9 /&-]{2,90})$"
)
BULLET_RE = re.compile(r"^(?:[-•*]|\d+[.)])\s+")
TABLE_ROW_RE = re.compile(r".+\|.+|^\s*\S+\s{2,}\S+")
LIST_ENTRY_RE = re.compile(r".+\s[-–—]\s+.+")
FORM_PROMPT_RE = re.compile(
    r"(?i)^(provide a list of|identify any|list all|describe any|please provide|"
    r"awards and/or honors|awards/honors)\b"
)
PROHIBITION_RE = re.compile(
    r"(?i)\b(not allowed|prohibited|forbidden|may not|cannot|do not|no\s+\w+)\b"
)


def create_universal_chunks(
    *,
    pages: Sequence[ExtractedPage],
    company_id: str,
    document_id: str,
    document_name: str,
    source_type: SourceType = SourceType.pdf,
    source_url: Optional[str] = None,
    target_min: int = 40,
    target_max: int = 220,
) -> List[DocumentChunk]:
    """Create universal chunks preserving page, section, subsection, labels."""
    units = _extract_units(pages)
    packed = _pack_units(units, target_min=target_min, target_max=target_max)
    uploaded_at = datetime.utcnow()
    chunks: List[DocumentChunk] = []

    for index, unit in enumerate(packed):
        content = unit["content"].strip()
        # Key-value fields may be short ("Major: CS") but must still be indexed.
        min_len = 3 if unit.get("kind") == "label" else 12
        if len(content) < min_len:
            continue
        content_hash = hashlib.sha256(
            f"{company_id}:{document_id}:universal:{index}:{content}".encode("utf-8")
        ).hexdigest()
        chunk_id = hashlib.sha256(
            f"{company_id}:{document_id}:{content_hash}".encode("utf-8")
        ).hexdigest()[:32]
        content_type = {
            "label": "key_value",
            "list": "list",
            "list_group": "list",
            "heading": "heading",
            "prompt": "prompt",
            "paragraph": "paragraph",
        }.get(str(unit.get("kind") or ""), "paragraph")
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                company_id=company_id,
                document_id=document_id,
                document_name=document_name,
                page_number=unit.get("page_number"),
                section_title=unit.get("heading"),
                subsection_title=unit.get("subsection"),
                chunk_index=index,
                content=content,
                content_hash=content_hash,
                source_type=source_type,
                source_url=source_url,
                uploaded_at=uploaded_at,
                record_type="universal",
                title=unit.get("label") or document_name,
                person_name=unit.get("person_name"),
                content_type=content_type,
                label=unit.get("label"),
                value=unit.get("value"),
            )
        )
    return chunks


def _extract_units(pages: Sequence[ExtractedPage]) -> List[dict]:
    units: List[dict] = []
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    for page in pages:
        text = clean_text(page.text)
        if not text:
            continue
        raw_lines: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            raw_lines.extend(_expand_inline_labels(line))

        buffer: List[str] = []

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            content = " ".join(buffer).strip()
            if content:
                units.append(
                    {
                        "content": content,
                        "page_number": page.page_number,
                        "heading": current_section,
                        "subsection": current_subsection,
                        "kind": "paragraph",
                    }
                )
            buffer = []

        for line in raw_lines:
            label_match = LABEL_RE.match(line)
            if label_match and len(label_match.group("label")) <= 80:
                flush_buffer()
                label = label_match.group("label").strip()
                value = label_match.group("value").strip()
                content = f"{label}: {value}"
                person_name = (
                    value
                    if re.search(
                        r"\b((?:full\s*)?name|candidate\s*name|applicant\s*name|"
                        r"nominee'?s?\s*(?:full\s*)?name)\b",
                        label,
                        re.I,
                    )
                    else None
                )
                units.append(
                    {
                        "content": content,
                        "page_number": page.page_number,
                        "heading": current_section,
                        "subsection": current_subsection,
                        "kind": "label",
                        "label": label,
                        "value": value,
                        "person_name": person_name,
                    }
                )
                continue

            if (
                HEADING_RE.match(line)
                and len(line) < 100
                and ":" not in line[:40]
                and not line.endswith(".")
            ):
                flush_buffer()
                # Top-level ALL-CAPS or short Title Case → section.
                # Nested Title Case under an existing section → subsection.
                is_all_caps = line == line.upper() and re.search(r"[A-Z]", line)
                if is_all_caps or current_section is None or _looks_like_top_section(line):
                    current_section = line[:120]
                    current_subsection = None
                else:
                    current_subsection = line[:120]
                units.append(
                    {
                        "content": line,
                        "page_number": page.page_number,
                        "heading": current_section,
                        "subsection": current_subsection,
                        "kind": "heading",
                    }
                )
                continue

            if FORM_PROMPT_RE.match(line):
                flush_buffer()
                units.append(
                    {
                        "content": line,
                        "page_number": page.page_number,
                        "heading": current_section,
                        "subsection": current_subsection,
                        "kind": "prompt",
                    }
                )
                continue

            if (
                BULLET_RE.match(line)
                or (LIST_ENTRY_RE.match(line) and len(line) > 30)
                or TABLE_ROW_RE.match(line)
            ):
                flush_buffer()
                body = BULLET_RE.sub("", line).strip()
                units.append(
                    {
                        "content": body,
                        "page_number": page.page_number,
                        "heading": current_section,
                        "subsection": current_subsection,
                        "kind": "list",
                        "open_list": not body.endswith((".", "!", "?")),
                        "prohibition": bool(PROHIBITION_RE.search(body)),
                    }
                )
                continue

            # Continue an open list/bullet entry onto following soft-wrapped lines.
            if (
                units
                and units[-1].get("kind") == "list"
                and units[-1].get("open_list")
                and not LABEL_RE.match(line)
                and not HEADING_RE.match(line)
                and not BULLET_RE.match(line)
            ):
                units[-1]["content"] = f"{units[-1]['content']} {line}".strip()
                if units[-1]["content"].endswith((".", "!", "?")) or len(
                    units[-1]["content"]
                ) > 280:
                    units[-1]["open_list"] = False
                continue

            if buffer and not buffer[-1].endswith((".", "?", "!", ":")) and len(line) < 90:
                buffer.append(line)
            else:
                if buffer and (
                    estimate_tokens(" ".join(buffer)) >= 80
                    or buffer[-1].endswith((".", "?", "!"))
                ):
                    flush_buffer()
                buffer.append(line)

        for unit in units:
            if unit.get("kind") == "list":
                unit["open_list"] = False

        flush_buffer()

    # Attach section context to list items without duplicating overview noise.
    for unit in units:
        if unit.get("kind") != "list":
            continue
        heading = unit.get("heading")
        if heading and not unit["content"].startswith(f"[{heading}]"):
            unit["content"] = f"[{heading}] {unit['content']}"

    return units


def _looks_like_top_section(line: str) -> bool:
    """Heuristic: short Title-Case phrases that name document sections."""
    words = line.split()
    if len(words) > 6:
        return False
    # Known generic section vocabulary (not org-specific facts).
    return bool(
        re.search(
            r"(?i)\b("
            r"membership|volunteer|sustainability|donation|rental|rentals|"
            r"refund|cancellation|accessibility|future|plans|timeline|"
            r"overview|policy|policies|rules|hours|program|practices|"
            r"development|services|contact|about|research|interests|"
            r"education|experience|leadership|activities|involvement"
            r")\b",
            line,
        )
    )


def _pack_units(
    units: Sequence[dict],
    *,
    target_min: int,
    target_max: int,
) -> List[dict]:
    packed: List[dict] = []
    prose_buf: List[dict] = []
    list_buf: List[dict] = []

    def flush_prose() -> None:
        nonlocal prose_buf
        if not prose_buf:
            return
        content = "\n".join(item["content"] for item in prose_buf).strip()
        packed.append(
            {
                "content": content,
                "page_number": prose_buf[0]["page_number"],
                "heading": prose_buf[0].get("heading"),
                "subsection": prose_buf[0].get("subsection"),
                "kind": "paragraph",
            }
        )
        prose_buf = []

    def flush_lists() -> None:
        """Keep related bullets together so prohibition lists stay intact."""
        nonlocal list_buf, prose_buf
        if not list_buf:
            return
        # Merge a trailing intro line ending with ':' into the list group.
        intro = ""
        if (
            prose_buf
            and prose_buf[-1].get("heading") == list_buf[0].get("heading")
            and prose_buf[-1].get("subsection") == list_buf[0].get("subsection")
            and str(prose_buf[-1].get("content") or "").rstrip().endswith(":")
        ):
            intro_unit = prose_buf.pop()
            if not prose_buf:
                # Clear any flushed-less buffer residue handled below.
                pass
            intro = str(intro_unit.get("content") or "").strip()
            # If other prose remains, flush it first.
            if prose_buf:
                content = "\n".join(item["content"] for item in prose_buf).strip()
                packed.append(
                    {
                        "content": content,
                        "page_number": prose_buf[0]["page_number"],
                        "heading": prose_buf[0].get("heading"),
                        "subsection": prose_buf[0].get("subsection"),
                        "kind": "paragraph",
                    }
                )
                prose_buf = []

        heading = list_buf[0].get("heading")
        cleaned_lines = []
        for item in list_buf:
            text = item["content"]
            if heading and text.startswith(f"[{heading}] "):
                text = text[len(heading) + 3 :]
            cleaned_lines.append(f"• {text}")
        body = "\n".join(cleaned_lines)
        if intro:
            body = f"{intro}\n{body}"
        if heading:
            body = f"[{heading}]\n{body}"
        packed.append(
            {
                "content": body,
                "page_number": list_buf[0]["page_number"],
                "heading": heading,
                "subsection": list_buf[0].get("subsection"),
                "kind": "list_group",
            }
        )
        list_buf = []

    for unit in units:
        kind = unit.get("kind")
        if kind == "list":
            # Keep a colon-ended intro paragraph in prose_buf so flush_lists can merge it.
            if prose_buf and not str(prose_buf[-1].get("content") or "").rstrip().endswith(":"):
                flush_prose()
            elif prose_buf and len(prose_buf) > 1:
                # Flush earlier prose, keep only the colon intro.
                intro_unit = prose_buf[-1]
                prose_buf = prose_buf[:-1]
                flush_prose()
                prose_buf = [intro_unit]
            # Group consecutive list items under the same section.
            if list_buf and (
                list_buf[0].get("heading") != unit.get("heading")
                or list_buf[0].get("subsection") != unit.get("subsection")
                or list_buf[0].get("page_number") != unit.get("page_number")
                or estimate_tokens(
                    "\n".join(item["content"] for item in list_buf) + "\n" + unit["content"]
                )
                > target_max
            ):
                flush_lists()
            list_buf.append(unit)
            continue

        flush_lists()
        if kind in {"label", "heading", "prompt"}:
            flush_prose()
            # Keep short headings as metadata anchors (do not drop).
            if kind == "heading" and estimate_tokens(unit["content"]) < 3:
                continue
            packed.append(unit)
            continue

        prose_buf.append(unit)
        tokens = estimate_tokens(" ".join(item["content"] for item in prose_buf))
        if tokens >= target_max:
            flush_prose()
        elif tokens >= target_min and unit["content"].endswith((".", "?", "!")):
            flush_prose()

    flush_lists()
    flush_prose()
    return packed


def _expand_inline_labels(line: str) -> List[str]:
    if line.count(":") < 2:
        return [line]
    parts = [part.strip() for part in INLINE_LABEL_SPLIT_RE.split(line) if part.strip()]
    if len(parts) <= 1:
        return [line]
    if all(LABEL_RE.match(part) for part in parts):
        return parts
    return [line]
