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
# Reject sentence-like title-case lines that are really body content.
HEADING_BODY_NOISE_RE = re.compile(
    r"(?i)\b(within|must|should|register|bring|submit|provide|required|days?)\b"
)
# Labels that read like section titles with a sentence value (not key-value fields).
SECTION_TITLE_LABEL_RE = re.compile(
    r"(?i).*\b("
    r"policy|policies|guidelines?|rules?|overview|specification|"
    r"handbook|procedure|procedures"
    r")\s*$"
)
BULLET_RE = re.compile(r"^[-•●▪◦·]\s+")
NUMBERED_RE = re.compile(r"^\d+[.)]\s+")
LIST_MARKER_RE = re.compile(r"^(?:[-•●▪◦·]|\d+[.)])\s+")
DEADLINE_UNIT_RE = re.compile(
    r"(?i)\b("
    r"within\s+\d+\s+(?:business\s+)?(?:days?|weeks?|months?)|"
    r"must\s+(?:register|apply|submit|enroll|file)|"
    r"no later than|before\s+\d+|after\s+moving"
    r")\b"
)
REQUIREMENT_INTRO_RE = re.compile(
    r"(?i)\b("
    r"required (?:documents?|items?|materials?)|"
    r"bring(?: the following)?|documents? (?:needed|required)|"
    r"you (?:will )?need|submit the following|provide the following"
    r")\b"
)
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
    file_type: Optional[str] = None,
    document_status: str = "unknown",
    effective_date: Optional[str] = None,
    version: Optional[str] = None,
    metadata_json: Optional[dict] = None,
    service_domain: Optional[str] = None,
    target_min: int = 40,
    target_max: int = 300,
) -> List[DocumentChunk]:
    """Create universal chunks preserving page, section, subsection, labels.

    Token limits are guidance only — complete facts are not split solely to
    stay inside the range. Prefer ~40–300 tokens when practical.
    """
    units = _stamp_page_fields(_extract_units(pages), pages)
    packed = _pack_units(units, target_min=target_min, target_max=target_max)
    packed = _merge_heading_with_procedure(packed, target_max=target_max)
    packed = _merge_procedural_neighbors(packed, target_max=target_max)
    uploaded_at = datetime.utcnow()
    chunks: List[DocumentChunk] = []
    shared_meta = dict(metadata_json or {})
    if service_domain:
        shared_meta.setdefault("service_domain", service_domain)

    for index, unit in enumerate(packed):
        content = unit["content"].strip()
        # Key-value fields and short section headings must still be indexed.
        kind = unit.get("kind")
        if kind in {"label", "heading"}:
            min_len = 3
        else:
            min_len = 12
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
            "numbered_list": "numbered_list",
            "numbered_list_group": "numbered_list",
            "table_row": "table",
            "heading": "heading",
            "prompt": "prompt",
            "paragraph": "paragraph",
        }.get(str(unit.get("kind") or ""), "paragraph")
        unit_meta = dict(shared_meta)
        unit_meta.update(unit.get("metadata") or {})
        row_effective = None
        if unit.get("table_data") and isinstance(unit.get("table_data"), dict):
            row_effective = unit["table_data"].get("effective_date")
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                company_id=company_id,
                document_id=document_id,
                document_name=document_name,
                page_number=unit.get("page_number"),
                slide_number=unit.get("slide_number"),
                row_number=unit.get("row_number"),
                section_title=unit.get("heading"),
                subsection_title=unit.get("subsection"),
                chunk_index=index,
                content=content,
                content_hash=content_hash,
                source_type=source_type,
                source_url=source_url,
                file_type=file_type or source_type.value,
                uploaded_at=uploaded_at,
                created_at=uploaded_at,
                record_type="universal",
                title=unit.get("label") or document_name,
                person_name=unit.get("person_name"),
                content_type=content_type,
                label=unit.get("label"),
                value=unit.get("value"),
                table_data=unit.get("table_data"),
                metadata_json=unit_meta,
                effective_date=row_effective or effective_date,
                version=version,
                document_status=document_status,  # type: ignore[arg-type]
                service_domain=service_domain,
            )
        )
    return chunks


def _extract_units(pages: Sequence[ExtractedPage]) -> List[dict]:
    units: List[dict] = []
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    for page in pages:
        text = clean_text(page.text)
        raw_lines: List[str] = []
        for line in text.split("\n") if text else []:
            line = line.strip()
            if not line:
                continue
            raw_lines.extend(_expand_inline_labels(line))

        buffer: List[str] = []
        if page.section_heading and current_section is None:
            current_section = page.section_heading

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
                # "Remote Work Policy: Employees may…" → section heading + paragraph.
                if _looks_like_section_title_label(label, value):
                    is_all_caps = label == label.upper() and re.search(r"[A-Z]", label)
                    if (
                        is_all_caps
                        or current_section is None
                        or _looks_like_top_section(label)
                    ):
                        current_section = label[:120]
                        current_subsection = None
                    else:
                        current_subsection = label[:120]
                    units.append(
                        {
                            "content": label,
                            "page_number": page.page_number,
                            "heading": current_section,
                            "subsection": current_subsection,
                            "kind": "heading",
                        }
                    )
                    if value:
                        units.append(
                            {
                                "content": value,
                                "page_number": page.page_number,
                                "heading": current_section,
                                "subsection": current_subsection,
                                "kind": "paragraph",
                            }
                        )
                    continue
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
                and not line.endswith((".", "!", "?"))
                and not HEADING_BODY_NOISE_RE.search(line)
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
                or NUMBERED_RE.match(line)
                or (LIST_ENTRY_RE.match(line) and len(line) > 30)
                or TABLE_ROW_RE.match(line)
            ):
                flush_buffer()
                is_numbered = bool(NUMBERED_RE.match(line))
                is_table = bool(TABLE_ROW_RE.match(line)) and not (
                    BULLET_RE.match(line) or NUMBERED_RE.match(line)
                )
                body = LIST_MARKER_RE.sub("", line).strip()
                kind = (
                    "table_row"
                    if is_table
                    else ("numbered_list" if is_numbered else "list")
                )
                units.append(
                    {
                        "content": body,
                        "page_number": page.page_number,
                        "heading": current_section,
                        "subsection": current_subsection,
                        "kind": kind,
                        "open_list": not body.endswith((".", "!", "?")),
                        "prohibition": bool(PROHIBITION_RE.search(body)),
                    }
                )
                continue

            # Continue an open list/bullet entry onto following soft-wrapped lines.
            if (
                units
                and units[-1].get("kind") in {"list", "numbered_list", "table_row"}
                and units[-1].get("open_list")
                and not LABEL_RE.match(line)
                and not HEADING_RE.match(line)
                and not LIST_MARKER_RE.match(line)
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
            if unit.get("kind") in {"list", "numbered_list", "table_row"}:
                unit["open_list"] = False

        flush_buffer()

        # Native PDF table extraction supplies complete logical rows. Keep each
        # row atomic: its label, column labels, values, and date travel together.
        for table_row in page.table_rows:
            table_data = table_row.model_dump()
            units.append(
                {
                    "content": table_row.human_text,
                    "page_number": page.page_number,
                    "heading": table_row.heading or current_section,
                    "subsection": current_subsection,
                    "kind": "table_row",
                    "label": table_row.column_labels[0]
                    if table_row.column_labels
                    else None,
                    "value": table_row.row_label,
                    "table_data": table_data,
                    "open_list": False,
                    "row_number": page.row_number or table_row.row_index,
                }
            )

    # Section context stays in metadata only — never baked into source text.
    return units


def _stamp_page_fields(
    units: Sequence[dict], pages: Sequence[ExtractedPage]
) -> List[dict]:
    by_page = {page.page_number: page for page in pages}
    stamped: List[dict] = []
    for unit in units:
        page = by_page.get(unit.get("page_number"))
        item = dict(unit)
        if page is not None:
            if item.get("slide_number") is None:
                item["slide_number"] = page.slide_number
            if item.get("row_number") is None:
                item["row_number"] = page.row_number
            meta = dict(page.metadata or {})
            meta.update(item.get("metadata") or {})
            item["metadata"] = meta
            if not item.get("heading") and page.section_heading:
                item["heading"] = page.section_heading
        stamped.append(item)
    return stamped


def _looks_like_section_title_label(label: str, value: str) -> bool:
    """True when Label: Value is really a section heading with body text."""
    if not SECTION_TITLE_LABEL_RE.match(label.strip()):
        return False
    value = (value or "").strip()
    if not value:
        return False
    # Sentence-like body (not a short field value like "Yes" or "OrbitDock X2").
    if len(value) < 24:
        return False
    if " " not in value:
        return False
    return bool(re.match(r"^[A-Z]", value)) or value.endswith((".", "!", "?"))


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
            r"education|experience|leadership|activities|involvement|"
            r"leave|specification|handbook|remote|guidelines?|"
            r"registration|register|moving|resident|requirements?|"
            r"documents|certificates?|fees|application|enrollment"
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
                "slide_number": prose_buf[0].get("slide_number"),
                "row_number": prose_buf[0].get("row_number"),
                "heading": prose_buf[0].get("heading"),
                "subsection": prose_buf[0].get("subsection"),
                "kind": "paragraph",
                "metadata": prose_buf[0].get("metadata") or {},
            }
        )
        prose_buf = []

    def flush_lists() -> None:
        """Keep related bullets together so prohibition lists stay intact."""
        nonlocal list_buf, prose_buf
        if not list_buf:
            return
        # Keep procedure deadline + requirement intro with the bullet list.
        intro_parts: List[str] = []
        while (
            prose_buf
            and prose_buf[-1].get("heading") == list_buf[0].get("heading")
            and prose_buf[-1].get("subsection") == list_buf[0].get("subsection")
        ):
            candidate = str(prose_buf[-1].get("content") or "").strip()
            if candidate.rstrip().endswith(":") or REQUIREMENT_INTRO_RE.search(
                candidate
            ) or DEADLINE_UNIT_RE.search(candidate):
                intro_unit = prose_buf.pop()
                intro_parts.insert(0, str(intro_unit.get("content") or "").strip())
                continue
            break
        if prose_buf:
            content = "\n".join(item["content"] for item in prose_buf).strip()
            packed.append(
                {
                    "content": content,
                    "page_number": prose_buf[0]["page_number"],
                    "slide_number": prose_buf[0].get("slide_number"),
                    "row_number": prose_buf[0].get("row_number"),
                    "heading": prose_buf[0].get("heading"),
                    "subsection": prose_buf[0].get("subsection"),
                    "kind": "paragraph",
                    "metadata": prose_buf[0].get("metadata") or {},
                }
            )
            prose_buf = []
        intro = "\n".join(part for part in intro_parts if part)

        heading = list_buf[0].get("heading")
        base_kind = str(list_buf[0].get("kind") or "list")
        cleaned_lines = []
        for i, item in enumerate(list_buf, start=1):
            text = item["content"]
            # Strip any legacy bracket heading prefixes if present.
            text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
            if base_kind == "numbered_list":
                cleaned_lines.append(f"{i}. {text}")
            elif base_kind == "table_row":
                cleaned_lines.append(text)
            else:
                cleaned_lines.append(f"• {text}")
        body = "\n".join(cleaned_lines)
        if intro:
            body = f"{intro}\n{body}"
        group_kind = {
            "numbered_list": "numbered_list_group",
            "table_row": "table_row",
            "list": "list_group",
        }.get(base_kind, "list_group")
        packed.append(
            {
                "content": body,
                "page_number": list_buf[0]["page_number"],
                "slide_number": list_buf[0].get("slide_number"),
                "row_number": list_buf[0].get("row_number"),
                "heading": heading,
                "subsection": list_buf[0].get("subsection"),
                "kind": group_kind,
                "metadata": list_buf[0].get("metadata") or {},
            }
        )
        list_buf = []

    for unit in units:
        kind = unit.get("kind")
        if kind == "table_row":
            flush_lists()
            flush_prose()
            packed.append(unit)
            continue
        if kind in {"list", "numbered_list"}:
            # Keep deadline / requirement intros in prose_buf so flush_lists can merge them.
            last = str(prose_buf[-1].get("content") or "") if prose_buf else ""
            keep_last = bool(
                last.rstrip().endswith(":")
                or DEADLINE_UNIT_RE.search(last)
                or REQUIREMENT_INTRO_RE.search(last)
            )
            if prose_buf and not keep_last:
                flush_prose()
            elif prose_buf and len(prose_buf) > 1 and keep_last:
                intro_unit = prose_buf[-1]
                prose_buf = prose_buf[:-1]
                flush_prose()
                prose_buf = [intro_unit]
            # Group consecutive list items under the same section and list style.
            if list_buf and (
                list_buf[0].get("heading") != unit.get("heading")
                or list_buf[0].get("subsection") != unit.get("subsection")
                or list_buf[0].get("page_number") != unit.get("page_number")
                or list_buf[0].get("kind") != unit.get("kind")
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
            # Keep short headings as metadata anchors (never drop section titles).
            packed.append(unit)
            continue

        prose_buf.append(unit)
        tokens = estimate_tokens(" ".join(item["content"] for item in prose_buf))
        # Prefer not splitting mid-paragraph; only flush at natural boundaries
        # once over the soft max, or at sentence ends past the soft min.
        if tokens >= target_max and unit["content"].endswith((".", "?", "!")):
            flush_prose()
        elif tokens >= target_min and unit["content"].endswith((".", "?", "!")):
            flush_prose()
        elif tokens >= int(target_max * 1.25):
            # Hard safety: avoid runaway chunks without requiring a sentence end.
            flush_prose()

    flush_lists()
    flush_prose()
    return packed


def _is_procedural_body(text: str) -> bool:
    value = text or ""
    return bool(
        DEADLINE_UNIT_RE.search(value)
        or REQUIREMENT_INTRO_RE.search(value)
        or re.search(
            r"(?i)\b("
            r"move-?in(?:\s+notification)?|moving\s+in|register|"
            r"within\s+\d+\s+(?:business\s+)?days?|service\s+window|"
            r"bring|required documents?|submit"
            r")\b",
            value,
        )
    )


def _merge_heading_with_procedure(
    packed: Sequence[dict], *, target_max: int
) -> List[dict]:
    """Keep short procedural section headings with their body paragraph."""
    if not packed:
        return []
    merged: List[dict] = []
    index = 0
    while index < len(packed):
        current = dict(packed[index])
        kind = str(current.get("kind") or "")
        if kind == "heading" and index + 1 < len(packed):
            nxt = packed[index + 1]
            nxt_kind = str(nxt.get("kind") or "")
            nxt_content = str(nxt.get("content") or "").strip()
            same_section = (
                (current.get("heading") == nxt.get("heading")
                 or current.get("content") == nxt.get("heading"))
                and (
                    current.get("subsection") == nxt.get("subsection")
                    or not nxt.get("subsection")
                )
            )
            short_procedure = (
                nxt_kind in {"paragraph", "list", "list_group", "numbered_list"}
                and _is_procedural_body(nxt_content)
                and estimate_tokens(nxt_content) <= int(target_max * 0.9)
            )
            if same_section and short_procedure:
                heading_text = str(current.get("content") or "").strip()
                combined = f"{heading_text}\n{nxt_content}".strip()
                if estimate_tokens(combined) <= int(target_max * 1.35):
                    merged.append(
                        {
                            "content": combined,
                            "page_number": current.get("page_number")
                            or nxt.get("page_number"),
                            "slide_number": current.get("slide_number")
                            or nxt.get("slide_number"),
                            "row_number": current.get("row_number")
                            or nxt.get("row_number"),
                            "heading": heading_text
                            or current.get("heading")
                            or nxt.get("heading"),
                            "subsection": nxt.get("subsection")
                            or current.get("subsection"),
                            "kind": "paragraph"
                            if nxt_kind == "paragraph"
                            else nxt_kind,
                            "metadata": {
                                **(current.get("metadata") or {}),
                                **(nxt.get("metadata") or {}),
                            },
                        }
                    )
                    index += 2
                    continue
        merged.append(current)
        index += 1
    return merged


def _merge_procedural_neighbors(
    packed: Sequence[dict], *, target_max: int
) -> List[dict]:
    """Keep deadline sentences adjacent to required-item lists under one chunk."""
    if not packed:
        return []
    merged: List[dict] = []
    index = 0
    while index < len(packed):
        current = dict(packed[index])
        kind = str(current.get("kind") or "")
        if kind in {"paragraph", "heading"} and index + 1 < len(packed):
            nxt = packed[index + 1]
            same_section = (
                current.get("heading") == nxt.get("heading")
                and current.get("subsection") == nxt.get("subsection")
            )
            related_procedure_sections = bool(
                re.search(
                    r"(?i)\b(moving\s+in|move-?in|registration|notification)\b",
                    str(current.get("heading") or ""),
                )
                and re.search(
                    r"(?i)\b(moving\s+in|move-?in|registration|notification)\b",
                    str(nxt.get("heading") or current.get("heading") or ""),
                )
                and re.search(
                    r"(?i)\b(bring|required documents?|my number|residence card)\b",
                    str(nxt.get("content") or ""),
                )
            )
            nxt_kind = str(nxt.get("kind") or "")
            nxt_content = str(nxt.get("content") or "")
            current_content = str(current.get("content") or "")
            # Do not split a short fused procedural paragraph further; also
            # glue deadline prose to an adjacent required-items list/intro.
            if (
                (same_section or related_procedure_sections)
                and nxt_kind in {
                    "list",
                    "list_group",
                    "numbered_list",
                    "numbered_list_group",
                    "paragraph",
                }
                and (
                    DEADLINE_UNIT_RE.search(current_content)
                    or REQUIREMENT_INTRO_RE.search(current_content)
                    or _is_procedural_body(current_content)
                )
                and (
                    nxt_kind != "paragraph"
                    or REQUIREMENT_INTRO_RE.search(nxt_content)
                    or _is_procedural_body(nxt_content)
                    or re.search(
                        r"(?i)^\s*bring\b|my number|residence card|passport",
                        nxt_content,
                    )
                )
            ):
                combined = (
                    f"{current_content.strip()}\n{nxt_content.strip()}"
                ).strip()
                # Avoid merging unrelated long bodies.
                if estimate_tokens(combined) <= int(target_max * 1.45):
                    out_kind = (
                        nxt_kind
                        if "list" in nxt_kind
                        else ("paragraph" if kind == "paragraph" else "paragraph")
                    )
                    current = {
                        "content": combined,
                        "page_number": current.get("page_number")
                        or nxt.get("page_number"),
                        "slide_number": current.get("slide_number")
                        or nxt.get("slide_number"),
                        "row_number": current.get("row_number")
                        or nxt.get("row_number"),
                        "heading": current.get("heading") or nxt.get("heading"),
                        "subsection": current.get("subsection")
                        or nxt.get("subsection"),
                        "kind": out_kind,
                        "metadata": {
                            **(current.get("metadata") or {}),
                            **(nxt.get("metadata") or {}),
                        },
                    }
                    merged.append(current)
                    index += 2
                    continue
        merged.append(current)
        index += 1
    return merged


def _expand_inline_labels(line: str) -> List[str]:
    if line.count(":") < 2:
        return [line]
    parts = [part.strip() for part in INLINE_LABEL_SPLIT_RE.split(line) if part.strip()]
    if len(parts) <= 1:
        return [line]
    if all(LABEL_RE.match(part) for part in parts):
        return parts
    return [line]
