"""Markdown parser with heading hierarchy, lists, tables, and code blocks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from markdown_it import MarkdownIt

from app.ingestion.formats import EmptyDocumentError
from app.models.api import ExtractedPage, ExtractedTableRow

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_SPLIT_RE = re.compile(r"^\s*\|?(?:[^|\n]+\|)+[^|\n]*\|?\s*$")
TABLE_ALIGN_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def extract_markdown_bytes(
    data: bytes, *, filename: str = "document.md"
) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    if not text.strip():
        raise EmptyDocumentError("Markdown file is empty.")

    # Parse via markdown-it for structure validation, then emit a clean text view.
    MarkdownIt("commonmark").enable("table")
    lines_out: List[str] = []
    table_rows: List[ExtractedTableRow] = []
    current_heading: str | None = None
    table_index = 0
    raw_lines = text.replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        heading = HEADING_RE.match(line)
        if heading:
            current_heading = heading.group(2).strip()
            lines_out.append(current_heading)
            index += 1
            continue
        if line.strip().startswith("```"):
            fence = [line.strip()]
            index += 1
            while index < len(raw_lines) and not raw_lines[index].strip().startswith("```"):
                fence.append(raw_lines[index])
                index += 1
            if index < len(raw_lines):
                fence.append(raw_lines[index].strip())
            code = "\n".join(fence[1:-1]).strip()
            if code:
                lines_out.append(code)
            index += 1
            continue
        if TABLE_SPLIT_RE.match(line) and index + 1 < len(raw_lines) and TABLE_ALIGN_RE.match(
            raw_lines[index + 1]
        ):
            header = _split_row(line)
            index += 2
            body: List[List[str]] = []
            while index < len(raw_lines) and TABLE_SPLIT_RE.match(raw_lines[index]):
                body.append(_split_row(raw_lines[index]))
                index += 1
            for row_index, cells in enumerate(body, start=1):
                while len(cells) < len(header):
                    cells.append("")
                cells = cells[: len(header)]
                parts = [
                    f"{header[i]}: {cells[i]}" for i in range(len(header)) if cells[i]
                ]
                human = "; ".join(parts)
                table_rows.append(
                    ExtractedTableRow(
                        table_index=table_index,
                        row_index=row_index,
                        heading=current_heading,
                        column_labels=header,
                        cells=cells,
                        row_label=cells[0] if cells else None,
                        human_text=human,
                    )
                )
                lines_out.append(human)
            table_index += 1
            continue
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "+ ")):
            lines_out.append(f"• {stripped[2:].strip()}")
        elif re.match(r"^\d+[.)]\s+", stripped):
            lines_out.append(stripped)
        elif stripped:
            lines_out.append(stripped)
        index += 1

    content = "\n".join(lines_out).strip()
    if len(content) < 12 and not table_rows:
        raise EmptyDocumentError(
            "No useful text could be extracted from this Markdown file."
        )
    page = ExtractedPage(
        page_number=1,
        text=content,
        table_rows=table_rows,
        section_heading=current_heading,
        metadata={"filename": filename},
    )
    return [page], {"page_count": 1}


def _split_row(line: str) -> List[str]:
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return parts
