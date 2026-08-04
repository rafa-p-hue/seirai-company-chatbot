"""DOCX parser preserving headings, paragraphs, lists, and tables."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.formats import EmptyDocumentError, PasswordProtectedError
from app.models.api import ExtractedPage, ExtractedTableRow


def extract_docx_bytes(data: bytes, *, filename: str = "document.docx") -> Tuple[
    List[ExtractedPage], Dict[str, Any]
]:
    if not data:
        raise EmptyDocumentError("DOCX file is empty.")
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise PasswordProtectedError(
                "This DOCX appears password-protected and cannot be processed."
            ) from exc
        raise ValueError(f"Could not parse DOCX file: {exc}") from exc

    lines: List[str] = []
    table_rows: List[ExtractedTableRow] = []
    current_heading: str | None = None
    table_index = 0

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            if not text:
                continue
            style_name = (block.style.name or "").lower() if block.style else ""
            if "list" in style_name or text.startswith(("•", "-", "*")):
                bullet = text if text[:1] in {"•", "-", "*"} else f"• {text}"
                lines.append(bullet)
            elif style_name.startswith("heading"):
                current_heading = text
                lines.append(text)
            else:
                lines.append(text)
        elif isinstance(block, Table):
            rows, human = _table_rows(block, table_index=table_index, heading=current_heading)
            table_rows.extend(rows)
            lines.extend(human)
            table_index += 1

    content = "\n".join(lines).strip()
    if len(content) < 12 and not table_rows:
        raise EmptyDocumentError("No useful text could be extracted from this DOCX file.")

    page = ExtractedPage(
        page_number=1,
        text=content,
        table_rows=table_rows,
        section_heading=current_heading,
        metadata={"filename": filename},
    )
    return [page], {"page_count": 1}


def _iter_block_items(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _table_rows(
    table: Table, *, table_index: int, heading: str | None
) -> Tuple[List[ExtractedTableRow], List[str]]:
    matrix = [[(cell.text or "").strip() for cell in row.cells] for row in table.rows]
    if not matrix:
        return [], []
    headers = matrix[0]
    if not any(headers):
        headers = [f"Column {i+1}" for i in range(len(matrix[0]))]
    out: List[ExtractedTableRow] = []
    human: List[str] = []
    for row_index, cells in enumerate(matrix[1:] or matrix, start=1):
        if not any(cells):
            continue
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[: len(headers)]
        parts = [f"{headers[i]}: {cells[i]}" for i in range(len(headers)) if cells[i]]
        human_text = "; ".join(parts)
        out.append(
            ExtractedTableRow(
                table_index=table_index,
                row_index=row_index,
                heading=heading,
                column_labels=list(headers),
                cells=list(cells),
                row_label=cells[0] if cells else None,
                human_text=human_text,
            )
        )
        human.append(human_text)
    return out, human
