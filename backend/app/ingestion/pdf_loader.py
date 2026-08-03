from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from app.models.api import ExtractedPage, ExtractedTableRow

logger = logging.getLogger(__name__)

# Only treat the PDF as empty/OCR-required when essentially no text is present.
# Short company one-pagers can be well under 100 characters and are still valid.
MIN_SELECTABLE_CHARS = 1


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be processed usefully."""


EFFECTIVE_DATE_RE = re.compile(
    r"(?i)\b(?:effective|valid|fees?\s+(?:as\s+of|from))\s*(?:date)?\s*:?\s*"
    r"((?:[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})|"
    r"(?:[A-Z][a-z]+\s+\d{4})|(?:\d{1,4}[/-]){2}\d{1,4}|(?:19|20)\d{2})"
)
CURRENCY_OR_NUMBER_RE = re.compile(
    r"(?:[$¥€£]\s*\d|\d[\d,.]*\s*(?:USD|JPY|EUR|GBP|dollars?|yen|euros?|pounds?))",
    re.I,
)


def _extract_page_text(
    page: "fitz.Page", table_bboxes: Sequence[Tuple[float, float, float, float]] = ()
) -> str:
    """Layout-aware extraction via PyMuPDF blocks, preserving line breaks."""
    blocks = page.get_text("blocks") or []
    usable: List[tuple[float, float, List[str]]] = []
    for block in blocks:
        if not isinstance(block, (list, tuple)) or len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not isinstance(text, str):
            continue
        if any(_rect_overlap((x0, y0, x1, y1), bbox) >= 0.55 for bbox in table_bboxes):
            # Table text is emitted as complete logical rows below. Keeping the
            # individual cell blocks here would create isolated, misleading chunks.
            continue
        cleaned = text.replace("\u0000", "").replace("\u200b", "").strip()
        if not cleaned:
            continue
        parts = [part.strip() for part in cleaned.split("\n") if part.strip()]
        if parts:
            usable.append((float(y0), float(x0), parts))

    if usable:
        usable.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
        lines: List[str] = []
        for _, _, parts in usable:
            lines.extend(parts)
        return "\n".join(lines).strip()

    if table_bboxes:
        return ""
    return (page.get_text("text") or "").strip()


def _rect_overlap(
    first: Sequence[float], second: Sequence[float]
) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (float(first[2]) - float(first[0])) * (float(first[3]) - float(first[1])))
    return intersection / area


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u0000", " ")).strip()


def _nearby_table_context(
    page: "fitz.Page", bbox: Sequence[float]
) -> Tuple[Optional[str], Optional[str]]:
    """Return a nearby heading and effective date without embedding page noise."""
    above: List[Tuple[float, str]] = []
    page_text_parts: List[str] = []
    for block in page.get_text("blocks") or []:
        if not isinstance(block, (list, tuple)) or len(block) < 5:
            continue
        text = _clean_cell(block[4])
        if not text:
            continue
        page_text_parts.append(text)
        if float(block[3]) <= float(bbox[1]) + 2:
            for line in str(block[4]).splitlines():
                cleaned = _clean_cell(line)
                if cleaned:
                    above.append((float(block[3]), cleaned))

    heading = None
    for _, candidate in sorted(above, key=lambda item: item[0], reverse=True):
        if len(candidate) <= 120 and not CURRENCY_OR_NUMBER_RE.search(candidate):
            if not EFFECTIVE_DATE_RE.search(candidate):
                heading = candidate
                break

    context = "\n".join(text for _, text in sorted(above, key=lambda item: item[0])[-8:])
    if not context:
        context = "\n".join(page_text_parts)
    date_match = EFFECTIVE_DATE_RE.search(context)
    effective_date = date_match.group(1).strip() if date_match else None
    return heading, effective_date


def _looks_like_header(cells: Sequence[str]) -> bool:
    populated = [cell for cell in cells if cell]
    if len(populated) < 2:
        return False
    numeric = sum(bool(CURRENCY_OR_NUMBER_RE.search(cell)) for cell in populated)
    return numeric == 0 and all(len(cell) <= 80 for cell in populated)


def _extract_table_rows(
    page: "fitz.Page",
) -> Tuple[List[ExtractedTableRow], List[Tuple[float, float, float, float]]]:
    rows: List[ExtractedTableRow] = []
    bboxes: List[Tuple[float, float, float, float]] = []
    try:
        finder = page.find_tables()
        tables = list(getattr(finder, "tables", []) or [])
    except Exception:  # noqa: BLE001
        logger.exception("Table detection failed on PDF page %s", page.number + 1)
        return rows, bboxes

    for table_index, table in enumerate(tables):
        bbox = tuple(float(value) for value in table.bbox)
        bboxes.append(bbox)
        matrix = [
            [_clean_cell(cell) for cell in raw_row]
            for raw_row in (table.extract() or [])
            if raw_row
        ]
        matrix = [row for row in matrix if any(row)]
        if not matrix:
            continue

        header_names = [
            _clean_cell(name)
            for name in (getattr(getattr(table, "header", None), "names", None) or [])
        ]
        if not any(header_names) or len(header_names) != len(matrix[0]):
            header_names = list(matrix[0]) if _looks_like_header(matrix[0]) else []
        first_is_header = bool(
            header_names
            and len(matrix[0]) == len(header_names)
            and all(
                _clean_cell(cell).lower() == _clean_cell(label).lower()
                for cell, label in zip(matrix[0], header_names)
            )
        )
        data_rows = matrix[1:] if first_is_header else matrix
        width = max(len(header_names), max(len(row) for row in data_rows))
        labels = [
            header_names[index] if index < len(header_names) and header_names[index]
            else f"Column {index + 1}"
            for index in range(width)
        ]
        heading, effective_date = _nearby_table_context(page, bbox)

        logical_rows: List[List[str]] = []
        for raw_row in data_rows:
            padded = list(raw_row) + [""] * (width - len(raw_row))
            # A blank first cell usually means a wrapped continuation of the
            # prior logical row. Merge corresponding cells instead of indexing it alone.
            if logical_rows and not padded[0] and any(padded[1:]):
                previous = logical_rows[-1]
                for index, value in enumerate(padded):
                    if value:
                        previous[index] = (
                            f"{previous[index]} {value}".strip()
                            if previous[index]
                            else value
                        )
                continue
            logical_rows.append(padded)

        for row_index, cells in enumerate(logical_rows):
            pairs = [
                (labels[index], value)
                for index, value in enumerate(cells)
                if value
            ]
            if not pairs:
                continue
            lines: List[str] = []
            if effective_date:
                lines.append(f"Effective date: {effective_date}")
            lines.extend(f"{label}: {value}" for label, value in pairs)
            rows.append(
                ExtractedTableRow(
                    table_index=table_index,
                    row_index=row_index,
                    heading=heading,
                    effective_date=effective_date,
                    column_labels=labels,
                    cells=cells,
                    row_label=next((value for value in cells if value), None),
                    human_text="\n".join(lines),
                )
            )
    return rows, bboxes


def extract_pdf_pages(path: Path) -> Tuple[List[ExtractedPage], dict]:
    """Extract selectable text page-by-page with PyMuPDF.

    OCR is not implemented yet. If no selectable text is found, callers should
    surface a clear OCR-required message. Extension point: plug an OCR backend
    into `ocr_page(page: fitz.Page) -> str` later.
    """
    try:
        document = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise PdfExtractionError(
                "This PDF appears password-protected and cannot be processed."
            ) from exc
        raise PdfExtractionError(f"Unable to open PDF: {exc}") from exc

    if getattr(document, "needs_pass", False):
        document.close()
        raise PdfExtractionError(
            "This PDF appears password-protected and cannot be processed."
        )

    pages: List[ExtractedPage] = []
    empty_pages = 0

    try:
        for index, page in enumerate(document, start=1):
            table_rows, table_bboxes = _extract_table_rows(page)
            text = _extract_page_text(page, table_bboxes)
            if not text and not table_rows:
                empty_pages += 1
                # Extension point for OCR:
                # text = ocr_page(page)
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text=text,
                    table_rows=table_rows,
                )
            )
    finally:
        document.close()

    total_chars = sum(
        len(page.text) + sum(len(row.human_text) for row in page.table_rows)
        for page in pages
    )
    metadata = {
        "page_count": len(pages),
        "empty_pages": empty_pages,
        "character_count": total_chars,
        "table_row_count": sum(len(page.table_rows) for page in pages),
        "ocr_required": total_chars < MIN_SELECTABLE_CHARS,
    }

    if metadata["ocr_required"]:
        logger.warning("PDF appears empty or image-only: %s", path.name)
        raise PdfExtractionError(
            "No selectable text was found in this PDF. It may be scanned and require OCR. "
            "Try a text-based PDF export, or wait for OCR support."
        )

    return pages, metadata


def ocr_page(page: "fitz.Page") -> str:
    """Placeholder OCR extension point.

    Implement with Tesseract/PaddleOCR/cloud OCR later. Must return extracted text
    for a single page or an empty string.
    """
    raise NotImplementedError("OCR is not implemented in this prototype.")
