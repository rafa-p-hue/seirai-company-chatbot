from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

from app.models.api import ExtractedPage

logger = logging.getLogger(__name__)

# Only treat the PDF as empty/OCR-required when essentially no text is present.
# Short company one-pagers can be well under 100 characters and are still valid.
MIN_SELECTABLE_CHARS = 1


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be processed usefully."""


def _extract_page_text(page: "fitz.Page") -> str:
    """Layout-aware extraction via PyMuPDF blocks, preserving line breaks."""
    blocks = page.get_text("blocks") or []
    usable: List[tuple[float, float, List[str]]] = []
    for block in blocks:
        if not isinstance(block, (list, tuple)) or len(block) < 5:
            continue
        x0, y0, _x1, _y1, text = block[0], block[1], block[2], block[3], block[4]
        if not isinstance(text, str):
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

    return (page.get_text("text") or "").strip()


def extract_pdf_pages(path: Path) -> Tuple[List[ExtractedPage], dict]:
    """Extract selectable text page-by-page with PyMuPDF.

    OCR is not implemented yet. If no selectable text is found, callers should
    surface a clear OCR-required message. Extension point: plug an OCR backend
    into `ocr_page(page: fitz.Page) -> str` later.
    """
    try:
        document = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise PdfExtractionError(f"Unable to open PDF: {exc}") from exc

    pages: List[ExtractedPage] = []
    empty_pages = 0

    try:
        for index, page in enumerate(document, start=1):
            text = _extract_page_text(page)
            if not text:
                empty_pages += 1
                # Extension point for OCR:
                # text = ocr_page(page)
            pages.append(ExtractedPage(page_number=index, text=text))
    finally:
        document.close()

    total_chars = sum(len(page.text) for page in pages)
    metadata = {
        "page_count": len(pages),
        "empty_pages": empty_pages,
        "character_count": total_chars,
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
