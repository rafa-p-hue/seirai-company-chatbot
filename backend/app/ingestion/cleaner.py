from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from app.models.api import ExtractedPage

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = text.replace("\u0000", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: Sequence[ExtractedPage]) -> List[ExtractedPage]:
    repeated = detect_repeated_headers_footers(pages)
    cleaned: List[ExtractedPage] = []

    for page in pages:
        lines = []
        seen = set()
        for raw_line in clean_text(page.text).split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            normalized = normalize_line(line)
            if normalized in repeated:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            lines.append(line)

        cleaned.append(
            ExtractedPage(page_number=page.page_number, text="\n".join(lines).strip())
        )

    return cleaned


def detect_repeated_headers_footers(
    pages: Sequence[ExtractedPage],
) -> set[str]:
    if len(pages) < 3:
        return set()

    counts: Counter[str] = Counter()
    threshold = max(3, int(len(pages) * 0.6))

    for page in pages:
        lines = [normalize_line(line) for line in page.text.splitlines() if line.strip()]
        candidates = set(lines[:3] + lines[-3:])
        for candidate in candidates:
            if 2 < len(candidate) <= 120:
                counts[candidate] += 1

    repeated = {line for line, count in counts.items() if count >= threshold}
    if repeated:
        logger.info("Removed %s repeated header/footer line(s)", len(repeated))
    return repeated


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def estimate_tokens(text: str) -> int:
    # Lightweight approximation for chunk sizing without requiring tiktoken at import time.
    return max(1, len(text.split()))
