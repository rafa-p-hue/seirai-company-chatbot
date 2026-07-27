"""LinkedIn/resume-oriented cleaning helpers."""

from __future__ import annotations

import re
from typing import List, Sequence, Set

from app.models.api import ExtractedPage

UI_CHROME = re.compile(
    r"^(connect|follow|message|promoted|show more|show less|join now|sign in|"
    r"enhance with ai|profile language|private to you|who your viewers also viewed|"
    r"i'?m looking for.*|also follow|also viewed|join the conversation|terms apply|"
    r"subscribe for.*|money talks|find your welcome|hitachi is driving.*)$",
    re.I,
)

LANGUAGE_ONLY = re.compile(
    r"^(english|spanish|french|german|japanese|korean|chinese)$",
    re.I,
)

PAGE_COUNTER = re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.I)
SIDEBAR_DEGREE = re.compile(r"\b\d+(st|nd|rd|th)\b", re.I)
AD_OR_SIDEBAR = re.compile(
    r"\b(promoted|welcome offer|terms apply|subscribe for \$|other connections?|"
    r"connections? also|also follow|also viewed)\b",
    re.I,
)


def clean_linkedin_pages(pages: Sequence[ExtractedPage]) -> List[ExtractedPage]:
    """Remove LinkedIn UI chrome, ads, and repeated header/footer noise."""
    from app.ingestion.cleaner import clean_text, detect_repeated_headers_footers, normalize_line

    repeated = detect_repeated_headers_footers(pages)
    frequent = _frequent_noise_lines(pages)
    noise = set(repeated) | frequent
    cleaned: List[ExtractedPage] = []

    for page in pages:
        seen: Set[str] = set()
        lines: List[str] = []
        for raw in clean_text(page.text).split("\n"):
            line = raw.strip()
            if not line:
                continue
            normalized = normalize_line(line)
            if normalized in noise:
                continue
            reason = classify_noise_line(line)
            if reason:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            lines.append(line)
        cleaned.append(
            ExtractedPage(page_number=page.page_number, text="\n".join(lines).strip())
        )
    return cleaned


def classify_noise_line(line: str) -> str | None:
    normalized = re.sub(r"\s+", " ", line).strip().lower()
    if re.match(r"^https?://\S+$", line, re.I):
        return "url"
    if PAGE_COUNTER.match(normalized):
        return "page-counter"
    if re.search(r"\.(jpe?g|png|gif|webp|svg)(\s|$)", normalized, re.I):
        return "image-filename"
    if UI_CHROME.match(normalized) or LANGUAGE_ONLY.match(normalized):
        return "ui-chrome"
    if AD_OR_SIDEBAR.search(normalized) and len(normalized) < 140:
        return "ad-or-sidebar"
    if SIDEBAR_DEGREE.search(normalized) and len(normalized.split()) <= 6:
        return "sidebar-person"
    if len(normalized) <= 2 and not re.search(r"[A-Za-z]", normalized):
        return "empty-symbol"
    return None


def _frequent_noise_lines(pages: Sequence[ExtractedPage]) -> Set[str]:
    from app.ingestion.cleaner import normalize_line

    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    threshold = max(3, int(len(pages) * 0.4))
    for page in pages:
        unique = {
            normalize_line(line)
            for line in page.text.splitlines()
            if 2 < len(normalize_line(line)) < 80
        }
        for line in unique:
            counts[line] = counts.get(line, 0) + 1
    return {
        line
        for line, count in counts.items()
        if count >= threshold
        and (
            classify_noise_line(line)
            or line.startswith("http")
            or re.search(r"page\s+\d+|/\d+$", line)
            or len(line.split()) <= 4
        )
    }
