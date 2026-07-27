"""Strip internal retrieval/structured-record markers from user-facing text."""

from __future__ import annotations

import re
from typing import List


INTERNAL_LINE_RE = re.compile(
    r"(?i)^(?:record\s*type|person|section|profile\s*description|description|"
    r"employment\s*type|organization|title|dates|location|skills)\s*:\s*"
)
INTERNAL_INLINE_RE = re.compile(
    r"(?i)\b(?:record\s*type|person|section|profile\s*description)\s*:\s*[^\n;|]+"
)
BRACKET_HEADING_RE = re.compile(r"^\[([^\]]+)\]\s*")


def scrub_internal_metadata(text: str) -> str:
    """Remove structured-record scaffolding while keeping factual body text."""
    if not text:
        return ""
    # Normalize common labeled scaffolding even when flattened onto one line.
    text = re.sub(
        r"(?i)\brecord\s*type\s*:\s*\S+",
        " ",
        text,
    )
    text = re.sub(r"(?i)\bprofile\s*description\s*:\s*", " ", text)
    text = re.sub(r"(?i)\bperson\s*:\s*", " ", text)
    text = re.sub(r"(?i)\bsection\s*:\s*", " ", text)
    text = re.sub(r"(?i)\bdescription\s*:\s*", " ", text)
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if INTERNAL_LINE_RE.match(line):
            value = INTERNAL_LINE_RE.sub("", line).strip()
            if not value or re.match(
                r"(?i)^(profile|unknown|section|universal|experience|education)$",
                value,
            ):
                continue
            if re.match(r"(?i)^record\s*type\b", raw.strip()):
                continue
            lines.append(value)
            continue
        line = INTERNAL_INLINE_RE.sub("", line).strip(" ;|")
        if line:
            lines.append(line)
    cleaned = "\n".join(lines).strip() if lines else re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def looks_like_internal_metadata(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if re.match(r"(?i)^record\s*type\s*:", stripped):
        return True
    if re.match(r"(?i)^person\s*:", stripped) and len(stripped) < 80:
        return True
    if re.match(r"(?i)^profile\s*description\s*:", stripped):
        return True
    return False
