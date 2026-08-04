"""Detect primary entities from document content and filenames (no hardcoded people)."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from app.models.api import DocumentChunk, ExtractedPage

NAME_LABEL_RE = re.compile(
    r"(?:full\s*name|candidate\s*name|applicant\s*name|employee\s*name|author)"
    r"\s*:\s*([^\n:]+?)(?=\s+[A-Z][A-Za-z][^:\n]{0,40}:|$)",
    re.I,
)
EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
ORG_LABEL_RE = re.compile(
    r"(?:organization|company|employer|institution|university|school)\s*:\s*([^\n]+)",
    re.I,
)


def detect_primary_entities(
    *,
    pages: Sequence[ExtractedPage],
    document_name: str,
    chunks: Sequence[DocumentChunk] | None = None,
) -> List[str]:
    """Return likely subject names/entities found in the document."""
    entities: List[str] = []

    for page in pages:
        for line in page.text.splitlines():
            match = NAME_LABEL_RE.search(line.strip())
            if match:
                value = _clean_entity(match.group(1))
                if value:
                    entities.append(value)
            # First non-empty title-case line can be a profile name heading.
            stripped = line.strip()
            if PERSON_NAME_RE.fullmatch(stripped) and 2 <= len(stripped.split()) <= 4:
                entities.append(stripped)

    stem = re.sub(r"\.[Pp][Dd][Ff]$", "", document_name)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    left = re.split(
        r"\b(?:applicant|info|resume|cv|linkedin|manual|policy|report|handbook)\b",
        stem,
        flags=re.I,
    )[0].strip(" -_|")
    if PERSON_NAME_RE.fullmatch(left) or (
        2 <= len(left.split()) <= 4 and left[:1].isupper()
    ):
        cleaned = _clean_entity(left)
        if cleaned:
            entities.append(cleaned)

    if chunks:
        for chunk in chunks:
            if chunk.person_name:
                entities.append(_clean_entity(chunk.person_name))
            match = NAME_LABEL_RE.search(chunk.content)
            if match:
                entities.append(_clean_entity(match.group(1)))
            if chunk.organization:
                org = _clean_entity(chunk.organization, allow_long=True)
                if org:
                    entities.append(org)

    seen = set()
    unique: List[str] = []
    for entity in entities:
        key = entity.lower()
        if not entity or key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


def detect_document_headings(pages: Sequence[ExtractedPage], limit: int = 12) -> List[str]:
    headings: List[str] = []
    for page in pages:
        for line in page.text.splitlines():
            line = line.strip()
            if not line or len(line) > 90 or line.endswith("."):
                continue
            if re.match(r"^[A-Z][A-Za-z0-9 /&-]{2,80}$", line):
                headings.append(line)
            if len(headings) >= limit:
                return headings
    return headings


def resolve_subject_name(
    *,
    question: str,
    history_text: str = "",
    entities: Sequence[str],
    document_name: str = "",
) -> Optional[str]:
    """Pick a subject for short-query expansion from question/history/entities."""
    # Prefer the current question only for "who is …" — scanning history caused
    # fully-specified questions to inherit subjects like "the CEO now".
    match = re.search(
        r"who is\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\??",
        question or "",
    )
    if match:
        return match.group(1).strip()

    q = question.strip().rstrip("?")
    for entity in entities:
        first = entity.split()[0]
        if q.lower() == entity.lower() or q.lower() == first.lower():
            return entity
        if first.lower() in q.lower() and len(q.split()) <= 5:
            return entity

    # History may still help short elliptical queries that name nobody.
    if len(q.split()) <= 5 and history_text:
        hist_match = re.search(
            r"who is\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\??",
            history_text,
        )
        if hist_match and len(q.split()) <= 4:
            return hist_match.group(1).strip()

    if entities:
        return entities[0]

    from_name = detect_primary_entities(pages=[], document_name=document_name)
    return from_name[0] if from_name else None


def _clean_entity(value: str, *, allow_long: bool = False) -> str:
    value = value.strip().strip(" .,-")
    value = re.sub(r"\s+", " ", value)
    value = EMAIL_RE.sub("", value).strip()
    value = re.split(r"\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*:", value, maxsplit=1)[
        0
    ].strip()
    max_words = 8 if allow_long else 5
    if len(value.split()) > max_words or len(value) < 2:
        return ""
    return value
