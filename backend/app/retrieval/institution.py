"""Generic institution mention extraction (names and acronyms).

No document-specific institution names are hardcoded. Answers preserve the
institution text exactly as written in evidence and never invent expansions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from app.models.api import RetrievedChunk


INSTITUTION_WORD_RE = re.compile(
    r"(?i)\b("
    r"university|college|school|institute|institution|academy|campus|"
    r"polytechnic|conservatory"
    r")\b"
)
# Common non-institution all-caps tokens (generic, not document-specific).
ACRONYM_STOPWORDS = {
    "THE",
    "AND",
    "FOR",
    "PDF",
    "GPA",
    "CEO",
    "CTO",
    "COO",
    "CFO",
    "USA",
    "UTC",
    "HTML",
    "HTTP",
    "HTTPS",
    "JSON",
    "API",
    "FAQ",
    "TODO",
    "NOTE",
    "PAGE",
    "SECTION",
    "EMAIL",
    "PHONE",
    "NAME",
    "MAJOR",
    "MINOR",
    "DATE",
    "YEAR",
}

LABEL_INSTITUTION_RE = re.compile(
    r"(?im)^(?:\[[^\]]+\]\s*)?(?P<label>[^:\n]{0,80}?\b(?:"
    r"school|university|college|institution|campus|academy|institute"
    r")\b[^:\n]{0,40}):\s*(?P<value>.+)$"
)
AT_PHRASE_RE = re.compile(
    r"(?i)\b(?:(?:while|when)\s+)?(?:at|attend(?:s|ed|ing)?(?:\s+(?:the|a))?"
    r"|student\s+at|studying\s+at|enrolled\s+at)\s+"
    r"(?P<inst>(?:[A-Z]{2,6})|(?:[A-Z][A-Za-z0-9&.\'\-]+(?:\s+[A-Z][A-Za-z0-9&.\'\-]+){0,6}))"
)
ORG_LINE_RE = re.compile(
    r"(?im)^(?:organization|employer|institution)\s*:\s*(?P<value>.+)$"
)
EMAIL_DOMAIN_RE = re.compile(
    r"(?i)\b[A-Z0-9._%+-]+@(?P<domain>[A-Z0-9-]+)\.(?:edu|ac\.[A-Z]{2})\b"
)
SECTION_MARKER_RE = re.compile(r"^\[[^\]]+\]\s*")


def strip_section_markers(text: str) -> str:
    """Remove ingestion markers like [Recruitment] from user-facing text."""
    cleaned = SECTION_MARKER_RE.sub("", (text or "").strip())
    cleaned = re.sub(r"\s*\[[A-Za-z][^\]]{0,40}\]\s*", " ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def looks_like_institution_acronym(token: str) -> bool:
    text = (token or "").strip()
    if not re.fullmatch(r"[A-Z]{2,6}", text):
        return False
    return text not in ACRONYM_STOPWORDS


def looks_like_institution_name(text: str) -> bool:
    value = (text or "").strip().strip(" .,;")
    if not value or len(value) < 2:
        return False
    if looks_like_institution_acronym(value):
        return True
    if INSTITUTION_WORD_RE.search(value):
        return True
    # Multi-word proper-case organization names without the word "university"
    # are accepted only when labeled elsewhere; here require an institution word
    # or acronym.
    return False


def extract_institution_candidates(
    content: str, *, allow_email_weak: bool = True
) -> List[Tuple[str, str]]:
    """Return (institution_text, strength) pairs. strength is 'strong' or 'weak'."""
    text = strip_section_markers(content or "")
    if not text:
        return []
    found: List[Tuple[str, str]] = []
    seen = set()

    def add(value: str, strength: str) -> None:
        cleaned = value.strip(" \t\n\r.,;:|-")
        cleaned = strip_section_markers(cleaned)
        # Drop trailing degree / program suffixes accidentally captured in prose.
        cleaned = re.sub(
            r"(?i)\s+(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?|Bachelor|Master).*$",
            "",
            cleaned,
        ).strip(" ,;")
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        if strength == "strong" and not (
            looks_like_institution_acronym(cleaned)
            or INSTITUTION_WORD_RE.search(cleaned)
            or looks_like_institution_name(cleaned)
        ):
            # Labeled school values are always accepted as written.
            if not re.search(
                r"(?i)\b(school|university|college|institution|campus)\b", text
            ):
                return
        seen.add(key)
        found.append((cleaned, strength))

    for match in LABEL_INSTITUTION_RE.finditer(text):
        add(match.group("value"), "strong")

    for match in ORG_LINE_RE.finditer(text):
        value = match.group("value").strip()
        if looks_like_institution_acronym(value) or INSTITUTION_WORD_RE.search(value):
            add(value, "strong")

    for match in AT_PHRASE_RE.finditer(text):
        add(match.group("inst"), "strong")

    # Bare institution phrases in prose (not only labeled fields).
    # Prefer phrases that begin with the institution word to avoid absorbing names.
    for match in re.finditer(
        r"(?i)\b("
        r"(?:University|College|Institute|Academy|Polytechnic|Conservatory)"
        r"(?:\s+of)?"
        r"(?:[\s,]+[A-Z][A-Za-z0-9&.\'\-]+){0,6}"
        r")",
        text,
    ):
        add(match.group(1), "strong")

    # Standalone acronyms near institution context words.
    if INSTITUTION_WORD_RE.search(text) or re.search(
        r"(?i)\b(student|attend|campus|enrolled|alumni)\b", text
    ):
        for match in re.finditer(r"\b([A-Z]{2,6})\b", text):
            token = match.group(1)
            if looks_like_institution_acronym(token):
                add(token, "strong")

    if allow_email_weak:
        for match in EMAIL_DOMAIN_RE.finditer(text):
            domain = match.group("domain")
            if domain and looks_like_institution_acronym(domain.upper()):
                add(domain.upper(), "weak")
            elif domain:
                # Keep domain token as weak only when short enough to be an acronym-like slug.
                slug = re.sub(r"[^A-Za-z]", "", domain)
                if 2 <= len(slug) <= 6:
                    add(slug.upper(), "weak")

    return found


def content_has_institution_evidence(content: str) -> bool:
    candidates = extract_institution_candidates(content, allow_email_weak=True)
    return any(strength == "strong" for _, strength in candidates)


def extract_best_institution(
    evidence: Sequence[RetrievedChunk],
) -> Optional[str]:
    """Pick the strongest institution mention exactly as written in evidence."""
    strong: List[str] = []
    weak: List[str] = []
    for item in evidence:
        blob = "\n".join(
            part
            for part in (
                item.content or "",
                item.organization or "",
                item.value or "",
                f"{item.label}: {item.value}" if item.label and item.value else "",
            )
            if part
        )
        for value, strength in extract_institution_candidates(blob):
            if strength == "strong":
                strong.append(value)
            else:
                weak.append(value)
        if item.organization and (
            looks_like_institution_acronym(item.organization.strip())
            or INSTITUTION_WORD_RE.search(item.organization)
        ):
            strong.append(item.organization.strip())

    if strong:
        # Prefer labeled/fuller forms when both acronym and full name appear;
        # keep exact text — do not expand acronyms.
        strong_sorted = sorted(strong, key=lambda value: (-len(value), value))
        return strong_sorted[0]
    # Email domains alone are insufficient.
    return None


def answer_contains_institution(answer: str, evidence: Sequence[RetrievedChunk]) -> bool:
    institution = extract_best_institution(evidence)
    if institution and institution.lower() in (answer or "").lower():
        return True
    # Accept any strong candidate substring present in the answer.
    for item in evidence:
        for value, strength in extract_institution_candidates(item.content or ""):
            if strength == "strong" and value.lower() in (answer or "").lower():
                return True
    return bool(INSTITUTION_WORD_RE.search(answer or ""))
