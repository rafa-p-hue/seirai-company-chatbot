"""Generic query↔section heading matching and overview detection.

Document-agnostic: synonym families map questions to heading tokens without
hardcoding any organization's facts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


# Query cue → tokens that commonly appear in section headings.
SECTION_CUE_FAMILIES: Tuple[Tuple[re.Pattern[str], Tuple[str, ...]], ...] = (
    (
        re.compile(
            r"\b(membership|member(?:ship)?\s+cost|dues|join|annual\s+fee|member\s+fee)\b",
            re.I,
        ),
        ("membership", "member", "dues", "fees"),
    ),
    (
        re.compile(r"\b(pet|pets|animal|animals|dog|cat)\b", re.I),
        ("membership", "rules", "policies", "pet", "animals"),
    ),
    (
        re.compile(
            r"\b(volunteer|orientation|shift|volunteering)\b",
            re.I,
        ),
        ("volunteer", "volunteering", "orientation", "program"),
    ),
    (
        re.compile(
            r"\b(donat(?:e|ed|ion|ions)|food\s+donation|pounds|ton(?:s)?)\b",
            re.I,
        ),
        ("donation", "donations", "food", "program"),
    ),
    (
        re.compile(
            r"\b(compost|sustainab|recycl|organic\s+waste|green\s+waste)\b",
            re.I,
        ),
        ("sustainability", "compost", "practices", "recycling"),
    ),
    (
        re.compile(
            r"\b(rental|rentals|event\s+space|alcohol|venue|private\s+event)\b",
            re.I,
        ),
        ("event", "rentals", "rental", "venue", "alcohol"),
    ),
    (
        re.compile(
            r"\b(refund|cancellation|cancel|reimburs)\b",
            re.I,
        ),
        ("cancellation", "refund", "policy", "cancel"),
    ),
    (
        re.compile(
            r"\b(accessib(?:le|ility)|disabilit(?:y|ies)|wheelchair|ada|ramp)\b",
            re.I,
        ),
        ("accessibility", "accessible", "disability", "ada"),
    ),
    (
        re.compile(
            r"\b(future\s+plans?|opening\s+(?:date|period)|timeline|roadmap|"
            r"coming\s+soon|planned\s+opening|not\s+(?:yet\s+)?announced)\b",
            re.I,
        ),
        ("future", "plans", "timeline", "development", "opening"),
    ),
    (
        re.compile(r"\b(hours|open(?:ing)?\s+hours|schedule|when\s+open)\b", re.I),
        ("hours", "schedule", "operations"),
    ),
    (
        re.compile(r"\b(price|cost|fee|fees|pricing|how\s+much|\$)\b", re.I),
        ("membership", "fees", "pricing", "cost", "rates", "rentals"),
    ),
)

OVERVIEW_HEADING_RE = re.compile(
    r"(?i)\b("
    r"overview|about(?:\s+us)?|introduction|summary|welcome|mission|"
    r"company\s+overview|document\s+overview|general\s+information"
    r")\b"
)

OVERVIEW_CONTENT_RE = re.compile(
    r"(?i)^("
    r"overview|about(?:\s+us)?|introduction|welcome to|this (?:document|guide|handbook)"
    r")"
)


def section_heading_boost(question: str, payload: dict) -> float:
    """Strong boost when query cues align with section / subsection headings."""
    heading = str(payload.get("section_title") or "")
    subsection = str(payload.get("subsection_title") or "")
    content = str(payload.get("content") or "")
    blob = f"{heading} {subsection}".lower()
    if not blob.strip() and not content:
        return 0.0

    score = 0.0
    for pattern, tokens in SECTION_CUE_FAMILIES:
        if not pattern.search(question or ""):
            continue
        hits = sum(1 for token in tokens if token in blob)
        if hits:
            score += 0.35 + 0.12 * min(hits, 3)
        # Also reward cue tokens appearing early in content under a matching section.
        content_l = content.lower()
        if heading and any(token in heading.lower() for token in tokens):
            score += 0.2
        elif any(token in content_l[:200] for token in tokens):
            score += 0.08
    return min(1.25, score)


def is_overview_chunk(payload: dict) -> bool:
    heading = str(payload.get("section_title") or "")
    content = str(payload.get("content") or "").strip()
    if OVERVIEW_HEADING_RE.search(heading):
        return True
    if OVERVIEW_CONTENT_RE.match(content) and len(content) < 400:
        return True
    # Structured profile dumps often act as noisy overviews for policy Qs.
    record_type = str(payload.get("record_type") or "").lower()
    if record_type == "profile" and not payload.get("content_type"):
        return True
    return False


def overview_penalty(question: str, query_type: str, payload: dict) -> float:
    """Down-rank overview / profile chunks for specific fact questions."""
    if not is_overview_chunk(payload):
        return 0.0
    if query_type in {"summary", "identity", "greeting", "help"}:
        return 0.0
    # Broad "tell me about" stays neutral/positive via broad_quality.
    if query_type == "general" and re.search(
        r"(?i)\b(overview|summary|tell me about|about this document)\b",
        question or "",
    ):
        return 0.0
    return -0.85


def matched_section_tokens(question: str) -> List[str]:
    tokens: List[str] = []
    for pattern, family in SECTION_CUE_FAMILIES:
        if pattern.search(question or ""):
            tokens.extend(family)
    return tokens
