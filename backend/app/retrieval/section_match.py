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
            r"\b(leave|paid\s+leave|pto|vacation|time\s+off)\b",
            re.I,
        ),
        ("leave", "pto", "vacation", "handbook", "policy"),
    ),
    (
        re.compile(r"\b(remote\s+work|work\s+from\s+home|wfh|telework)\b", re.I),
        ("remote", "work", "handbook", "policy"),
    ),
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
    (
        re.compile(
            r"\b("
            r"register|registration|moving\s+in|move-?in(?:\s+notification)?|"
            r"move(?:d)?\s+in|move-in\s+notification|new\s+address|"
            r"new\s+resident|resident(?:s)?\s+registration|"
            r"citizen\s+affairs|change\s+of\s+address"
            r")\b",
            re.I,
        ),
        (
            "registration",
            "register",
            "moving",
            "move-in",
            "move",
            "notification",
            "resident",
            "residents",
            "address",
            "citizen",
            "affairs",
            "requirements",
        ),
    ),
    (
        re.compile(
            r"\b("
            r"health\s+insurance|national\s+health|nhi|employer(?:'?s)?\s+insurance|"
            r"insurance\s+enrollment|enroll(?:ment)?|premiums?|"
            r"insurance\s+(?:and|&)\s+pension|certificate\s+of\s+loss"
            r")\b",
            re.I,
        ),
        (
            "insurance",
            "enrollment",
            "enroll",
            "nhi",
            "premium",
            "pension",
            "employer",
            "medical",
        ),
    ),
    (
        re.compile(
            r"\b("
            r"bring|required\s+documents?|documents?\s+required|"
            r"what\s+to\s+bring|checklist|submit|provide"
            r")\b",
            re.I,
        ),
        (
            "required",
            "requirements",
            "documents",
            "registration",
            "checklist",
            "materials",
            "bring",
            "submit",
            "provide",
        ),
    ),
    (
        re.compile(
            r"\b(deadline|due\s+date|within\s+\d+\s+days?|within|days?|"
            r"before|after\s+moving)\b",
            re.I,
        ),
        ("registration", "deadline", "timeline", "requirements", "moving", "within", "days"),
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
    # Broad document-summary phrasing stays neutral/positive via broad_quality.
    if query_type == "general" and re.search(
        r"(?i)\b("
        r"overview|summary|tell me about|about this document|"
        r"what is this (?:document|file|guide|handbook) about|"
        r"what does this (?:document|file) cover|summarize"
        r")\b",
        question or "",
    ):
        return 0.0
    return -0.85


OFFICE_HOURS_RE = re.compile(
    r"(?i)\b(office hours?|opening hours?|business hours?|hours of operation)\b"
)
CONTACT_SECTION_RE = re.compile(
    r"(?i)\b(contact information|phone numbers?|email addresses?|directions?)\b"
)
DOCUMENT_TITLE_RE = re.compile(
    r"(?i)\b(guide|handbook|manual|brochure|services?\s+guide)\b"
)


def procedural_noise_penalty(question: str, query_type: str, payload: dict) -> float:
    """Down-rank titles, hours, contacts, and unrelated tables for procedure Qs."""
    if query_type not in {"date", "checklist", "policy", "price"}:
        return 0.0
    heading = f"{payload.get('section_title') or ''} {payload.get('subsection_title') or ''}"
    content = str(payload.get("content") or "")
    ctype = str(payload.get("content_type") or "")
    blob = f"{heading}\n{content}"
    penalty = 0.0
    if ctype == "heading" and len(content.split()) <= 8:
        penalty -= 0.9
    if OFFICE_HOURS_RE.search(blob) and not re.search(
        r"(?i)\b(hours|schedule|when .+ open)\b", question or ""
    ):
        penalty -= 0.95
    if CONTACT_SECTION_RE.search(heading):
        penalty -= 0.7
    if (
        query_type in {"date", "checklist"}
        and ctype in {"table", "structured_table_row"}
        and not re.search(r"(?i)\b(fee|cost|price|certificate)\b", question or "")
    ):
        penalty -= 0.55
    # Bare document titles never answer procedural facts.
    if ctype == "heading" and DOCUMENT_TITLE_RE.search(content) and len(content.split()) <= 8:
        penalty -= 0.85
    return penalty


def summary_section_boost(payload: dict) -> float:
    """Prefer overview/purpose/major section content for document summaries."""
    heading = f"{payload.get('section_title') or ''} {payload.get('subsection_title') or ''}"
    content = str(payload.get("content") or "")
    ctype = str(payload.get("content_type") or "")
    score = 0.0
    if is_overview_chunk(payload) or OVERVIEW_HEADING_RE.search(heading):
        score += 0.55
    if re.search(
        r"(?i)\b(purpose|this (?:guide|document|handbook) (?:explains|covers|describes))\b",
        content,
    ):
        score += 0.45
    if ctype == "heading" and len(content.split()) <= 6:
        score += 0.2  # major headings help coverage lists
    if OFFICE_HOURS_RE.search(f"{heading}\n{content}") or CONTACT_SECTION_RE.search(heading):
        score -= 0.85
    return score


def matched_section_tokens(question: str) -> List[str]:
    tokens: List[str] = []
    for pattern, family in SECTION_CUE_FAMILIES:
        if pattern.search(question or ""):
            tokens.extend(family)
    return tokens
