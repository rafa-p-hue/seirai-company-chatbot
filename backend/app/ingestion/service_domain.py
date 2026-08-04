"""Generic municipal / handbook service-domain detection.

Domains are inferred from document titles, headings, and body text.
No city-specific facts, filenames, prices, or office windows are hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.models.api import ExtractedPage

# Stable domain ids used in payloads and ranking.
RESIDENT_REGISTRATION = "resident_registration"
HEALTH_INSURANCE = "health_insurance"
CERTIFICATE_FEES = "certificate_fees"
WASTE_RECYCLING = "waste_recycling"
CHILDCARE_SUPPORT = "childcare_support"
DISASTER_PREPAREDNESS = "disaster_preparedness"
GENERAL = "general"

SERVICE_DOMAINS: Tuple[str, ...] = (
    RESIDENT_REGISTRATION,
    HEALTH_INSURANCE,
    CERTIFICATE_FEES,
    WASTE_RECYCLING,
    CHILDCARE_SUPPORT,
    DISASTER_PREPAREDNESS,
    GENERAL,
)

# Cue patterns learned generically from titles/headings/body — not filenames.
DOMAIN_CUES: Dict[str, Tuple[re.Pattern[str], ...]] = {
    RESIDENT_REGISTRATION: (
        re.compile(
            r"(?i)\b("
            r"moving\s+in|move-?in|moved\s+(?:to|into)|just\s+moved|"
            r"moving\s+out|resident\s+registration|"
            r"new\s+resident|register(?:ation)?\s+(?:as\s+a\s+)?resident|"
            r"change\s+of\s+address|residence\s+notification|"
            r"move-in\s+notification|citizen\s+registration|"
            r"must\s+i\s+register|when\s+must\s+i\s+register"
            r")\b"
        ),
    ),
    HEALTH_INSURANCE: (
        re.compile(
            r"(?i)\b("
            r"health\s+insurance|national\s+health\s+insurance|"
            r"medical\s+insurance|insurance\s+enrollment|"
            r"employer'?s?\s+insurance|leave\s+(?:my\s+)?employer|"
            r"enroll(?:ment)?\s+in\s+(?:health\s+)?insurance|"
            r"insurance\s+premium|"
            r"medical\s+costs?|patient\s+share|co-?payment|"
            r"cost[- ]?share|insured\s+treatment|"
            r"pay\s+(?:at\s+the\s+)?counter|percentage\s+paid|"
            r"share\s+of\s+medical"
            r")\b"
        ),
    ),
    CERTIFICATE_FEES: (
        re.compile(
            r"(?i)\b("
            r"residence\s+certificate|certificate\s+(?:fee|fees|issuance)|"
            r"certificates?\s+and\s+fees|fee\s+schedule|"
            r"counter\s+fee|kiosk\s+fee|"
            r"juminhyo|certificate\s+cost|issuance\s+fee|"
            r"family\s+register(?:\s+abstract)?"
            r")\b"
        ),
    ),
    WASTE_RECYCLING: (
        re.compile(
            r"(?i)\b("
            r"burnable\s+garbage|garbage\s+collection|waste\s+(?:and\s+)?recycling|"
            r"recycl(?:e|ing)|trash\s+(?:collection|pickup)|"
            r"collection\s+day|household\s+waste|refuse|"
            r"oversized\s+garbage|bulky\s+(?:waste|garbage|item)|"
            r"throw\s+away|dispose\s+of|disposal|"
            r"garbage\s+sticker|city\s+bags?|yellow\s+(?:city\s+)?bags?"
            r")\b"
        ),
    ),
    CHILDCARE_SUPPORT: (
        re.compile(
            r"(?i)\b("
            r"child\s+allowance|childcare\s+(?:support|allowance)|"
            r"child\s+benefit|parenting\s+support|"
            r"allowance\s+for\s+(?:children|child)|"
            r"nursery|childcare\s+application|"
            r"allowance\s+(?:paid|payment|payments)"
            r")\b"
        ),
    ),
    DISASTER_PREPAREDNESS: (
        re.compile(
            r"(?i)\b("
            r"disaster\s+preparedness|emergency\s+(?:kit|water|supplies)|"
            r"evacuation(?:\s+shelter)?|earthquake\s+preparedness|"
            r"emergency\s+stockpile|keep\s+for\s+an?\s+emergency|"
            r"disaster\s+guide|pet[- ]friendly\s+shelter|accepts?\s+pets?"
            r")\b"
        ),
    ),
}


@dataclass(frozen=True)
class DomainSignal:
    domain: str
    score: float
    matched: str = ""


def detect_service_domain(
    *,
    filename: str = "",
    pages: Sequence[ExtractedPage] | None = None,
    headings: Sequence[str] | None = None,
    text: str = "",
) -> str:
    """Return the strongest service domain for a document or query blob."""
    ranked = rank_service_domains(
        filename=filename, pages=pages, headings=headings, text=text
    )
    if not ranked:
        return GENERAL
    best = ranked[0]
    return best.domain if best.score >= 0.35 else GENERAL


def rank_service_domains(
    *,
    filename: str = "",
    pages: Sequence[ExtractedPage] | None = None,
    headings: Sequence[str] | None = None,
    text: str = "",
) -> List[DomainSignal]:
    """Score all domains against titles, headings, and body text."""
    title_blob = " ".join(
        part
        for part in (
            filename,
            *(headings or []),
            *((page.section_heading or "") for page in (pages or [])[:8]),
        )
        if part
    )
    body_parts: List[str] = [text]
    for page in (pages or [])[:12]:
        body_parts.append(page.text or "")
        body_parts.append(page.section_heading or "")
    body_blob = "\n".join(body_parts)
    scores: Dict[str, DomainSignal] = {}
    for domain, patterns in DOMAIN_CUES.items():
        title_hits = _count_hits(patterns, title_blob)
        body_hits = _count_hits(patterns, body_blob)
        if title_hits <= 0 and body_hits <= 0:
            continue
        # Titles/headings outweigh body so related procedures stay distinct.
        score = title_hits * 1.35 + min(2.4, body_hits * 0.55)
        matched = _first_match(patterns, title_blob) or _first_match(patterns, body_blob)
        scores[domain] = DomainSignal(domain=domain, score=score, matched=matched or "")
    return sorted(scores.values(), key=lambda item: item.score, reverse=True)


def extract_query_service_domain(question: str) -> str:
    """Identify the requested service domain from the full original question."""
    cleaned = (question or "").strip()
    if not cleaned:
        return GENERAL
    ranked = rank_service_domains(text=cleaned, filename="", headings=[])
    if not ranked:
        # Bare "register/bring" with move context falls through from ranking.
        if re.search(r"(?i)\b(moved|moving|move-?in)\b", cleaned) and re.search(
            r"(?i)\b(register|bring|notification)\b", cleaned
        ):
            return RESIDENT_REGISTRATION
        return GENERAL
    # Prefer insurance over registration when both employer-insurance and
    # register cues appear (leaving employer insurance ≠ moving in).
    domains = {item.domain: item for item in ranked}
    if HEALTH_INSURANCE in domains and RESIDENT_REGISTRATION in domains:
        insurance = domains[HEALTH_INSURANCE]
        resident = domains[RESIDENT_REGISTRATION]
        if re.search(
            r"(?i)\b(employer|insurance|enroll|left\s+my|leave\s+my)\b", cleaned
        ):
            if insurance.score + 0.4 >= resident.score:
                return HEALTH_INSURANCE
    if CHILDCARE_SUPPORT in domains and RESIDENT_REGISTRATION in domains:
        child = domains[CHILDCARE_SUPPORT]
        if re.search(r"(?i)\b(child\s+allowance|childcare|child\s+benefit)\b", cleaned):
            return CHILDCARE_SUPPORT if child.score >= 0.3 else ranked[0].domain
    if CERTIFICATE_FEES in domains and re.search(
        r"(?i)\b(residence\s+certificate|certificate\s+(?:fee|cost)|how\s+much)\b",
        cleaned,
    ):
        # Don't let bare "how much" steal health-insurance / childcare questions.
        if HEALTH_INSURANCE in domains and re.search(
            r"(?i)\b(health\s+insurance|medical\s+costs?|patient\s+share|co-?payment)\b",
            cleaned,
        ):
            return HEALTH_INSURANCE
        if CHILDCARE_SUPPORT in domains and re.search(
            r"(?i)\b(child\s+allowance|childcare|child\s+benefit)\b", cleaned
        ):
            return CHILDCARE_SUPPORT
        return CERTIFICATE_FEES
    if HEALTH_INSURANCE in domains and re.search(
        r"(?i)\b("
        r"national\s+health\s+insurance|medical\s+costs?|patient\s+share|"
        r"co-?payment|insured\s+treatment|share\s+of\s+medical"
        r")\b",
        cleaned,
    ):
        return HEALTH_INSURANCE
    best = ranked[0]
    return best.domain if best.score >= 0.3 else GENERAL


def domain_match_score(
    query_domain: Optional[str],
    candidate_domain: Optional[str],
    *,
    document_name: str = "",
    section_title: str = "",
    content: str = "",
) -> float:
    """Hierarchy-aware boost/penalty for service-domain alignment.

    Positive = match. Strongly negative = conflicting municipal procedure.
    """
    q = (query_domain or GENERAL).strip() or GENERAL
    c = (candidate_domain or "").strip()
    if not c or c == GENERAL:
        # Infer lightly from the chunk when metadata is missing (legacy vectors).
        c = detect_service_domain(
            filename=document_name,
            headings=[section_title] if section_title else [],
            text=content[:1200],
        )
    if q == GENERAL:
        return 0.0
    if c == q:
        return 1.35
    if c == GENERAL:
        return -0.15
    # Conflicting procedural domains — semantic similarity must not win, but this
    # remains a ranking penalty (not a hard drop). Exact-entity rescue can still
    # keep strong lexical matches eligible.
    return -1.25


def title_match_boost(question: str, document_name: str) -> float:
    """Soft boost when distinctive query tokens appear in the document title."""
    q_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", (question or "").lower())
        if token
        not in {
            "what",
            "when",
            "where",
            "which",
            "with",
            "from",
            "this",
            "that",
            "have",
            "must",
            "need",
            "bring",
            "after",
            "just",
            "moved",
            "city",
            "does",
            "much",
            "cost",
            "here",
        }
    }
    name = (document_name or "").lower()
    if not q_tokens or not name:
        return 0.0
    hits = sum(1 for token in q_tokens if token in name)
    if hits <= 0:
        return 0.0
    return min(0.55, 0.18 * hits)


def _count_hits(patterns: Iterable[re.Pattern[str]], text: str) -> int:
    if not text:
        return 0
    total = 0
    for pattern in patterns:
        total += len(pattern.findall(text))
    return total


def _first_match(patterns: Iterable[re.Pattern[str]], text: str) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text or "")
        if match:
            return match.group(0)
    return None


# Lightweight procedural metadata cues — generic, not city-specific.
_OFFICE_RE = re.compile(
    r"(?i)\b((?:citizen|resident|insurance|welfare|waste|disaster|city)\s+"
    r"(?:services?|desk|counter|office|division|department)|"
    r"(?:municipal|city)\s+hall(?:\s+\w+)?|"
    r"window\s+\d+)\b"
)
_DEADLINE_RE = re.compile(
    r"(?i)\b("
    r"within\s+\d+\s+(?:business\s+)?days?|"
    r"no\s+later\s+than\s+[^.]+|"
    r"deadline[:\s]+[^.]+|"
    r"must\s+(?:register|apply|enroll|submit|notify)\s+[^.]+"
    r")\b"
)
_REQUIRED_RE = re.compile(
    r"(?i)\b("
    r"bring\s+(?:your\s+)?[^.]+|"
    r"required\s+(?:documents?|items?)[:\s]+[^.]+|"
    r"you\s+must\s+bring\s+[^.]+"
    r")\b"
)
_ACTION_RE = re.compile(
    r"(?i)\b("
    r"register(?:ation)?|enroll(?:ment)?|apply|application|"
    r"submit|notify|notification|collect(?:ion)?|issu(?:e|ance)|"
    r"prepare|stockpile"
    r")\b"
)


def extract_chunk_procedure_metadata(
    *,
    content: str,
    section_title: Optional[str] = None,
    subsection_title: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Infer chunk-level procedure fields from heading + body text."""
    blob = "\n".join(
        part for part in (section_title or "", subsection_title or "", content or "") if part
    )
    office = None
    office_match = _OFFICE_RE.search(blob)
    if office_match:
        office = office_match.group(0).strip()[:120]
    deadline = None
    deadline_match = _DEADLINE_RE.search(blob)
    if deadline_match:
        deadline = deadline_match.group(0).strip()[:160]
    required = None
    required_match = _REQUIRED_RE.search(blob)
    if required_match:
        required = required_match.group(0).strip()[:200]
    procedure = None
    heading = (section_title or subsection_title or "").strip()
    if heading:
        procedure = heading[:120]
    else:
        action_match = _ACTION_RE.search(blob)
        if action_match:
            procedure = action_match.group(0).strip()[:80]
    return {
        "procedure_action": procedure,
        "responsible_office": office,
        "deadline": deadline,
        "required_items": required,
    }
