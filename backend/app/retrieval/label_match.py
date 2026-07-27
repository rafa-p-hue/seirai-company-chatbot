"""Generic labeled field matching for document-agnostic retrieval.

Maps query intents to label families (major, minor, email, name, …) without
document-specific facts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# Canonical label families → regex patterns that appear in field labels.
LABEL_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "name": (
        r"full\s*name",
        r"candidate\s*name",
        r"applicant\s*name",
        r"nominee'?s?\s*(?:full\s*)?name",
        r"lead\s*author",
        r"\bname\b",
    ),
    "major": (
        r"\bmajor\b",
        r"field\s*of\s*study",
        r"academic\s*program",
        r"degree\s*program",
        r"\bdegree\b",
    ),
    "minor": (r"\bminor\b", r"secondary\s*(?:field|major)"),
    "email": (r"\bemail\b", r"e-?mail\s*address", r"\bcontact\b"),
    "date": (r"\bdate\b", r"\bdeadline\b", r"due\s*date", r"effective\s*date"),
    "birthday": (r"\bbirthday\b", r"date\s*of\s*birth", r"birth\s*date", r"\bdob\b"),
    "gpa": (r"\bgpa\b", r"grade[- ]?point\s*average", r"cumulative\s*gpa"),
    "food": (r"favorite\s*food", r"favourite\s*food", r"food\s*preference", r"\bfood\b"),
    "policy": (r"\bpolicy\b", r"\bprocedure\b", r"\bguideline\b"),
    "product": (r"\bproduct\b", r"\bmodel\b", r"\bsku\b", r"part\s*number"),
    "phone": (r"\bphone\b", r"\bmobile\b", r"\btel\b"),
    "organization": (
        r"\borganization\b",
        r"\bcompany\b",
        r"\bemployer\b",
        r"\bschool\b",
        r"\buniversity\b",
        r"\bcollege\b",
        r"\binstitution\b",
        r"\bcampus\b",
    ),
}

# Query intent → preferred label families (ordered by priority).
INTENT_LABEL_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "education": ("major", "minor", "organization", "name"),
    "school": ("organization",),
    "major": ("major",),
    "gpa": ("gpa",),
    "birthday": ("birthday", "date"),
    "favorite_food": ("food",),
    "identity": ("name", "email", "major", "organization"),
    "summary": ("name", "major", "email", "organization"),
    "policy": ("policy", "date"),
    "product": ("product", "date"),
    "location": ("organization", "name"),
    "general": ("name", "email", "major", "policy", "product"),
}

# Query text cues → label families (for synonym matching beyond intent).
QUERY_FAMILY_CUES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(major|field of study|academic program|degree|stud(?:y|ies|ying))\b", re.I), "major"),
    (re.compile(r"\b(school|college|university|institution|attend)\b", re.I), "organization"),
    (re.compile(r"\bminor\b", re.I), "minor"),
    (re.compile(r"\b(email|contact|e-?mail)\b", re.I), "email"),
    (re.compile(r"\b(who is|who'?s|name|nominee)\b", re.I), "name"),
    (re.compile(r"\b(gpa|grade[- ]?point)\b", re.I), "gpa"),
    (re.compile(r"\b(birthday|date of birth|dob)\b", re.I), "birthday"),
    (re.compile(r"\b(favorite food|food preference|food)\b", re.I), "food"),
    (re.compile(r"\b(date|deadline|due)\b", re.I), "date"),
    (re.compile(r"\b(policy|procedure|guideline)\b", re.I), "policy"),
    (re.compile(r"\b(product|model|sku|specification)\b", re.I), "product"),
)

RESEARCH_NARRATIVE_RE = re.compile(
    r"(?i)\b(research\s+(?:topic|subject|focus|project)|thesis|dissertation|"
    r"studied\s+the|investigat(?:e|ion)|experiment)\b"
)
INSTRUMENT_RE = re.compile(
    r"(?i)\b(instrument|guitar|piano|violin|drums|flute|trumpet|cello|clarinet|saxophone|bass)\b"
)


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


def label_family(label: str) -> Optional[str]:
    text = normalize_label(label)
    if not text:
        return None
    for family, patterns in LABEL_FAMILIES.items():
        for pattern in patterns:
            if re.search(pattern, text, re.I):
                return family
    return None


def families_for_query(question: str, query_type: str) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for pattern, family in QUERY_FAMILY_CUES:
        if pattern.search(question or ""):
            if family not in seen:
                ordered.append(family)
                seen.add(family)
    for family in INTENT_LABEL_FAMILIES.get(query_type, ()):
        if family not in seen:
            ordered.append(family)
            seen.add(family)
    # Prefer major over minor when the question is about study generally.
    if "major" in seen and re.search(r"(?i)\bstud(?:y|ies|ying)\b", question or ""):
        ordered = ["major"] + [f for f in ordered if f != "major"]
        if "minor" in ordered and not re.search(r"(?i)\bminor\b", question or ""):
            # Keep minor as secondary for study questions, after major.
            ordered = [f for f in ordered if f != "minor"] + ["minor"]
    return ordered


def payload_label(payload: Dict[str, Any]) -> str:
    return str(payload.get("label") or payload.get("title") or "")


def payload_value(payload: Dict[str, Any]) -> str:
    value = payload.get("value")
    if value:
        return str(value).strip()
    content = str(payload.get("content") or "")
    # Strip optional [heading] prefix then parse Label: value
    content = re.sub(r"^\[[^\]]+\]\s*", "", content.strip())
    match = re.match(r"^([^:\n]{1,80}):\s*(.+)$", content, re.S)
    if match:
        return match.group(2).strip()
    return ""


def is_key_value_payload(payload: Dict[str, Any]) -> bool:
    if str(payload.get("content_type") or "") == "key_value":
        return True
    content = str(payload.get("content") or "")
    content = re.sub(r"^\[[^\]]+\]\s*", "", content.strip())
    return bool(re.match(r"^[^:\n]{1,80}:\s*\S+", content)) and "\n" not in content.strip()


def exact_label_match_score(
    *,
    question: str,
    query_type: str,
    payload: Dict[str, Any],
) -> Tuple[float, Optional[str]]:
    """Return (boost, matched_family) for exact/synonym label matches."""
    if not is_key_value_payload(payload):
        return 0.0, None
    label = payload_label(payload)
    if not label:
        content = re.sub(r"^\[[^\]]+\]\s*", "", str(payload.get("content") or "").strip())
        match = re.match(r"^([^:\n]{1,80}):\s*", content)
        label = match.group(1).strip() if match else ""
    family = label_family(label)
    if not family:
        return 0.0, None
    wanted = families_for_query(question, query_type)
    if family not in wanted:
        # Mild boost for any key-value when asking identity/summary.
        if query_type in {"identity", "summary"} and family in {"name", "email", "major"}:
            return 0.35, family
        return 0.0, None

    priority = wanted.index(family)
    # Exact label matches must outrank narrative: large boost, higher for top priority.
    boost = 0.95 - (0.08 * priority)
    value = payload_value(payload)
    if not value or value.lower() in {"n/a", "na", "none", "-", "tbd"}:
        return 0.05, family
    return boost, family


def education_research_penalty(content: str, query_type: str, question: str) -> float:
    """Down-rank research-topic narrative for academic study/major questions."""
    if query_type not in {"education", "major"}:
        return 0.0
    if re.search(r"(?i)\b(research experience|research role)\b", question or ""):
        return 0.0
    if is_key_value_payload({"content": content, "content_type": "key_value"}):
        return 0.0
    text = content or ""
    if RESEARCH_NARRATIVE_RE.search(text):
        return -0.55
    # "study/studies" in research assistant lines without a major label.
    if re.search(r"(?i)\b(research assistant|assist in studies|lab)\b", text):
        if not re.search(r"(?i)\b(major|degree|field of study)\s*:", text):
            return -0.4
    return 0.0


def collect_key_value_fields(payloads: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    for payload in payloads:
        if not is_key_value_payload(payload):
            continue
        label = payload_label(payload)
        if not label:
            content = re.sub(r"^\[[^\]]+\]\s*", "", str(payload.get("content") or "").strip())
            match = re.match(r"^([^:\n]{1,80}):\s*(.+)$", content, re.S)
            if match:
                label = match.group(1).strip()
        value = payload_value(payload)
        fields.append(
            {
                "label": label,
                "value": value,
                "family": label_family(label),
                "page_number": payload.get("page_number"),
                "surrounding_heading": payload.get("section_title"),
                "content": payload.get("content"),
                "chunk_id": payload.get("chunk_id"),
            }
        )
    return fields


def find_exact_label_payloads(
    payloads: Sequence[Dict[str, Any]],
    *,
    question: str,
    query_type: str,
) -> List[Dict[str, Any]]:
    """Scan all payloads for key-value fields matching the query labels."""
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for payload in payloads:
        boost, family = exact_label_match_score(
            question=question, query_type=query_type, payload=payload
        )
        if boost >= 0.35 and family:
            scored.append((boost, payload))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in scored]


def has_major_label_evidence(evidence: Sequence[Any]) -> bool:
    for item in evidence:
        payload = item if isinstance(item, dict) else _chunk_as_payload(item)
        boost, family = exact_label_match_score(
            question="major", query_type="education", payload=payload
        )
        if family in {"major", "minor"} and boost >= 0.35:
            return True
        content = str(payload.get("content") or "")
        if re.search(r"(?i)\b(major|degree|field of study)\s*:", content):
            return True
    return False


def extract_labeled_value(evidence: Sequence[Any], families: Sequence[str]) -> Optional[str]:
    for family in families:
        for item in evidence:
            payload = item if isinstance(item, dict) else _chunk_as_payload(item)
            label = payload_label(payload) or ""
            if label_family(label) == family:
                value = payload_value(payload)
                if value:
                    return value
            content = str(payload.get("content") or "")
            for pattern in LABEL_FAMILIES.get(family, ()):
                match = re.search(
                    rf"(?im)(?:^|\])\s*([^\n:]{{0,80}}?(?:{pattern})[^\n:]{{0,40}}):\s*(.+)$",
                    content,
                )
                if match:
                    return match.group(2).strip()
    return None


def _chunk_as_payload(item: Any) -> Dict[str, Any]:
    return {
        "content": getattr(item, "content", "") or "",
        "content_type": getattr(item, "content_type", None),
        "label": getattr(item, "label", None),
        "value": getattr(item, "value", None),
        "title": getattr(item, "title", None),
        "section_title": getattr(item, "section_title", None),
        "page_number": getattr(item, "page_number", None),
        "record_type": getattr(item, "record_type", None),
    }
