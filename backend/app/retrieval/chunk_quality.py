"""Generic chunk quality scoring for document-agnostic retrieval."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

IDENTITY_ONLY_RE = re.compile(
    r"(?im)^(?:\[[^\]]+\]\s*)?(?:"
    r"(?:[A-Za-z']+\s+)?(?:full\s*)?name|"
    r"(?:[A-Za-z']+\s+)?email(?:\s+address)?|"
    r"(?:[A-Za-z']+\s+)?minor(?:\s*\([^)]*\))?|"
    r"(?:[A-Za-z']+\s+)?major|"
    r"email|phone|address|chapter\s+name"
    r")\s*:\s*.+$"
)
SECTION_PROMPT_RE = re.compile(
    r"(?i)^(provide a list of|identify any|list all|describe any|please provide|"
    r"awards and/or honors|awards/honors)\b"
)
AWARDS_HEADING_RE = re.compile(
    r"(?i)^(awards?|honors?|recognitions?|certifications?)\b"
)
LINK_LABEL_RE = re.compile(
    r"(?i)^(https?://|www\.|linkedin\.com|connect|follow|message)\b"
)
DATE_RE = re.compile(
    r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|"
    r"\b(?:19|20)\d{2}\b|\b\d+\s+years?\b"
)
ROLE_SIGNAL_RE = re.compile(
    r"(?i)\b("
    r"intern|internship|assistant|advisor|fellow|founder|chair|lead|leader|"
    r"manager|managers|coordinator|researcher|engineer|analyst|officer|officers|"
    r"educator|staff|resident|orientation|volunteer|director|president|secretary|"
    r"technician|operator|operators|buddy"
    r")\b"
)
ORG_SIGNAL_RE = re.compile(
    r"(?i)\b(lab|center|centre|company|organization|fraternity|society|program|university|college)\b"
)
RESPONSIBILITY_RE = re.compile(
    r"(?i)( – | — | \? |: ).{15,}|"
    r"\b(responsible for|helped|supported|conduct|conducted|organized|designed|"
    r"built|developed|led|assist|assisted|managed|coordinated|collected|trained|"
    r"established|welcome|guide|mounts|runs the)\b"
)


def experience_quality(content: str, payload: Dict[str, Any] | None = None) -> Tuple[float, str]:
    """Return (boost_or_penalty, reason) for role/experience ranking."""
    text = (content or "").strip()
    if not text:
        return -1.0, "empty"

    record_type = str((payload or {}).get("record_type") or "")
    lower = text.lower()

    # Reject / heavily down-rank weak identity and chrome chunks.
    if LINK_LABEL_RE.match(text) or text.lower() in {"connect", "follow", "message"}:
        return -0.9, "link_label"
    if SECTION_PROMPT_RE.match(text):
        return -0.85, "section_prompt"
    if AWARDS_HEADING_RE.match(text) and len(text) < 80 and " – " not in text:
        return -0.7, "awards_heading"
    if IDENTITY_ONLY_RE.match(text) and not ROLE_SIGNAL_RE.search(text):
        return -0.8, "identity_or_contact_field"
    if record_type == "profile" and not ROLE_SIGNAL_RE.search(text):
        return -0.65, "profile_metadata"
    if record_type == "education" and not ROLE_SIGNAL_RE.search(text):
        return -0.5, "education_metadata"
    if lower.startswith("record type: profile") and "description:" in lower:
        # Thin profile wrappers without real responsibilities.
        desc = lower.split("description:", 1)[-1]
        if len(desc.strip()) < 40 and not ROLE_SIGNAL_RE.search(desc):
            return -0.7, "thin_profile_wrapper"
    # Isolated short name / fragment
    if len(text.split()) <= 4 and not ROLE_SIGNAL_RE.search(text) and ":" not in text:
        return -0.75, "isolated_name_or_fragment"
    if len(text) < 35 and not RESPONSIBILITY_RE.search(text):
        return -0.55, "incomplete_fragment"

    boost = 0.0
    reason = "neutral"
    if RESPONSIBILITY_RE.search(text):
        boost += 0.35
        reason = "responsibility_description"
    if ROLE_SIGNAL_RE.search(text):
        boost += 0.25
        reason = "role_title"
    if DATE_RE.search(text):
        boost += 0.12
    if ORG_SIGNAL_RE.search(text):
        boost += 0.1
    if " – " in text or " — " in text or " ? " in text:
        boost += 0.15
        reason = "list_entry_with_description"
    if record_type in {"experience", "internship", "research", "leadership"}:
        boost += 0.12
    if record_type == "universal" and ROLE_SIGNAL_RE.search(text):
        boost += 0.08
    return min(0.9, boost), reason


def is_strong_experience_evidence(content: str) -> bool:
    """True when chunk contains a complete role/activity/research description."""
    text = strip_leading_form_noise(content or "")
    if len(text) < 40:
        return False
    if IDENTITY_ONLY_RE.match(text):
        return False
    if SECTION_PROMPT_RE.match(text) and len(text.split()) < 18:
        return False
    if AWARDS_HEADING_RE.match(text) and len(text) < 60:
        return False
    lower = text.lower()
    if lower.startswith("record type: profile"):
        desc = lower.split("description:", 1)[-1] if "description:" in lower else lower
        if not ROLE_SIGNAL_RE.search(desc) or len(desc.strip()) < 50:
            return False
    if lower.startswith("record type: education") and not ROLE_SIGNAL_RE.search(text):
        return False
    return bool(
        RESPONSIBILITY_RE.search(text)
        or (
            ROLE_SIGNAL_RE.search(text)
            and (DATE_RE.search(text) or ORG_SIGNAL_RE.search(text) or len(text) > 60)
        )
    )


def content_fingerprint(content: str) -> str:
    normalized = re.sub(r"\s+", " ", (content or "").lower()).strip()
    return normalized[:240]


def strip_leading_form_noise(content: str) -> str:
    """Remove leading form instructions / empty headings so useful body can score."""
    text = (content or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    while lines and (
        SECTION_PROMPT_RE.match(lines[0])
        or (AWARDS_HEADING_RE.match(lines[0]) and len(lines[0]) < 60)
        or LINK_LABEL_RE.match(lines[0])
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


def broad_quality(content: str, payload: Dict[str, Any] | None = None) -> Tuple[float, str]:
    """Quality score for broad summary / general questions."""
    raw = (content or "").strip()
    if not raw:
        return -1.0, "empty"
    if LINK_LABEL_RE.match(raw) and "\n" not in raw:
        return -0.9, "link_label"
    text = strip_leading_form_noise(raw) or raw
    if SECTION_PROMPT_RE.match(text) and len(text.split()) < 18:
        return -0.9, "form_instruction"
    if AWARDS_HEADING_RE.match(text) and len(text) < 60:
        return -0.55, "heading_without_content"
    if IDENTITY_ONLY_RE.match(text):
        # Isolated contact/label fields are weak for broad overviews.
        return -0.45, "isolated_contact_or_label"
    if len(text.split()) <= 3:
        return -0.7, "heading_or_fragment"

    boost = 0.0
    reason = "paragraph_or_entry"
    if RESPONSIBILITY_RE.search(text) or " – " in text or " — " in text or " - " in text:
        boost += 0.35
        reason = "complete_list_entry"
    if len(text) >= 120:
        boost += 0.2
        reason = "complete_paragraph"
    if ROLE_SIGNAL_RE.search(text):
        boost += 0.15
    if re.search(r"(?i)\b(education|major|university|college|degree)\b", text):
        boost += 0.12
    if re.search(r"(?i)\b(research|project|lab|fellow)\b", text):
        boost += 0.12
    if re.search(r"(?i)\b(music|ensemble|band|performance|chapter|fraternity|organization)\b", text):
        boost += 0.12
    record_type = str((payload or {}).get("record_type") or "")
    if record_type in {"experience", "internship", "research", "leadership", "education"}:
        boost += 0.08
    return min(0.85, boost), reason


def is_weak_broad_evidence(content: str) -> bool:
    text = strip_leading_form_noise(content or "")
    if not text:
        return True
    if LINK_LABEL_RE.match(text) or (
        SECTION_PROMPT_RE.match(text) and len(text.split()) < 18
    ):
        return True
    if IDENTITY_ONLY_RE.match(text):
        return True
    if len(text.split()) <= 3:
        return True
    return False
