"""Deterministic grounded answers for common query types."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import CitationSource, RetrievedChunk, SourceType
from app.retrieval.query_understanding import QueryUnderstanding

GREETING_ANSWER = "Hello! What would you like to know about this document?"

NAV_NOISE = re.compile(
    r"(i'?m looking for|enhance with ai|profile language|promoted|show more|"
    r"who your viewers also viewed|connect|follow)",
    re.I,
)


def compose_answer(
    *,
    understanding: QueryUnderstanding,
    evidence: Sequence[RetrievedChunk],
) -> Tuple[str, List[CitationSource]]:
    if understanding.query_type == "greeting":
        return GREETING_ANSWER, []

    if understanding.query_type == "unsupported":
        return FALLBACK_ANSWER, []

    cleaned = [item for item in evidence if item.content and not NAV_NOISE.search(item.content)]
    if not cleaned:
        return FALLBACK_ANSWER, []

    if understanding.query_type == "identity":
        answer = _identity_answer(cleaned)
    elif understanding.query_type == "education":
        answer = _education_answer(understanding, cleaned)
    elif understanding.query_type in {"experience", "internship", "research", "leadership"}:
        answer = _experience_answer(understanding, cleaned)
    elif understanding.query_type == "skills":
        answer = _skills_answer(cleaned)
    else:
        answer = _specific_answer(understanding, cleaned)

    if not answer or NAV_NOISE.search(answer):
        return FALLBACK_ANSWER, []

    sources = _sources_for(cleaned[:3])
    return answer.strip(), sources


def _identity_answer(evidence: Sequence[RetrievedChunk]) -> str:
    profile = _first_of_types(evidence, {"profile"})
    education = _first_of_types(evidence, {"education"})
    experience = [item for item in evidence if (item.record_type or "") in {"experience", "internship"}][:2]

    parts: List[str] = []
    name = None
    if profile:
        name = profile.person_name or _extract_name(profile.content)
        if name:
            parts.append(name)
        headline = _first_line(profile.content)
        if headline and headline.lower() != (name or "").lower():
            if not headline.lower().startswith("record type"):
                parts.append(headline)
    if education:
        school = education.organization or _field(education.content, "Organization")
        major = education.title or _field(education.content, "Title")
        edu_bits = [bit for bit in [school, major] if bit]
        if edu_bits:
            parts.append("Education: " + ", ".join(edu_bits))
    roles = []
    for item in experience:
        title = item.title or _field(item.content, "Title")
        org = item.organization or _field(item.content, "Organization")
        if title and org:
            roles.append(f"{title} at {org}")
        elif title or org:
            roles.append(title or org or "")
    if roles:
        parts.append("Experience includes " + "; ".join(roles) + ".")

    if not parts:
        return _clip_sentences(evidence[0].content, 2)
    # Build 1-3 concise sentences
    if name and len(parts) >= 2:
        rest = " ".join(parts[1:])
        return f"{name} — {rest}"
    return " ".join(parts)


def _education_answer(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    education = [
        item
        for item in evidence
        if (item.record_type or "") in {"education", "profile"}
    ] or list(evidence)
    primary = education[0]
    school = primary.organization or _field(primary.content, "Organization")
    major = primary.title or _field(primary.content, "Title")
    q = understanding.expanded_question.lower()
    name = understanding.subject_name or primary.person_name or "This person"

    if "major" in q or "field of study" in q or "stud" in q:
        if major and not _is_school_only(major):
            return f"{name}'s major or field of study is {major}."
        # Search other education records
        for item in education:
            candidate = item.title or _field(item.content, "Title")
            if candidate and not _is_school_only(candidate):
                return f"{name}'s major or field of study is {candidate}."
        return FALLBACK_ANSWER

    if school:
        return f"{name} attends {school}." if "attend" in q or "school" in q else f"School: {school}."
    return FALLBACK_ANSWER


def _is_school_only(text: str) -> bool:
    return bool(re.search(r"\b(university|college|school|institute)\b", text, re.I)) and not bool(
        re.search(
            r"\b(bachelor|master|b\.?s|m\.?s|major|degree|science|arts|engineering)\b",
            text,
            re.I,
        )
    )

def _experience_answer(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    wanted = {
        "experience": {"experience", "internship", "research", "leadership"},
        "internship": {"internship"},
        "research": {"research", "experience"},
        "leadership": {"leadership", "experience"},
    }.get(understanding.query_type, {"experience", "internship"})
    items = [item for item in evidence if (item.record_type or "experience") in wanted] or list(evidence)
    lines = []
    for item in items[:5]:
        title = item.title or _field(item.content, "Title")
        org = item.organization or _field(item.content, "Organization")
        dates = item.section_title if False else _field(item.content, "Dates")
        bit = " — ".join(part for part in [title, org, dates] if part)
        if bit:
            lines.append(bit)
    if not lines:
        return FALLBACK_ANSWER
    prefix = {
        "internship": "Internship experience includes:",
        "research": "Research experience includes:",
        "leadership": "Leadership/involvement includes:",
    }.get(understanding.query_type, "Work experience includes:")
    return prefix + " " + "; ".join(lines) + "."


def _skills_answer(evidence: Sequence[RetrievedChunk]) -> str:
    for item in evidence:
        if (item.record_type or "") == "skills" or "Skills:" in item.content:
            skills = _field(item.content, "Skills") or item.content
            skills = re.sub(r"(?i)^skills:\s*", "", skills)
            return f"Listed skills include: {_clip(skills, 240)}"
    return FALLBACK_ANSWER


def _specific_answer(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    text = evidence[0].content
    # Prefer description lines over metadata dump
    if "Description:" in text:
        text = text.split("Description:", 1)[1].strip()
    return _clip_sentences(text, 2)


def _sources_for(evidence: Sequence[RetrievedChunk]) -> List[CitationSource]:
    sources: List[CitationSource] = []
    for index, item in enumerate(evidence, start=1):
        sources.append(
            CitationSource(
                number=index,
                document_name=item.document_name,
                page_number=item.page_number,
                source_url=item.source_url,
                source_type=SourceType.website if item.source_url else SourceType.pdf,
            )
        )
    return sources


def _first_of_types(evidence: Sequence[RetrievedChunk], types: set) -> RetrievedChunk | None:
    for item in evidence:
        if (item.record_type or "") in types:
            return item
    return None


def _field(content: str, label: str) -> str | None:
    match = re.search(rf"(?im)^{re.escape(label)}:\s*(.+)$", content)
    return match.group(1).strip() if match else None


def _extract_name(content: str) -> str | None:
    return _field(content, "Person") or _field(content, "Title")


def _first_line(content: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("record type") or line.lower().startswith("person:"):
            continue
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
        return line
    return ""


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clip_sentences(text: str, count: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:count]).strip()
