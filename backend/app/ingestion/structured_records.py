"""Structure-aware record parsing for LinkedIn/resume-style PDFs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from app.models.api import ExtractedPage

SECTION_HEADINGS = {
    "experience",
    "education",
    "skills",
    "about",
    "summary",
    "profile",
    "projects",
    "project",
    "publications",
    "licenses",
    "certifications",
    "volunteer",
    "volunteering",
    "organizations",
    "involvement",
    "activities",
    "leadership",
    "research",
    "awards",
    "honors",
    "interests",
    "overview",
}

DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r"|\b\d{4}\s*[-–—]\s*(?:\d{4}|Present|Current)\b"
    r"|\b(?:Present|Current)\b",
    re.I,
)
EMPLOYMENT_RE = re.compile(
    r"\b(internship|intern|part-time|full-time|contract|freelance|self-employed|"
    r"on-site|onsite|remote|hybrid|seasonal|temporary)\b",
    re.I,
)
DEGREE_RE = re.compile(
    r"\b(bachelor|master|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|ph\.?d\.?|associate|"
    r"degree|major|minor|field of study)\b",
    re.I,
)
SCHOOL_RE = re.compile(
    r"\b(university|college|school|institute|academy)\b",
    re.I,
)


@dataclass
class StructuredRecord:
    record_id: str
    record_type: str
    person_name: Optional[str] = None
    title: Optional[str] = None
    organization: Optional[str] = None
    dates: Optional[str] = None
    location: Optional[str] = None
    description: str = ""
    heading: Optional[str] = None
    employment_type: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    source_page: Optional[int] = None
    page_end: Optional[int] = None

    def embedding_text(self) -> str:
        """Human-readable content for retrieval — no internal scaffold labels."""
        lines: List[str] = []
        if self.person_name:
            lines.append(self.person_name)
        if self.title:
            lines.append(self.title)
        if self.organization:
            lines.append(self.organization)
        if self.dates:
            lines.append(self.dates)
        if self.location:
            lines.append(self.location)
        if self.employment_type:
            lines.append(self.employment_type)
        if self.skills:
            lines.append(", ".join(self.skills))
        if self.description:
            lines.append(self.description)
        return "\n".join(line for line in lines if line).strip()

    def display_text(self) -> str:
        return self.embedding_text()


def parse_structured_records(
    pages: Sequence[ExtractedPage],
    *,
    document_id: str,
) -> List[StructuredRecord]:
    lines = _flatten_lines(pages)
    if not lines:
        return []
    # Skip structured resume parsing for handbook/policy-style documents.
    if not _looks_like_resume_or_profile(lines):
        return []

    person_name = _detect_person_name(lines)
    section = "Profile"
    drafts: List[dict] = []
    current: Optional[dict] = None

    def flush() -> None:
        nonlocal current
        if current and _is_meaningful(current):
            drafts.append(current)
        current = None

    def start(**kwargs) -> None:
        nonlocal current
        flush()
        current = {
            "record_type": kwargs.get("record_type", "unknown"),
            "title": kwargs.get("title"),
            "organization": kwargs.get("organization"),
            "dates": kwargs.get("dates"),
            "location": kwargs.get("location"),
            "description_lines": list(kwargs.get("description_lines") or []),
            "heading": kwargs.get("heading", section),
            "employment_type": kwargs.get("employment_type"),
            "skills": list(kwargs.get("skills") or []),
            "source_page": kwargs.get("source_page"),
            "page_end": kwargs.get("source_page"),
            "person_name": person_name,
        }

    i = 0
    while i < len(lines):
        text, page = lines[i]
        nxt = lines[i + 1][0] if i + 1 < len(lines) else ""
        lower = text.lower().strip()

        if _is_section_heading(text):
            flush()
            section = text.strip().title()
            i += 1
            continue

        if section.lower() in {"skills"} and "," in text:
            skills = [part.strip() for part in re.split(r"[,•·|]", text) if part.strip()]
            if skills:
                start(
                    record_type="skills",
                    title="Skills",
                    skills=skills,
                    description_lines=skills,
                    heading=section,
                    source_page=page,
                )
                flush()
            i += 1
            continue

        # Education: school line then degree/major lines
        if section.lower() == "education":
            if SCHOOL_RE.search(text) or _looks_like_org(text):
                desc = []
                dates = None
                title = None
                j = i + 1
                while j < len(lines) and not _is_section_heading(lines[j][0]):
                    candidate, cpage = lines[j]
                    if SCHOOL_RE.search(candidate) and j > i + 1:
                        break
                    if DATE_RE.search(candidate) and len(candidate) < 80:
                        dates = candidate
                    elif DEGREE_RE.search(candidate) or _looks_like_title(candidate):
                        title = candidate if not title else f"{title}; {candidate}"
                    elif EMPLOYMENT_RE.search(candidate):
                        pass
                    else:
                        if _looks_like_org(candidate) and j > i + 2:
                            break
                        desc.append(candidate)
                    j += 1
                start(
                    record_type="education",
                    organization=text,
                    title=title,
                    dates=dates,
                    description_lines=desc,
                    heading="Education",
                    source_page=page,
                )
                flush()
                i = j
                continue

        # Experience / research / leadership entries
        if section.lower() in {
            "experience",
            "research",
            "leadership",
            "projects",
            "volunteer",
            "volunteering",
            "organizations",
            "involvement",
            "activities",
        }:
            # Role pattern: Title, Organization, Dates, bullets...
            if _looks_like_title(text) or _looks_like_org(text):
                title = text if _looks_like_title(text) else None
                organization = None if title else text
                dates = None
                location = None
                employment = None
                desc: List[str] = []
                j = i + 1
                while j < len(lines) and not _is_section_heading(lines[j][0]):
                    candidate = lines[j][0]
                    nxt_line = lines[j + 1][0] if j + 1 < len(lines) else ""
                    # Boundary: another Title + (Org|Date) after we already captured a role.
                    if (
                        (title or organization)
                        and (dates or desc)
                        and _looks_like_title(candidate)
                        and (
                            _looks_like_org(nxt_line)
                            or DATE_RE.search(nxt_line)
                            or _looks_like_title(nxt_line)
                        )
                    ):
                        break
                    if DATE_RE.search(candidate) and len(candidate) < 90:
                        dates = candidate
                    elif EMPLOYMENT_RE.search(candidate) and len(candidate.split()) <= 6:
                        employment = candidate
                    elif _looks_like_location(candidate):
                        location = candidate
                    elif organization is None and _looks_like_org(candidate):
                        organization = candidate
                    elif title is None and _looks_like_title(candidate):
                        title = candidate
                    else:
                        desc.append(candidate.lstrip("•- ").strip())
                    j += 1
                record_type = _infer_type(section, employment, title, organization)
                if title and re.search(r"\bintern", title, re.I):
                    record_type = "internship"
                if (title and re.search(r"\bresearch", title, re.I)) or (
                    organization and re.search(r"\bresearch", organization, re.I)
                ):
                    if record_type != "internship":
                        record_type = "research"
                start(
                    record_type=record_type,
                    title=title,
                    organization=organization,
                    dates=dates,
                    location=location,
                    employment_type=employment,
                    description_lines=desc,
                    heading=section,
                    source_page=page,
                )
                flush()
                i = j
                continue

        # Profile / about prose near top
        if section.lower() in {"profile", "about", "summary", "overview"} or i < 8:
            if _looks_like_person_name(text) and person_name is None:
                person_name = text
            if current is None and (
                _looks_like_person_name(text) or lower.startswith("about")
            ):
                desc = [nxt] if nxt and not _is_section_heading(nxt) else []
                start(
                    record_type="profile",
                    title=text if _looks_like_person_name(text) else "Profile",
                    person_name=person_name or (text if _looks_like_person_name(text) else None),
                    description_lines=desc,
                    heading="Profile",
                    source_page=page,
                )
                if desc:
                    i += 2
                    flush()
                    continue
            if current and current.get("record_type") == "profile":
                current["description_lines"].append(text)
                current["page_end"] = page
                i += 1
                continue

        # Fallback: attach prose to current or start section record
        if current:
            current["description_lines"].append(text)
            current["page_end"] = page
        i += 1

    flush()

    records = [_to_record(document_id, index, draft) for index, draft in enumerate(drafts)]
    records = [record for record in records if record]

    # Ensure a profile record exists when we know the name.
    if person_name and not any(record.record_type == "profile" for record in records):
        education = next((r for r in records if r.record_type == "education"), None)
        profile = StructuredRecord(
            record_id=_stable_id(document_id, "profile", person_name),
            record_type="profile",
            person_name=person_name,
            title=person_name,
            organization=education.organization if education else None,
            description=_profile_description(person_name, records),
            heading="Profile",
            source_page=1,
        )
        records.insert(0, profile)
    elif person_name:
        for record in records:
            if record.record_type == "profile" and not record.person_name:
                record.person_name = person_name

    if not records:
        return _section_fallback(pages, document_id)
    return records


def _profile_description(person_name: str, records: Sequence[StructuredRecord]) -> str:
    """Clean profile summary without internal label scaffolding."""
    parts = [person_name] if person_name else []
    for record in records:
        if record.record_type == "education":
            bit = ", ".join(
                part
                for part in [record.organization, record.title, record.dates]
                if part
            )
            if bit:
                parts.append(bit)
        if record.record_type in {"experience", "internship"} and record.title:
            role = record.title
            if record.organization:
                role += f" at {record.organization}"
            parts.append(role)
            if len([p for p in parts if p != person_name]) >= 3:
                break
    return "\n".join(parts)


def _to_record(document_id: str, index: int, draft: dict) -> Optional[StructuredRecord]:
    description = "\n".join(draft.get("description_lines") or []).strip()
    title = draft.get("title")
    organization = draft.get("organization")
    if not any([title, organization, description, draft.get("skills")]):
        return None
    key = f"{draft.get('record_type')}:{title}:{organization}:{index}"
    return StructuredRecord(
        record_id=_stable_id(document_id, "record", key),
        record_type=draft.get("record_type") or "unknown",
        person_name=draft.get("person_name"),
        title=title,
        organization=organization,
        dates=draft.get("dates"),
        location=draft.get("location"),
        description=description,
        heading=draft.get("heading"),
        employment_type=draft.get("employment_type"),
        skills=list(draft.get("skills") or []),
        source_page=draft.get("source_page"),
        page_end=draft.get("page_end"),
    )


def _is_meaningful(draft: dict) -> bool:
    desc = draft.get("description_lines") or []
    if draft.get("record_type") == "profile" and (draft.get("title") or draft.get("person_name")):
        return True
    if draft.get("record_type") == "skills" and draft.get("skills"):
        return True
    if draft.get("organization") and (draft.get("title") or desc or draft.get("dates")):
        return True
    if draft.get("title") and (draft.get("dates") or desc):
        return True
    return len(" ".join(desc).split()) >= 8


def _section_fallback(
    pages: Sequence[ExtractedPage], document_id: str
) -> List[StructuredRecord]:
    records: List[StructuredRecord] = []
    for page in pages:
        if not page.text.strip():
            continue
        records.append(
            StructuredRecord(
                record_id=_stable_id(document_id, "section", f"{page.page_number}:{page.text[:40]}"),
                record_type="section",
                title=f"Page {page.page_number}",
                description=page.text.strip(),
                heading="Section",
                source_page=page.page_number,
            )
        )
    return records


def _flatten_lines(pages: Sequence[ExtractedPage]) -> List[tuple]:
    lines: List[tuple] = []
    for page in pages:
        for raw in page.text.splitlines():
            text = raw.strip()
            if text:
                lines.append((text, page.page_number))
    return lines


def _looks_like_resume_or_profile(lines: Sequence[tuple]) -> bool:
    """Require classic resume signals before emitting structured records."""
    blob = "\n".join(text for text, _ in lines[:80]).lower()
    resume_hits = sum(
        1
        for token in (
            "experience",
            "education",
            "skills",
            "internship",
            "linkedin",
            "bachelor",
            "master",
            "gpa",
            "resume",
            "curriculum vitae",
        )
        if token in blob
    )
    policy_hits = sum(
        1
        for token in (
            "membership",
            "refund",
            "cancellation",
            "accessibility",
            "sustainability",
            "compost",
            "event rentals",
            "volunteer program",
            "policy",
            "handbook",
        )
        if token in blob
    )
    if policy_hits >= 3 and resume_hits < 3:
        return False
    return resume_hits >= 2 or (
        _detect_person_name(lines) is not None and resume_hits >= 1
    )


def _is_section_heading(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in SECTION_HEADINGS and len(text) <= 40:
        return True
    return bool(
        re.match(
            r"^(Experience|Education|Skills|About|Summary|Profile|Projects|Research|"
            r"Leadership|Honors|Awards|Volunteer|Certifications|Organizations|Involvement)$",
            text.strip(),
            re.I,
        )
    )


def _looks_like_person_name(text: str) -> bool:
    if len(text) > 60 or DATE_RE.search(text) or EMPLOYMENT_RE.search(text):
        return False
    words = text.split()
    if not (2 <= len(words) <= 5):
        return False
    return all(word[:1].isupper() and word.replace(".", "").isalpha() for word in words)


def _detect_person_name(lines: Sequence[tuple]) -> Optional[str]:
    for text, _ in lines[:12]:
        if _looks_like_person_name(text):
            return text
    return None


def _looks_like_title(text: str) -> bool:
    if DATE_RE.search(text) or len(text) > 90:
        return False
    if text.endswith("."):
        return False
    words = text.split()
    return 1 <= len(words) <= 10 and text[:1].isupper()


def _looks_like_org(text: str) -> bool:
    if DATE_RE.search(text) or len(text) > 100:
        return False
    if SCHOOL_RE.search(text):
        return True
    words = text.split()
    return 1 <= len(words) <= 12 and text[:1].isupper() and not text.endswith(".")


def _looks_like_location(text: str) -> bool:
    if len(text) > 60:
        return False
    return bool(re.search(r",\s*[A-Z]{2}\b|, [A-Z][a-z]+$", text)) or bool(
        re.search(r"\b(Remote|Hybrid|On-site|United States|California)\b", text, re.I)
    )


def _infer_type(
    section: str, employment: Optional[str], title: Optional[str], organization: Optional[str]
) -> str:
    blob = " ".join(part for part in [section, employment, title, organization] if part).lower()
    if "intern" in blob:
        return "internship"
    if "research" in blob:
        return "research"
    if "education" in blob or SCHOOL_RE.search(blob or ""):
        return "education"
    if any(token in blob for token in ("leadership", "volunteer", "organization", "involvement")):
        return "leadership"
    if "project" in blob:
        return "project"
    return "experience"


def _stable_id(document_id: str, prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{prefix}:{value}".encode("utf-8")).hexdigest()
    return digest[:32]
