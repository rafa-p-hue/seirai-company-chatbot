"""Evidence-grounded answers that work with universal chunks and optional records."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import CitationSource, RetrievedChunk, SourceType
from app.retrieval.query_understanding import QueryUnderstanding

GREETING_ANSWER = "Hello! What would you like to know about this document?"
HELP_ANSWER = (
    "You can ask about people, organizations, dates, education, roles, "
    "policies, projects, and other information stated in the uploaded documents."
)

UNSUPPORTED_STATUS_RE = re.compile(
    r"(?i)\b("
    r"applying for|job applicant|applicant for|candidate for (?:a |an )?position|"
    r"seeking (?:a |an )?position|interviewing for"
    r")\b"
)

NAV_NOISE_PHRASE = re.compile(
    r"(?i)\b("
    r"i'?m looking for new opportunities|"
    r"enhance with ai|"
    r"profile language|"
    r"who your viewers also viewed|"
    r"show more|"
    r"\bpromoted\b"
    r")\b"
)
NAV_NOISE_LINE = re.compile(
    r"(?im)^(i'?m looking for.*|enhance with ai|profile language|promoted|"
    r"show more|who your viewers also viewed|connect|follow|english)$"
)


def compose_answer(
    *,
    understanding: QueryUnderstanding,
    evidence: Sequence[RetrievedChunk],
) -> Tuple[str, List[CitationSource]]:
    if understanding.query_type == "greeting":
        return GREETING_ANSWER, []
    if understanding.query_type == "help":
        return HELP_ANSWER, []
    if understanding.query_type == "unsupported":
        return FALLBACK_ANSWER, []
    if understanding.query_type in {"awards", "executive", "instrument"}:
        from app.retrieval.fact_types import (
            AWARDS_RE,
            EXECUTIVE_RE,
            INSTRUMENT_RE,
            filter_evidence_for_fact,
        )

        pattern = {
            "awards": AWARDS_RE,
            "executive": EXECUTIVE_RE,
            "instrument": INSTRUMENT_RE,
        }[understanding.query_type]
        typed = filter_evidence_for_fact(
            evidence, understanding.query_type, question=understanding.original_question
        )
        if not typed or not any(pattern.search(item.content or "") for item in typed):
            return FALLBACK_ANSWER, []
    if understanding.query_type == "instrument":
        from app.retrieval.fact_types import INSTRUMENT_RE

        if not any(INSTRUMENT_RE.search(item.content or "") for item in evidence):
            return FALLBACK_ANSWER, []

    cleaned = _scrub_evidence(evidence)
    if not cleaned and understanding.query_type not in {"greeting", "help"}:
        return FALLBACK_ANSWER, []

    # Explicit "not announced / TBD" — only when the question asks about that gap.
    unannounced = _unannounced_from_evidence(cleaned)
    if unannounced and understanding.query_type in {
        "location",
        "accessibility",
        "policy",
        "general",
    }:
        q = f"{understanding.original_question} {understanding.expanded_question}".lower()
        if re.search(r"\b(address|where|location|announce|site)\b", q) and not re.search(
            r"\b(opening period|opening date|when (?:is|does|will)|what year)\b", q
        ):
            return unannounced, _sources_for(cleaned[:2])

    if understanding.query_type == "identity":
        answer = _identity_from_evidence(cleaned, understanding.subject_name)
        answer = _sanitize_identity_answer(answer)
        if answer and (
            answer.lstrip().startswith("•")
            or re.match(r"(?i)^(lead analyst|field assistant)\b", answer)
        ):
            # Try again using only identity-friendly evidence.
            identity_ev = [
                item
                for item in cleaned
                if item.content_type not in {"list", "list_group"}
                and not (item.content or "").lstrip().startswith("•")
            ]
            answer = _sanitize_identity_answer(
                _identity_from_evidence(identity_ev, understanding.subject_name)
            )
    elif understanding.query_type == "summary":
        answer = _sanitize_identity_answer(
            _summary_from_evidence(cleaned, understanding)
        )
    elif understanding.query_type == "interest":
        answer = _interest_from_evidence(cleaned, understanding)
    elif understanding.query_type == "school":
        answer = _school_from_evidence(understanding, cleaned)
    elif understanding.query_type == "location":
        answer = _location_from_evidence(understanding, cleaned)
    elif understanding.query_type == "major":
        answer = _major_from_evidence(understanding, cleaned)
    elif understanding.query_type in {"gpa", "birthday", "favorite_food"}:
        answer = FALLBACK_ANSWER
        # Strict types: only answer if compose finds an exact matching label.
        from app.retrieval.fact_types import filter_evidence_for_fact

        typed = filter_evidence_for_fact(
            cleaned, understanding.query_type, question=understanding.original_question
        )
        if typed:
            answer = _fact_label_answer(understanding, typed)
    elif understanding.query_type == "education":
        answer = _education_from_evidence(understanding, cleaned)
        if not _validate_education_answer(answer, cleaned, understanding):
            return FALLBACK_ANSWER, []
    elif understanding.query_type == "leadership":
        q = f"{understanding.original_question} {understanding.expanded_question}".lower()
        if re.search(r"\b(found|establish|group|organization|club)\b", q):
            answer = _founded_from_evidence(understanding, cleaned)
        else:
            from app.retrieval.fact_types import LEADERSHIP_TITLE_RE

            strong = [
                item for item in cleaned if LEADERSHIP_TITLE_RE.search(item.content or "")
            ]
            if not strong:
                return FALLBACK_ANSWER, []
            # Yes/no title claims need the asked title (and org, if given) in evidence.
            title_claim = re.search(
                r"\b((?:vice[- ]?)?president|chair|director|ceo)\b", q
            )
            if title_claim and re.search(r"\b(is|was|are|were|does|did)\b", q):
                asked = title_claim.group(1)
                org_match = re.search(
                    r"\b(?:president|chair|director|ceo)\s+of\s+(.+?)\s*\??$",
                    q,
                )
                org = (org_match.group(1).strip() if org_match else "").lower()
                def _title_matches(content: str) -> bool:
                    text = content or ""
                    if asked in {"president"} and not re.search(
                        r"(?i)\b(?<!vice[- ])(?<!vice )president\b", text
                    ):
                        return False
                    if asked != "president" and not re.search(
                        rf"(?i)\b{re.escape(asked)}\b", text
                    ):
                        return False
                    if org:
                        # Require a substantial org token overlap in the same chunk.
                        org_tokens = [
                            t for t in re.findall(r"[a-z0-9]{4,}", org) if t not in {"the", "and"}
                        ]
                        if org_tokens and not any(t in text.lower() for t in org_tokens):
                            return False
                    return True

                if not any(_title_matches(item.content or "") for item in strong):
                    return FALLBACK_ANSWER, []
            answer = _leadership_titles_answer(understanding, strong)
    elif understanding.query_type in {"research", "experience", "internship"}:
        from app.retrieval.chunk_quality import is_strong_experience_evidence
        from app.retrieval.fact_types import RESEARCH_EVIDENCE_RE

        strong = [item for item in cleaned if is_strong_experience_evidence(item.content)]
        if understanding.query_type == "research":
            strong = [
                item
                for item in strong
                if RESEARCH_EVIDENCE_RE.search(item.content or "")
                or item.record_type == "research"
            ]
        if not strong:
            return FALLBACK_ANSWER, []
        answer = _list_from_evidence(
            strong,
            prefix=_list_prefix(understanding.query_type),
            keywords=_list_keywords(understanding.query_type),
            require_keywords=understanding.query_type == "research",
        )
        if understanding.query_type == "experience" and not _validate_work_answer(
            answer, strong
        ):
            return FALLBACK_ANSWER, []
    elif understanding.query_type in {"awards", "executive"}:
        answer = _specific_from_evidence(understanding, cleaned)
        from app.retrieval.fact_types import AWARDS_RE, EXECUTIVE_RE

        check = AWARDS_RE if understanding.query_type == "awards" else EXECUTIVE_RE
        if not check.search(answer or ""):
            return FALLBACK_ANSWER, []
    elif understanding.query_type in {
        "policy",
        "price",
        "quantity",
        "date",
        "accessibility",
        "product",
        "general",
        "skills",
    }:
        if understanding.query_type == "price":
            answer = _price_from_evidence(understanding, cleaned)
        elif understanding.query_type == "quantity":
            answer = _quantity_from_evidence(understanding, cleaned)
        elif understanding.query_type == "date":
            answer = _date_from_evidence(understanding, cleaned)
        elif understanding.query_type == "accessibility":
            answer = _accessibility_from_evidence(understanding, cleaned)
        else:
            answer = _specific_from_evidence(understanding, cleaned)
    else:
        answer = _specific_from_evidence(understanding, cleaned)

    if not answer or answer == FALLBACK_ANSWER or NAV_NOISE_PHRASE.search(answer):
        return FALLBACK_ANSWER, []
    if UNSUPPORTED_STATUS_RE.search(answer) and not any(
        UNSUPPORTED_STATUS_RE.search(item.content or "") for item in cleaned
    ):
        return FALLBACK_ANSWER, []

    answer = _finalize_answer(answer)
    if not answer or answer == FALLBACK_ANSWER:
        return FALLBACK_ANSWER, []
    # Never return a raw single chunk dump as the final answer.
    if _looks_like_raw_chunk_dump(answer, cleaned):
        return FALLBACK_ANSWER, []

    used = _select_used_evidence(answer, cleaned)
    return answer.strip(), _sources_for(used)


def _looks_like_raw_chunk_dump(answer: str, evidence: Sequence[RetrievedChunk]) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    if text.endswith(("…", "...")):
        return True
    # Contact / identity labeled dumps are never valid answers to other questions.
    if re.search(
        r"(?i)^(full\s*name|email|nominee'?s?\s*.*email(?:\s*address)?|phone|"
        r"program\s*status)\s*:",
        text,
    ):
        return True

    def _norm(value: str) -> str:
        return re.sub(r"[.\s]+$", "", _clean_answer_text(value or "")).lower()

    text_n = _norm(text)
    for item in evidence:
        label_l = (item.label or "").lower()
        is_contact = bool(
            re.search(r"\b(email|e-?mail|phone|full\s*name)\b", label_l)
            or re.search(r"(?i)^(full\s*name|email|phone)\s*:", item.content or "")
        )
        if not is_contact:
            continue
        chunk = _clean_answer_text(item.content or "")
        if chunk and text_n == _norm(chunk):
            return True
        if item.label and item.value and text_n == _norm(f"{item.label}: {item.value}"):
            return True
        if item.value and item.value.lower() in text.lower() and ":" in text and len(text) < 140:
            return True
    # Concatenated raw chunks with no generated framing words.
    if ";" in text and not re.search(
        r"(?i)\b(include|includes|listed|founded|attended|involved|associated|is|are)\b",
        text,
    ):
        return True
    return False


def _list_prefix(query_type: str) -> str:
    return {
        "research": "Research described in the document includes:",
        "leadership": "Listed roles and affiliations include:",
        "experience": "Listed roles include:",
        "internship": "Listed internship or temporary roles include:",
    }.get(query_type, "Relevant details include:")


def _list_keywords(query_type: str) -> Sequence[str]:
    return {
        "research": (
            "research",
            "lab",
            "laboratory",
            "urop",
            "analyst",
            "sampling",
            "fellow",
        ),
        "leadership": (
            "president",
            "vice",
            "chair",
            "founder",
            "director",
            "coordinator",
            "educator",
            "officer",
            "lead",
        ),
        "experience": (
            "intern",
            "engineer",
            "assistant",
            "advisor",
            "role",
            "experience",
            "founder",
            "manager",
            "officer",
            "analyst",
            "technician",
            "operator",
            "work",
        ),
        "internship": ("intern", "internship"),
    }.get(query_type, ())


def _location_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.fact_types import LOCATION_SIGNAL_RE
    from app.retrieval.numeric_facts import content_states_unannounced

    name = understanding.subject_name or "The person"
    unannounced = _unannounced_from_evidence(evidence)
    if unannounced:
        return unannounced
    for item in evidence:
        text = _clean_answer_text(item.content or "")
        if content_states_unannounced(text):
            continue
        if not LOCATION_SIGNAL_RE.search(text):
            continue
        # Prefer explicit labeled location values.
        if item.label and item.value and re.search(
            r"(?i)\b(city|state|country|location|address|hometown)\b", item.label
        ):
            return f"{name}'s location is {item.value}."
        match = re.search(
            r"(?i)(?:located in|based in|lives? in|resides? in)\s+([^.;\n]+)",
            text,
        )
        if match:
            place = _trim_complete_phrase(match.group(1).strip(" ,"), max_len=80)
            if place:
                return f"{name} is located in {place}."
        match = re.search(
            r"(?i)\b(?:city|state|country|location|address|hometown)\s*:\s*([^\n;]+)",
            text,
        )
        if match:
            place = _trim_complete_phrase(match.group(1).strip(" ,"), max_len=80)
            if place:
                return f"{name}'s location is {place}."
    return FALLBACK_ANSWER


def _scrub_evidence(evidence: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
    from app.generation.text_scrub import looks_like_internal_metadata, scrub_internal_metadata

    cleaned: List[RetrievedChunk] = []
    for item in evidence:
        if not item.content:
            continue
        # Scrub structured markers before whitespace collapsing.
        content = scrub_internal_metadata(item.content)
        if looks_like_internal_metadata(content) or len(content.strip()) < 8:
            continue
        # Preserve list structure (bullets / newlines) for policy and activity answers.
        if item.content_type in {"list", "list_group"} or "\n" in content or "•" in content:
            scrubbed = re.sub(r"[ \t]{2,}", " ", content)
            scrubbed = re.sub(r"\n{3,}", "\n\n", scrubbed).strip()
        else:
            content = _clean_answer_text(content)
            lines = [
                line
                for line in content.splitlines()
                if line.strip() and not NAV_NOISE_LINE.match(line.strip())
            ]
            scrubbed = NAV_NOISE_PHRASE.sub(" ", "\n".join(lines) if lines else content)
            scrubbed = re.sub(r"[ \t]{2,}", " ", scrubbed)
            scrubbed = re.sub(r"\n{3,}", "\n\n", scrubbed).strip()
        if len(scrubbed) < 8:
            continue
        updates = {"content": scrubbed}
        if item.value:
            updates["value"] = _clean_answer_text(item.value)
        if scrubbed != item.content or updates.get("value") != item.value:
            item = item.model_copy(update=updates)
        cleaned.append(item)
    return cleaned


def _identity_from_evidence(
    evidence: Sequence[RetrievedChunk], subject_name: Optional[str]
) -> str:
    from app.retrieval.label_match import extract_labeled_value

    name = (
        extract_labeled_value(evidence, ["name"])
        or _label_value(
            evidence,
            r"full\s*name|nominee'?s?\s*(?:full\s*)?name|lead\s*author|(?<![A-Za-z])name(?![A-Za-z])",
        )
        or subject_name
    )
    for item in evidence:
        if item.content_type == "key_value" and item.person_name:
            name = item.person_name
            break
    if not name or name.lower() in {
        "the main person",
        "the main subject described in the document",
    }:
        for item in evidence:
            if item.person_name and len(item.person_name.split()) >= 2:
                name = item.person_name
                break
    email = extract_labeled_value(evidence, ["email"]) or _label_value(evidence, r"email")
    major = extract_labeled_value(evidence, ["major"]) or _label_value(evidence, r"major")
    minor = extract_labeled_value(evidence, ["minor"]) or _label_value(evidence, r"minor")
    school = _organization_from_education(evidence)
    status = _label_value(evidence, r"program\s*status|nomination\s*status|\bstatus\b")
    title = None
    for item in evidence:
        if item.record_type in {"profile", "education"} and item.title and item.title != name:
            if re.search(r"(?i)\b(interest|experience|education|skills|section)\b", item.title):
                continue
            title = item.title
            break
    parts: List[str] = []
    if name and name.lower() not in {
        "the main person",
        "the main subject described in the document",
    }:
        parts.append(name)
    details = []
    if status and not UNSUPPORTED_STATUS_RE.search(status):
        details.append(status)
    if title and title.lower() not in (name or "").lower():
        details.append(title)
    if major:
        details.append(f"field of study {major}")
    if minor:
        details.append(f"secondary field {minor}")
    if school:
        details.append(f"associated with {school}")
    if email:
        details.append(f"email {email}")
    if details:
        parts.append(", ".join(details))
    for item in evidence:
        if item.content_type in {"list", "list_group"}:
            continue
        text = re.sub(r"\s+", " ", item.content).strip()
        if len(text) < 40:
            continue
        if text.lstrip().startswith("•"):
            continue
        if re.match(r"(?i)^(full\s*name|email|minor|major|nominee|record type|lead analyst)\b", text):
            continue
        if re.search(
            r"(?i)\b(intern|advisor|assistant|founder|research|leader|manager|member)\b",
            text,
        ):
            snippet = text[:160]
            if snippet not in parts and (name or "") not in snippet[:40]:
                parts.append(snippet)
            break
    if not parts:
        return FALLBACK_ANSWER
    if name and len(parts) > 1 and name.lower() not in {
        "the main person",
        "the main subject described in the document",
    }:
        return f"{parts[0]} — {'; '.join(parts[1:])}."
    return ". ".join(parts) + ("." if not parts[-1].endswith(".") else "")


def _school_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.institution import extract_best_institution

    name = understanding.subject_name or "The person"
    school = extract_best_institution(evidence)
    if not school:
        return FALLBACK_ANSWER
    # Never answer school with a major-only value.
    if re.search(r"(?i)\b(major|field of study)\b", school) and not re.search(
        r"(?i)\b(university|college|institute|school)\b", school
    ) and not re.fullmatch(r"[A-Z]{2,6}", school.strip()):
        return FALLBACK_ANSWER
    return f"{name} attended {school}."


def _founded_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    """One direct sentence for founder/establish questions."""
    name = understanding.subject_name or "The person"
    for item in evidence:
        raw = item.content or ""
        # Search bullet/line fragments — scrubbing may flatten newlines.
        candidates = re.split(r"\n+|•", raw) if re.search(r"[\n•]", raw) else [raw]
        # If flattened to one line, also split on role-like separators.
        if len(candidates) == 1 and "Founder" in raw:
            candidates = re.split(r"(?=\b(?:Founder|Co-Founder|President|Chair)\b)", raw)
        for line in candidates:
            text = _clean_answer_text(line)
            if not re.search(r"(?i)\b(founder|co-?founder|founded|established)\b", text):
                continue
            group = None
            patterns = (
                r"(?i)\bfounder(?:,|\s+of)?\s+([^–—\-\n:]+?)(?:\s*[-–—:]\s*|\s*$)",
                r"(?i)\bfounded\s+(?:the\s+)?([^.,;\n]+)",
                r"(?i)\bestablished\s+(?:the\s+)?([^.,;\n]+)",
                r"(?i)\bco-?founder(?:,|\s+of)?\s+([^–—\-\n:]+?)(?:\s*[-–—:]\s*|\s*$)",
            )
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    group = match.group(1).strip(" ,;.")
                    break
            if not group:
                left = re.split(r"\s[-–—]\s", text, maxsplit=1)[0]
                left = re.sub(r"(?i)^(?:co-)?founder(?:,|\s+of)?\s*", "", left).strip(" •")
                if left and len(left) > 2:
                    group = left
            if group and not re.search(r"(?i)^(founder|established|record type)\b", group):
                group = _trim_complete_phrase(group, max_len=120)
                return f"{name} founded {group}."
    return FALLBACK_ANSWER


def _clean_answer_text(text: str) -> str:
    from app.retrieval.institution import strip_section_markers

    cleaned = strip_section_markers(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _trim_complete_phrase(text: str, *, max_len: int = 160) -> str:
    """Trim without cutting mid-word or mid-sentence."""
    text = _clean_answer_text(text)
    if len(text) <= max_len:
        return text.rstrip(" ,;:")
    cut = text[:max_len]
    # Prefer sentence boundary.
    for sep in (". ", "; ", ", ", " "):
        idx = cut.rfind(sep)
        if idx >= max(40, max_len // 3):
            cut = cut[: idx + (1 if sep == ". " else 0)]
            break
    else:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:…")


def _finalize_answer(answer: str) -> str:
    """Sanitize user-facing answers: no markers, no mid-word truncations."""
    if not answer or answer == FALLBACK_ANSWER:
        return answer
    from app.generation.text_scrub import looks_like_internal_metadata, scrub_internal_metadata

    text = scrub_internal_metadata(_clean_answer_text(answer))
    if looks_like_internal_metadata(text) or re.search(
        r"(?i)\b(record\s*type|profile\s*description)\s*:", text
    ):
        return FALLBACK_ANSWER
    # Drop trailing incomplete ellipsis fragments that look like raw dumps.
    if text.endswith("…") or text.endswith("..."):
        text = text.rstrip(".…")
        text = _trim_complete_phrase(text, max_len=len(text))
        if text and not text.endswith((".", "!", "?")):
            text = text + "."
    # Reject answers that are mostly raw labeled dumps with no sentence structure
    # when they end mid-token (no space after last 3+ alnum run cut by ellipsis).
    if re.search(r"[A-Za-z0-9]{3,}$", text) and len(text) > 200 and ";" in text:
        # Prefer first complete sentence-like clause.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if parts:
            text = parts[0]
            if not text.endswith((".", "!", "?")):
                text += "."
    return text


def _unannounced_from_evidence(evidence: Sequence[RetrievedChunk]) -> Optional[str]:
    from app.retrieval.numeric_facts import UNANNOUNCED_RE

    for item in evidence:
        text = _clean_answer_text(item.content or "")
        match = UNANNOUNCED_RE.search(text)
        if not match:
            continue
        # Prefer a complete sentence containing the unannounced statement.
        for part in re.split(r"(?<=[.!?])\s+", text):
            if UNANNOUNCED_RE.search(part):
                sentence = part.strip()
                if not sentence.endswith((".", "!", "?")):
                    sentence += "."
                return sentence
        return f"{match.group(0).rstrip('.')}."
    return None


def _major_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.label_match import extract_labeled_value

    name = understanding.subject_name or "The person"
    major = (
        extract_labeled_value(evidence, ["major"])
        or _label_value(evidence, r"major|field\s*of\s*study|academic\s*program")
        or _degree_from_evidence(evidence)
    )
    if not major:
        return FALLBACK_ANSWER
    return f"{name}'s field of study is {major}."


def _fact_label_answer(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.label_match import extract_labeled_value

    qtype = understanding.query_type
    name = understanding.subject_name or "The person"
    if qtype == "gpa":
        value = extract_labeled_value(evidence, ["gpa"]) or _label_value(
            evidence, r"gpa|grade[- ]?point\s*average"
        )
        if not value:
            return FALLBACK_ANSWER
        return f"{name}'s GPA is {value}."
    if qtype == "birthday":
        value = extract_labeled_value(evidence, ["birthday"]) or _label_value(
            evidence, r"birthday|date\s*of\s*birth|birth\s*date|dob"
        )
        if not value:
            return FALLBACK_ANSWER
        return f"{name}'s date of birth is {value}."
    if qtype == "favorite_food":
        value = extract_labeled_value(evidence, ["food"]) or _label_value(
            evidence, r"favorite\s*food|favourite\s*food|food\s*preference|\bfood\b"
        )
        if not value:
            return FALLBACK_ANSWER
        return f"{name}'s favorite food is {value}."
    return FALLBACK_ANSWER


def _education_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.label_match import extract_labeled_value

    q = understanding.expanded_question.lower() + " " + understanding.original_question.lower()
    name = understanding.subject_name or "The document"
    wants_major = bool(
        re.search(r"(?i)\b(major|field of study|academic program|degree)\b", q)
    )
    wants_minor = bool(re.search(r"(?i)\b(minor|secondary)\b", q))
    if wants_major and wants_minor:
        major = (
            extract_labeled_value(evidence, ["major"])
            or _label_value(evidence, r"major|field\s*of\s*study|academic\s*program")
        )
        minor = extract_labeled_value(evidence, ["minor"]) or _label_value(evidence, r"minor")
        parts = []
        if major:
            parts.append(f"major is {major}")
        if minor:
            parts.append(f"minor is {minor}")
        if parts:
            return f"{name}'s " + " and ".join(parts) + "."
        return FALLBACK_ANSWER
    if wants_minor:
        minor = extract_labeled_value(evidence, ["minor"]) or _label_value(evidence, r"minor")
        if minor:
            return f"{name}'s secondary field of study is {minor}."
        return FALLBACK_ANSWER
    # Route school vs major explicitly when education intent is ambiguous.
    wants_school = bool(
        re.search(r"(?i)\b(school|university|college|institution|campus|attend)\b", q)
    )
    wants_major = bool(
        re.search(r"(?i)\b(major|field of study|academic program|degree|stud(?:y|ies|ying))\b", q)
    )
    if wants_school and not wants_major:
        return _school_from_evidence(understanding, evidence)
    if wants_major:
        return _major_from_evidence(understanding, evidence)
    school = _school_from_evidence(understanding, evidence)
    if school != FALLBACK_ANSWER:
        return school
    return _major_from_evidence(understanding, evidence)


def _validate_education_answer(
    answer: str, evidence: Sequence[RetrievedChunk], understanding: QueryUnderstanding
) -> bool:
    if not answer or answer == FALLBACK_ANSWER:
        return False
    from app.retrieval.label_match import extract_labeled_value

    q = f"{understanding.original_question} {understanding.expanded_question}".lower()
    if "minor" in q:
        value = extract_labeled_value(evidence, ["minor"]) or _label_value(evidence, r"minor")
        return bool(value and value.lower() in answer.lower())
    value = (
        extract_labeled_value(evidence, ["major"])
        or _label_value(evidence, r"major|field\s*of\s*study|academic\s*program")
        or _degree_from_evidence(evidence)
    )
    if value and value.lower() in answer.lower():
        return True
    for item in evidence:
        if item.record_type == "education" and item.title and item.title.lower() in answer.lower():
            return True
    # School-only education answers are OK when not asking for major/study.
    if not re.search(r"(?i)\b(major|stud(?:y|ies)|degree|field of study)\b", q):
        return True
    return False


def _validate_work_answer(answer: str, evidence: Sequence[RetrievedChunk]) -> bool:
    if not answer or answer == FALLBACK_ANSWER:
        return False
    from app.retrieval.chunk_quality import ROLE_SIGNAL_RE

    joined = " ".join(item.content for item in evidence)
    return bool(ROLE_SIGNAL_RE.search(answer) or ROLE_SIGNAL_RE.search(joined))


def _sanitize_identity_answer(answer: str) -> str:
    if not answer or answer == FALLBACK_ANSWER:
        return answer
    # Strip unsupported application/status inferences if somehow present.
    cleaned = UNSUPPORTED_STATUS_RE.sub("", answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.-")
    if not cleaned:
        return FALLBACK_ANSWER
    return cleaned


def _organization_from_education(evidence: Sequence[RetrievedChunk]) -> Optional[str]:
    for item in evidence:
        if item.organization and (
            item.record_type == "education"
            or re.search(r"university|college|school|institute", item.organization, re.I)
        ):
            return item.organization.strip()
        match = re.search(r"(?im)^Organization:\s*(.+)$", item.content)
        if match:
            value = match.group(1).strip()
            if re.search(r"university|college|school|institute", value, re.I):
                return value
    return None


def _degree_from_evidence(evidence: Sequence[RetrievedChunk]) -> Optional[str]:
    for item in evidence:
        if item.record_type == "education" and item.title:
            return item.title.strip()
        match = re.search(r"(?im)^Title:\s*(.+)$", item.content)
        if match and (item.record_type or "universal") in {
            "education",
            "universal",
            "profile",
            None,
        }:
            title = match.group(1).strip()
            if re.search(r"\b(B\.?S\.?|B\.?A\.?|M\.?S\.?|Ph\.?D\.?|major|degree)\b", title, re.I):
                return title
    return None


def _list_from_evidence(
    evidence: Sequence[RetrievedChunk],
    *,
    prefix: str,
    keywords: Sequence[str],
    require_keywords: bool = False,
) -> str:
    items: List[str] = []
    for chunk in evidence:
        text = _clean_answer_text(chunk.content or "")
        if not text:
            continue
        lower = text.lower()
        if keywords and not any(keyword in lower for keyword in keywords):
            continue
        snippet = text
        if re.search(r"\s[-–—]\s", text):
            left, right = re.split(r"\s[-–—]\s", text, maxsplit=1)
            snippet = left.strip()
            if right.strip():
                snippet = f"{snippet} – {_trim_complete_phrase(right, max_len=100)}"
        else:
            snippet = _trim_complete_phrase(text, max_len=160)
        if snippet and snippet not in items:
            items.append(snippet)
        if len(items) >= 4:
            break
    # For strict research lists, never fall back to unrelated leadership/activity rows.
    if not items and not require_keywords:
        for chunk in evidence[:3]:
            text = _clean_answer_text(chunk.content or "")
            if not text or re.search(r"(?i)full\s*name|email address", text):
                continue
            title = re.split(r"\s[-–—]\s", text, maxsplit=1)[0].strip()
            title = _trim_complete_phrase(title, max_len=120)
            if title and title not in items:
                items.append(title)
    if not items:
        return FALLBACK_ANSWER
    return prefix + " " + "; ".join(items) + "."


def _leadership_titles_answer(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    name = understanding.subject_name or "The person"
    titles: List[str] = []
    for item in evidence:
        text = _clean_answer_text(item.content or "")
        snippet = re.split(r"\s[-–—]\s", text, maxsplit=1)[0].strip()
        snippet = _trim_complete_phrase(snippet, max_len=100)
        if snippet and snippet not in titles:
            titles.append(snippet)
        if len(titles) >= 4:
            break
    if not titles:
        return FALLBACK_ANSWER
    return f"Leadership roles listed for {name} include: " + "; ".join(titles) + "."


def _price_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.numeric_facts import CURRENCY_RE

    for item in evidence:
        text = _clean_answer_text(item.content or "")
        match = CURRENCY_RE.search(text)
        if not match:
            continue
        amount = match.group(0)
        section = item.section_title or "the document"
        # Prefer a complete sentence/list item containing the amount.
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            if amount in part:
                snippet = _trim_complete_phrase(part.strip(" •"), max_len=180)
                if snippet:
                    return snippet if snippet.endswith((".", "!", "?")) else f"{snippet}."
        return f"According to {section}, the amount is {amount}."
    return FALLBACK_ANSWER


def _quantity_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.numeric_facts import QUANTITY_RE, PERCENT_RE

    for item in evidence:
        text = _clean_answer_text(item.content or "")
        match = QUANTITY_RE.search(text) or PERCENT_RE.search(text)
        if not match:
            continue
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            if match.group(0) in part:
                snippet = _trim_complete_phrase(part.strip(" •"), max_len=180)
                if snippet:
                    return snippet if snippet.endswith((".", "!", "?")) else f"{snippet}."
        return f"The document states {match.group(0)}."
    return FALLBACK_ANSWER


def _date_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.numeric_facts import content_has_date_or_period

    q = f"{understanding.original_question} {understanding.expanded_question}".lower()
    for item in evidence:
        text = _clean_answer_text(item.content or "")
        if not content_has_date_or_period(text):
            continue
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            if content_has_date_or_period(part):
                snippet = _trim_complete_phrase(part.strip(" •"), max_len=180)
                if snippet:
                    return snippet if snippet.endswith((".", "!", "?")) else f"{snippet}."
    # Only fall back to unannounced when the question itself asks about an unknown date/address.
    if re.search(r"\b(address|announce|tbd|tba)\b", q):
        unannounced = _unannounced_from_evidence(evidence)
        if unannounced:
            return unannounced
    return FALLBACK_ANSWER


def _accessibility_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.numeric_facts import content_has_accessibility

    unannounced = _unannounced_from_evidence(evidence)
    q = understanding.original_question.lower()
    if unannounced and re.search(r"\b(address|location|where)\b", q):
        return unannounced
    for item in evidence:
        text = _clean_answer_text(item.content or "")
        if not content_has_accessibility(text):
            continue
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            if content_has_accessibility(part):
                snippet = _trim_complete_phrase(part.strip(" •"), max_len=200)
                if snippet:
                    return snippet if snippet.endswith((".", "!", "?")) else f"{snippet}."
        return _trim_complete_phrase(text, max_len=200)
    return FALLBACK_ANSWER


def _interest_from_evidence(
    evidence: Sequence[RetrievedChunk], understanding: QueryUnderstanding
) -> str:
    pattern = re.compile(
        r"(?i)\b(music|musician|ensemble|band|performance|choir|orchestra)\b"
    )
    name = understanding.subject_name or "The person"
    for item in evidence:
        text = item.content or ""
        if not pattern.search(text):
            continue
        # Split on newlines or bullets — scrubbing may flatten newlines to spaces.
        parts = re.split(r"\n+|•|;", text)
        for part in parts:
            part = _clean_answer_text(part)
            if not pattern.search(part):
                continue
            title = re.split(r"\s[-–—]\s", part, maxsplit=1)[0].strip(" •[]")
            title = _trim_complete_phrase(title, max_len=100)
            if title:
                return f"{name} is involved in music through {title}."
        # Fallback: window around the first music keyword.
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 40)
            window = text[start : match.end() + 60]
            # Prefer text after the previous bullet/dash boundary.
            window = re.split(r"[•\n]", window)[-1]
            title = re.split(r"\s[-–—]\s", _clean_answer_text(window), maxsplit=1)[0]
            title = _trim_complete_phrase(title.strip(" •[]"), max_len=100)
            if title and pattern.search(title) or title:
                # Prefer titles that themselves look like activity names.
                if re.search(r"(?i)\b(ensemble|band|choir|orchestra|music)\b", title):
                    return f"{name} is involved in music through {title}."
        cleaned = _clean_answer_text(text)
        # Last resort: extract "Campus Concert Ensemble"-style phrase near keyword.
        near = pattern.search(cleaned)
        if near:
            left = cleaned[: near.start()]
            # Take the last capitalized phrase before the keyword.
            caps = re.findall(r"(?:[A-Z][A-Za-z0-9&'/-]+\s+){1,6}[A-Z][A-Za-z0-9&'/-]+", left)
            if caps:
                return f"{name} is involved in music through {caps[-1].strip()}."
    return FALLBACK_ANSWER


def _specific_from_evidence(
    understanding: QueryUnderstanding, evidence: Sequence[RetrievedChunk]
) -> str:
    from app.retrieval.label_match import extract_labeled_value

    question_l = f"{understanding.expanded_question} {understanding.original_question}".lower()
    # Prefer exact labeled fields for common intents.
    if understanding.query_type == "product" or "product" in question_l:
        value = extract_labeled_value(evidence, ["product"])
        if value:
            return f"The product is {value}."
    if understanding.query_type == "policy" or "policy" in question_l:
        value = extract_labeled_value(evidence, ["policy"]) or _label_value(
            evidence, r"policy|leave\s*policy|remote\s*work\s*policy"
        )
        if value:
            subject = understanding.subject_name or "The document"
            return f"{subject}: policy is {value}."
    for pattern in ("major", "minor", "email", "full name", "school", "policy", "feature", "product", "model"):
        if pattern in question_l:
            value = extract_labeled_value(
                evidence,
                ["product"]
                if pattern in {"product", "model"}
                else ["major"]
                if pattern == "major"
                else ["minor"]
                if pattern == "minor"
                else ["email"]
                if pattern == "email"
                else ["name"]
                if pattern == "full name"
                else ["policy"]
                if pattern == "policy"
                else [],
            ) or _label_value(evidence, pattern.replace(" ", r"\s*"))
            if value:
                subject = understanding.subject_name or "The document"
                return f"{subject}: {pattern} is {value}."

    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", question_l)
        if token
        not in {
            "what",
            "does",
            "about",
            "this",
            "that",
            "with",
            "from",
            "have",
            "tell",
            "person",
            "document",
            "main",
            "subject",
            "receive",
            "company",
        }
    }
    asks_contact = bool(re.search(r"\b(email|phone|full\s*name)\b", question_l))
    # Distinctive tokens should win over weak overlaps (e.g. "event" alone).
    priority_tokens = {
        token
        for token in query_tokens
        if token
        in {
            "alcohol",
            "compost",
            "prohibited",
            "refund",
            "cancellation",
            "pets",
            "volunteer",
            "orientation",
            "wheelchair",
            "accessible",
            "membership",
            "donation",
            "pounds",
        }
    }
    scored: List[Tuple[int, str]] = []
    for item in evidence:
        # Prefer short key-value hits when they match the query family.
        if (item.content_type == "key_value" or item.label) and item.value:
            label_l = (item.label or "").lower()
            if re.search(r"\b(email|phone|full\s*name|name)\b", label_l) and not asks_contact:
                continue
            if any(token in label_l for token in query_tokens) or any(
                token in (item.content or "").lower() for token in query_tokens
            ):
                # Never return bare Label: value for claim questions.
                if understanding.query_type in {
                    "awards",
                    "executive",
                    "instrument",
                    "leadership",
                    "research",
                    "location",
                    "policy",
                }:
                    continue
                return f"{item.label}: {item.value}." if item.label else f"{item.value}."
        for part in re.split(r"(?<=[.!?])\s+|\n+", item.content or ""):
            text = re.sub(r"\s+", " ", part).strip(" -•\t[]")
            if len(text) < 8:
                continue
            if re.match(r"(?i)^(full\s*name|email|minor|provide a list)\b", text):
                continue
            lower = text.lower()
            overlap = sum(1 for token in query_tokens if token in lower)
            priority = sum(1 for token in priority_tokens if token in lower)
            if priority or overlap >= 1:
                scored.append((priority * 10 + overlap, text[:260]))

    scored.sort(key=lambda item: item[0], reverse=True)
    sentences = []
    seen = set()
    for _score, text in scored:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(text)
        if len(sentences) >= (3 if understanding.query_type == "policy" else 2):
            break

    # For prohibition / list questions, include bullet lines from matching chunks.
    if re.search(r"\b(prohibited|forbidden|not allowed|cannot|may not)\b", question_l):
        for item in evidence:
            raw = item.content or ""
            if "•" not in raw and "\n" not in raw:
                continue
            if not any(
                token in raw.lower()
                for token in ("prohibit", "compost", "alcohol", "not allowed", "forbidden")
            ):
                continue
            for part in re.split(r"\n+|•", raw):
                text = re.sub(r"\s+", " ", part).strip(" -•\t[]")
                if len(text) < 4:
                    continue
                if re.search(
                    r"(?i)^(the following|accepted|include|prohibited in)\b", text
                ):
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                sentences.append(text)
                if len(sentences) >= 5:
                    break
            if len(sentences) >= 5:
                break

    if not sentences:
        # Soft fallback for general topical questions: use substantive non-contact text.
        # Never soft-fallback for strict policy/price/quantity/date/accessibility intents.
        if understanding.query_type in {
            "policy",
            "price",
            "quantity",
            "date",
            "accessibility",
            "awards",
            "executive",
            "instrument",
        }:
            return FALLBACK_ANSWER
        for item in evidence:
            text = _clean_answer_text(item.content or "")
            if len(text) < 40:
                continue
            if re.match(r"(?i)^(full\s*name|email|minor|major|nominee|phone)\b", text):
                continue
            if (item.content_type == "key_value" or item.label) and re.search(
                r"(?i)\b(email|phone|full\s*name)\b", item.label or ""
            ):
                continue
            # Prefer list groups when asking about prohibitions.
            if "prohibited" in question_l or "compost" in question_l:
                if "•" in (item.content or "") or item.content_type == "list":
                    sentences.append(text[:320])
                    break
            sentences.append(text[:220])
            break

    if not sentences:
        return FALLBACK_ANSWER
    return " ".join(sentences)


def _summary_from_evidence(
    evidence: Sequence[RetrievedChunk], understanding: QueryUnderstanding
) -> str:
    from app.retrieval.chunk_quality import is_weak_broad_evidence
    from app.retrieval.diversity import infer_topic
    from app.retrieval.fact_types import LEADERSHIP_TITLE_RE
    from app.retrieval.label_match import extract_labeled_value

    name = (
        extract_labeled_value(evidence, ["name"])
        or understanding.subject_name
        or "The main subject"
    )
    major = extract_labeled_value(evidence, ["major"]) or _label_value(evidence, r"major")
    school = (
        extract_labeled_value(evidence, ["organization"])
        or _organization_from_education(evidence)
    )
    sentences: List[str] = []
    if name and name.lower() not in {
        "the main subject",
        "the main person",
        "the main subject described in the document",
    }:
        sentences.append(
            f"This document describes {name}."
        )
    else:
        sentences.append("This document describes the main subject of the uploaded file.")

    detail_bits: List[str] = []
    if major:
        detail_bits.append(f"field of study {major}")
    if school:
        detail_bits.append(f"associated with {school}")
    if detail_bits:
        sentences.append(f"Key education details include {', '.join(detail_bits)}.")

    roles: List[str] = []
    research: List[str] = []
    for item in evidence:
        if is_weak_broad_evidence(item.content):
            continue
        text = re.sub(r"\s+", " ", item.content).strip()
        if len(text) < 30 or re.match(r"(?i)^(full\s*name|email|minor|major)\s*:", text):
            continue
        topic = infer_topic(text, {"record_type": item.record_type})
        # Prefer complete list entries / sentences, never truncated mid-word dumps.
        snippet = text
        if " - " in text or " – " in text or " — " in text:
            snippet = re.split(r"\s[-–—]\s", text, maxsplit=1)[0].strip()
        snippet = snippet[:120].rstrip(" ,;:")
        if topic == "research" and snippet not in research:
            research.append(snippet)
        elif topic in {"experience", "affiliation"} or LEADERSHIP_TITLE_RE.search(text):
            if snippet not in roles:
                roles.append(snippet)
        if len(roles) >= 3 and len(research) >= 1:
            break

    if roles:
        sentences.append(
            "Experience and activities include " + "; ".join(roles[:3]) + "."
        )
    if research:
        sentences.append("Research described includes " + "; ".join(research[:2]) + ".")
    if len(sentences) < 2:
        identity = _identity_from_evidence(evidence, understanding.subject_name)
        if identity != FALLBACK_ANSWER:
            return _sanitize_identity_answer(identity)
        return FALLBACK_ANSWER
    return " ".join(sentences)




def _label_value(evidence: Sequence[RetrievedChunk], label_pattern: str) -> Optional[str]:
    pattern = re.compile(
        rf"(?im)(?:^|\s|\])([^\n:\]]{{0,60}}?\b(?:{label_pattern})\b[^\n:]{{0,40}}):\s*"
        rf"(.+?)(?=\s+[A-Z][A-Za-z][^:\n]{{0,40}}:|$)"
    )
    for item in evidence:
        match = pattern.search(item.content)
        if match:
            value = match.group(2).strip(" \n\t|-[]")
            if value:
                return value
    return None


def _select_used_evidence(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> List[RetrievedChunk]:
    used: List[RetrievedChunk] = []
    pages_used: set = set()
    answer_l = answer.lower()
    for item in evidence:
        content_l = item.content.lower()
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]{4,}", content_l)
            if token not in {"nominee", "with", "that", "this", "from", "have", "record", "type"}
        ]
        if any(token in answer_l for token in tokens[:16]):
            used.append(item)
            if item.page_number is not None:
                pages_used.add(item.page_number)
        if len(used) >= 4:
            break
    if len(pages_used) < 2:
        for item in evidence:
            if item in used or item.page_number is None or item.page_number in pages_used:
                continue
            if any(token in answer_l for token in re.findall(r"[a-z0-9]{5,}", item.content.lower())[:20]):
                used.append(item)
                break
    return used or list(evidence[:2])


def _sources_for(evidence: Sequence[RetrievedChunk]) -> List[CitationSource]:
    sources: List[CitationSource] = []
    seen_pages = set()
    for item in evidence:
        key = (item.document_name, item.page_number)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        sources.append(
            CitationSource(
                number=len(sources) + 1,
                document_name=item.document_name,
                page_number=item.page_number,
                source_url=item.source_url,
                source_type=SourceType.website if item.source_url else SourceType.pdf,
            )
        )
        if len(sources) >= 4:
            break
    return sources
