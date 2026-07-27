"""Fact-type detection and strict evidence validation (document-agnostic)."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from app.models.api import RetrievedChunk
from app.retrieval.label_match import extract_labeled_value, label_family


INSTITUTION_RE = re.compile(
    r"(?i)\b("
    r"university|college|school|institute|institution|academy|campus|"
    r"polytechnic|conservatory"
    r")\b"
)
MAJOR_VALUE_RE = re.compile(
    r"(?i)\b(major|field of study|academic program|degree program)\b\s*:|"
    r"\b(B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?)\b"
)
GPA_RE = re.compile(
    r"(?i)\b(gpa|grade[- ]?point\s+average|cumulative\s+gpa|overall\s+gpa)\b|"
    r"\b\d\.\d{1,2}\s*/\s*[45](?:\.0)?\b|"
    r"\bgpa\s*[:=]\s*\d"
)
BIRTHDAY_RE = re.compile(
    r"(?i)\b(birthday|date of birth|birth\s*date|dob|born on)\b"
)
FOOD_RE = re.compile(
    r"(?i)\b(favorite\s+food|favourite\s+food|food\s+preference|preferred\s+food|"
    r"likes to eat|favorite\s+meal)\b"
)
INSTRUMENT_RE = re.compile(
    r"(?i)\b(instrument|guitar|piano|violin|drums|flute|trumpet|cello|"
    r"clarinet|saxophone|bass|ukulele|harp)\b"
)
LEADERSHIP_TITLE_RE = re.compile(
    r"(?i)\b("
    r"president|vice[- ]?president|chair(?:person)?|founder|co-?founder|"
    r"director|coordinator|educator|officer|secretary|treasurer|lead(?:er)?"
    r")\b"
)
AWARDS_RE = re.compile(
    r"(?i)\b("
    r"awards?|honors?|honours?|recognitions?|scholarships?|"
    r"grammy|oscar|emmy|nobel|pulitzer|fellowship award"
    r")\b"
)
EXECUTIVE_RE = re.compile(
    r"(?i)\b(ceo|chief executive|cfo|cto|coo|founder and ceo|company .+ ceo)\b"
)
LOCATION_SIGNAL_RE = re.compile(
    r"\b("
    r"located in|based in|headquartered|address|hometown|"
    r"lives? in|resides?(?:\s+in)?|city|state|country|region"
    r")\b|"
    r"\b(?:city|state|country|location|address|hometown)\s*:|"
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2}\b",
    re.I,
)
RESEARCH_EVIDENCE_RE = re.compile(
    r"(?i)("
    r"\bresearch\b|"
    r"\burop\b|"
    r"\b(?:lab|laboratory)\b|"
    r"\bfield\s+(?:research|analyst|assistant)\b|"
    r"\b(?:lead\s+)?analyst\b|"
    r"\bsampling\b"
    r")"
)
UNRELATED_IDENTITY_RE = re.compile(
    r"(?i)^("
    r"full\s*name|email|nominee'?s?\s*(?:uci\s*)?email(?:\s*address)?|"
    r"phone|minor|major|program\s*status"
    r")\s*:"
)
RAW_LABEL_ANSWER_RE = re.compile(
    r"(?i)^[A-Za-z0-9][^:\n]{0,80}:\s*\S+[.!]?\s*$"
)


FACT_TYPE_PATTERNS = {
    "gpa": GPA_RE,
    "birthday": BIRTHDAY_RE,
    "favorite_food": FOOD_RE,
    "instrument": INSTRUMENT_RE,
    "school": INSTITUTION_RE,
    "leadership": LEADERSHIP_TITLE_RE,
    "awards": AWARDS_RE,
    "executive": EXECUTIVE_RE,
    "research": RESEARCH_EVIDENCE_RE,
    "location": LOCATION_SIGNAL_RE,
}


def detect_fact_type(question: str, query_type: str) -> str:
    """Map a question to a strict fact type used for evidence gating."""
    lower = (question or "").lower()
    if re.search(
        r"\b(what (?:group|organization|club).+(?:found|establish)|"
        r"(?:group|organization|club).+(?:found|establish)|"
        r"founded|founder|establish(?:ed)?)\b",
        lower,
    ):
        return "founded"
    if AWARDS_RE.search(lower) and re.search(
        r"\b(award|honor|honour|recognition|scholarship|grammy|oscar|emmy|nobel|win|won|receive)\b",
        lower,
    ):
        return "awards"
    if EXECUTIVE_RE.search(lower) or re.search(
        r"\b(what company|which company).+\b(ceo|run|lead)\b|\bceo of\b",
        lower,
    ):
        return "executive"
    if INSTRUMENT_RE.search(lower) and re.search(
        r"\b(instrument|play|plays|playing|played)\b", lower
    ):
        return "instrument"
    if query_type == "location" or re.search(
        r"\b(what state|which state|where .+ located|school located|hometown)\b",
        lower,
    ):
        return "location"
    if query_type in {
        "price",
        "quantity",
        "date",
        "policy",
        "accessibility",
    }:
        return query_type
    from app.retrieval.numeric_facts import detect_numeric_fact_type

    numeric = detect_numeric_fact_type(lower)
    if numeric:
        return numeric
    if re.search(r"\b(accessib|disabilit|wheelchair|ada)\b", lower):
        return "accessibility"
    if re.search(
        r"\b(refund|cancellation|policy|prohibited|allowed|pets?|volunteer|"
        r"compost|rental|alcohol)\b",
        lower,
    ):
        return "policy"
    if query_type in FACT_TYPE_PATTERNS and query_type not in {"school"}:
        return query_type
    if query_type == "school":
        return "school"
    if query_type in {"major", "education"}:
        if re.search(r"\bminor\b", lower) and re.search(r"\bmajor\b", lower):
            return "major_and_minor"
        if re.search(r"\bminor\b", lower):
            return "minor"
        if re.search(
            r"\b(school|college|university|institution|attend)", lower
        ) and not re.search(r"\b(major|degree|field of study|stud(?:y|ies))", lower):
            return "school"
        return "major"
    if query_type == "instrument":
        return "instrument"
    if query_type == "leadership":
        if re.search(r"\b(found|establish)\b", lower):
            return "founded"
        return "leadership"
    if query_type == "research":
        return "research"
    if GPA_RE.search(lower) and re.search(r"\b(gpa|grade[- ]?point|average)\b", lower):
        return "gpa"
    if BIRTHDAY_RE.search(lower):
        return "birthday"
    if FOOD_RE.search(lower) or re.search(r"\bfavorite food\b", lower):
        return "favorite_food"
    return query_type


def evidence_matches_fact_type(content: str, fact_type: str, chunk: RetrievedChunk | None = None) -> bool:
    text = content or ""
    if fact_type == "gpa":
        return bool(GPA_RE.search(text))
    if fact_type == "birthday":
        return bool(BIRTHDAY_RE.search(text))
    if fact_type == "favorite_food":
        return bool(FOOD_RE.search(text) or re.search(r"(?i)\bfood\b\s*:", text))
    if fact_type == "instrument":
        return bool(INSTRUMENT_RE.search(text))
    if fact_type == "awards":
        return bool(AWARDS_RE.search(text)) and not re.match(
            r"(?i)^(awards?|honors?|recognitions?)(?:\s+and/?or\s+honors?)?\s*$",
            text.strip(),
        )
    if fact_type == "executive":
        return bool(
            EXECUTIVE_RE.search(text)
            or re.search(r"(?i)\b(chief executive|ceo)\b", text)
        )
    if fact_type == "research":
        return bool(
            RESEARCH_EVIDENCE_RE.search(text)
            or (chunk is not None and chunk.record_type == "research")
        )
    if fact_type == "location":
        from app.retrieval.numeric_facts import (
            content_has_address_or_place,
            content_states_unannounced,
        )

        return bool(
            LOCATION_SIGNAL_RE.search(text)
            or content_has_address_or_place(text)
            or content_states_unannounced(text)
        )
    if fact_type == "price":
        from app.retrieval.numeric_facts import content_has_currency

        return content_has_currency(text)
    if fact_type == "quantity":
        from app.retrieval.numeric_facts import content_has_quantity

        return content_has_quantity(text)
    if fact_type == "date":
        from app.retrieval.numeric_facts import (
            content_has_date_or_period,
            content_states_unannounced,
        )

        return content_has_date_or_period(text) or content_states_unannounced(text)
    if fact_type == "policy":
        from app.retrieval.numeric_facts import content_has_policy_rule

        return content_has_policy_rule(text) or bool(
            re.search(
                r"(?i)\b(membership|volunteer|compost|refund|cancel|rental|pet)\b",
                text,
            )
        )
    if fact_type == "accessibility":
        from app.retrieval.numeric_facts import (
            content_has_accessibility,
            content_states_unannounced,
        )

        return content_has_accessibility(text) or content_states_unannounced(text)
    if fact_type == "school":
        from app.retrieval.institution import (
            INSTITUTION_WORD_RE,
            content_has_institution_evidence,
        )

        blob = text
        if chunk is not None:
            parts = [text]
            if chunk.organization:
                parts.append(chunk.organization)
            if chunk.label and chunk.value:
                parts.append(f"{chunk.label}: {chunk.value}")
            blob = "\n".join(parts)
            family = label_family(chunk.label or "")
            if family == "organization" and (chunk.value or "").strip():
                return True
            if chunk.organization and INSTITUTION_WORD_RE.search(chunk.organization):
                return True
        if re.search(r"(?i)\bmajor\s*:", blob) and not content_has_institution_evidence(blob):
            return False
        return content_has_institution_evidence(blob) or bool(
            INSTITUTION_WORD_RE.search(blob)
        )
    if fact_type == "major":
        if chunk is not None:
            family = label_family(chunk.label or "")
            if family == "major":
                return True
            if chunk.record_type == "education" and chunk.title:
                return True
        return bool(
            re.search(r"(?i)\b(major|field of study|academic program)\s*:", text)
            or MAJOR_VALUE_RE.search(text)
        )
    if fact_type == "minor":
        return bool(re.search(r"(?i)\bminor\s*:", text) or (chunk and label_family(chunk.label or "") == "minor"))
    if fact_type == "major_and_minor":
        return evidence_matches_fact_type(text, "major", chunk) or evidence_matches_fact_type(
            text, "minor", chunk
        )
    if fact_type == "leadership":
        return bool(LEADERSHIP_TITLE_RE.search(text))
    if fact_type == "founded":
        return bool(
            re.search(r"(?i)\b(founder|co-?founder|founded|established)\b", text)
        )
    return True


def filter_evidence_for_fact(
    evidence: Sequence[RetrievedChunk],
    fact_type: str,
    *,
    question: str | None = None,
) -> List[RetrievedChunk]:
    """Keep only chunks that can support the requested fact type."""
    strict_types = {
        "gpa",
        "birthday",
        "favorite_food",
        "instrument",
        "school",
        "major",
        "minor",
        "major_and_minor",
        "leadership",
        "founded",
        "awards",
        "executive",
        "research",
        "location",
        "price",
        "quantity",
        "date",
        "policy",
        "accessibility",
    }
    if fact_type in strict_types:
        matched = [
            item
            for item in evidence
            if evidence_matches_fact_type(item.content, fact_type, item)
        ]
        return matched
    # Drop bare contact/identity KV dumps unless the question asks for them.
    q = (question or "").lower()
    keep_contact = bool(
        re.search(
            r"\b(email|e-?mail|phone|contact|full\s*name|what(?:'s| is) (?:his|her|their) name)\b",
            q,
        )
        or fact_type in {"identity", "email", "summary"}
    )
    if keep_contact:
        return list(evidence)
    return [item for item in evidence if not _is_contact_only_chunk(item)]


def _is_contact_only_chunk(chunk: RetrievedChunk) -> bool:
    family = label_family(chunk.label or "")
    if family in {"email", "phone", "name"} and chunk.content_type == "key_value":
        return True
    text = (chunk.content or "").strip()
    return bool(UNRELATED_IDENTITY_RE.match(text) and len(text) < 120)


def answer_matches_fact_type(
    answer: str,
    fact_type: str,
    evidence: Sequence[RetrievedChunk],
) -> bool:
    """Reject answers that don't contain supported evidence of the requested type."""
    if not answer or "could not find that information" in answer.lower():
        return False
    # Never accept raw labeled contact dumps as answers.
    if UNRELATED_IDENTITY_RE.match(answer.strip()) or (
        RAW_LABEL_ANSWER_RE.match(answer.strip())
        and fact_type
        not in {"identity", "email", "major", "minor", "gpa", "birthday", "favorite_food"}
    ):
        if fact_type not in {"identity", "email"}:
            return False
    if fact_type in {
        "gpa",
        "birthday",
        "favorite_food",
        "instrument",
        "school",
        "major",
        "minor",
        "major_and_minor",
        "leadership",
        "awards",
        "executive",
        "research",
        "location",
        "founded",
        "price",
        "quantity",
        "date",
        "policy",
        "accessibility",
    }:
        if UNRELATED_IDENTITY_RE.match(answer.strip()):
            return False
        if re.search(r"(?i)^(email|full\s*name|minor)\s+is\b", answer):
            return False

    if fact_type == "price":
        from app.retrieval.numeric_facts import content_has_currency

        return content_has_currency(answer) or any(
            content_has_currency(e.content) for e in evidence
        )
    if fact_type == "quantity":
        from app.retrieval.numeric_facts import content_has_quantity

        return content_has_quantity(answer) or any(
            content_has_quantity(e.content) for e in evidence
        )
    if fact_type == "date":
        from app.retrieval.numeric_facts import (
            content_has_date_or_period,
            content_states_unannounced,
        )

        return (
            content_has_date_or_period(answer)
            or content_states_unannounced(answer)
            or any(
                content_has_date_or_period(e.content) or content_states_unannounced(e.content)
                for e in evidence
            )
        )
    if fact_type == "policy":
        from app.retrieval.numeric_facts import content_has_policy_rule

        return content_has_policy_rule(answer) or _overlap(answer, evidence)
    if fact_type == "accessibility":
        from app.retrieval.numeric_facts import (
            content_has_accessibility,
            content_states_unannounced,
        )

        return (
            content_has_accessibility(answer)
            or content_states_unannounced(answer)
            or any(content_has_accessibility(e.content) for e in evidence)
        )
    if fact_type == "gpa":
        return bool(GPA_RE.search(answer) or any(GPA_RE.search(e.content) for e in evidence))
    if fact_type == "birthday":
        return bool(BIRTHDAY_RE.search(answer) or _value_from_evidence(evidence, ["birthday", "date"]) and _overlap(answer, evidence))
    if fact_type == "favorite_food":
        return bool(FOOD_RE.search(" ".join(e.content for e in evidence))) and _overlap(
            answer, evidence
        )
    if fact_type == "instrument":
        return bool(INSTRUMENT_RE.search(answer)) and _overlap(answer, evidence)
    if fact_type == "awards":
        return bool(AWARDS_RE.search(answer)) and _overlap(answer, evidence)
    if fact_type == "executive":
        return bool(EXECUTIVE_RE.search(answer) or re.search(r"(?i)\bceo\b", answer)) and _overlap(
            answer, evidence
        )
    if fact_type == "research":
        return bool(RESEARCH_EVIDENCE_RE.search(answer) or _overlap(answer, evidence)) and not (
            UNRELATED_IDENTITY_RE.match(answer.strip())
        )
    if fact_type == "location":
        return bool(LOCATION_SIGNAL_RE.search(answer) or _overlap(answer, evidence)) and not re.search(
            r"(?i)\b(inferred|email address|@\w+\.\w+)\b", answer
        )
    if fact_type == "school":
        from app.retrieval.institution import answer_contains_institution

        return answer_contains_institution(answer, evidence)
    if fact_type == "founded":
        return bool(
            re.search(r"(?i)\b(founded|founder|established)\b", answer)
            and _overlap(answer, evidence)
        )
    if fact_type == "major":
        value = (
            extract_labeled_value(evidence, ["major"])
            or _education_title(evidence)
        )
        if value and value.lower() in answer.lower():
            return True
        return bool(
            re.search(r"(?i)(field of study|major)\b", answer)
            and not (
                INSTITUTION_RE.search(answer)
                and not re.search(r"(?i)(field of study|major|degree)\b", answer)
            )
        )
    if fact_type == "minor":
        value = extract_labeled_value(evidence, ["minor"])
        return bool(value and value.lower() in answer.lower())
    if fact_type == "major_and_minor":
        major_ok = answer_matches_fact_type(answer, "major", evidence) or bool(
            extract_labeled_value(evidence, ["major"])
            and extract_labeled_value(evidence, ["major"]).lower() in answer.lower()
        )
        minor_ok = answer_matches_fact_type(answer, "minor", evidence)
        # Accept if at least one requested field is answered correctly; partial OK.
        return major_ok or minor_ok
    if fact_type == "leadership":
        return bool(LEADERSHIP_TITLE_RE.search(answer))
    return not bool(UNRELATED_IDENTITY_RE.match(answer.strip()))


def _value_from_evidence(evidence: Sequence[RetrievedChunk], families: Sequence[str]) -> Optional[str]:
    return extract_labeled_value(evidence, list(families))


def _education_title(evidence: Sequence[RetrievedChunk]) -> Optional[str]:
    for item in evidence:
        if item.record_type == "education" and item.title:
            return item.title
    return None


def _organization_in_answer(answer: str, evidence: Sequence[RetrievedChunk]) -> bool:
    for item in evidence:
        org = item.organization or extract_labeled_value([item], ["organization"])
        if org and org.lower() in answer.lower():
            return True
        if item.value and INSTITUTION_RE.search(item.value) and item.value.lower() in answer.lower():
            return True
    return False


def _overlap(answer: str, evidence: Sequence[RetrievedChunk]) -> bool:
    answer_l = answer.lower()
    answer_tokens = set(re.findall(r"[a-z0-9]{4,}", answer_l))
    for item in evidence:
        tokens = re.findall(r"[a-z0-9]{4,}", (item.content or "").lower())
        # Prefer overlap between answer tokens and evidence tokens (not just the
        # first evidence tokens, which may be unrelated list preamble).
        if answer_tokens and any(token in answer_tokens for token in tokens):
            return True
        if any(token in answer_l for token in tokens[:40]):
            return True
    return False
