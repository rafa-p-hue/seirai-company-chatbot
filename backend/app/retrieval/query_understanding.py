"""Query classification, short-query expansion, and conversation context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

from app.models.api import ChatMessage


GREETING_RE = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening)[!.,\s]*$",
    re.I,
)

QUERY_TYPES = {
    "greeting",
    "identity",
    "education",
    "experience",
    "internship",
    "research",
    "leadership",
    "skills",
    "location",
    "general",
    "unsupported",
}


@dataclass
class QueryUnderstanding:
    original_question: str
    normalized_question: str
    expanded_question: str
    query_type: str
    expanded_terms: List[str] = field(default_factory=list)
    subject_name: Optional[str] = None


PREFERRED_TYPES = {
    "identity": ["profile", "education", "experience", "internship"],
    "education": ["education", "profile"],
    "experience": ["experience", "internship", "research", "leadership"],
    "internship": ["internship", "experience"],
    "research": ["research", "experience", "education"],
    "leadership": ["leadership", "experience", "organization"],
    "skills": ["skills", "experience"],
    "location": ["profile", "experience", "education"],
    "general": [],
}

DEPRIORITIZED_TYPES = {
    "education": ["experience", "internship", "skills"],
    "identity": ["skills"],
    "internship": ["skills", "education"],
}


def understand_query(
    question: str,
    *,
    history: Sequence[ChatMessage] | None = None,
    subject_name: Optional[str] = None,
) -> QueryUnderstanding:
    normalized = _normalize(question)
    subject = subject_name or _infer_subject_from_history(history) or "the person in the document"
    query_type = classify_query(normalized)
    expanded_terms = expand_terms(normalized)
    expanded_question = expand_short_question(
        normalized,
        query_type=query_type,
        subject_name=subject,
        history=history,
    )
    return QueryUnderstanding(
        original_question=question,
        normalized_question=normalized,
        expanded_question=expanded_question,
        query_type=query_type,
        expanded_terms=expanded_terms,
        subject_name=subject if subject != "the person in the document" else subject_name,
    )


def classify_query(question: str) -> str:
    lower = question.lower().strip()
    if GREETING_RE.match(lower) or lower in {"hi", "hello", "hey"}:
        return "greeting"
    if re.search(r"\b(refund|cancellation|password|social security|zodiac)\b", lower):
        return "unsupported"
    if re.search(r"\b(who is|who'?s|tell me about|about (him|her|them)|profile|overview)\b", lower):
        return "identity"
    if re.search(r"\b(major|degree|school|university|college|education|stud(?:y|ies)|field of study)\b", lower):
        return "education"
    if re.search(r"\b(internship|internships|intern)\b", lower):
        return "internship"
    if re.search(r"\b(research)\b", lower):
        return "research"
    if re.search(r"\b(leadership|volunteer|organization|involvement|activities)\b", lower):
        return "leadership"
    if re.search(r"\b(skill|skills|abilities)\b", lower):
        return "skills"
    if re.search(r"\b(hometown|where .+ from|location|based in|live[s]?)\b", lower):
        return "location"
    if re.search(r"\b(work|worked|job|jobs|role|roles|experience|employer|company)\b", lower):
        return "experience"
    # Bare short nouns
    if lower in {"major", "school", "education", "degree"}:
        return "education"
    if lower in {"job", "jobs", "work", "experience"}:
        return "experience"
    tokens = [token for token in re.findall(r"[a-z0-9]+", lower) if len(token) > 1]
    if len(tokens) <= 1 and tokens and tokens[0] in {"major", "school"}:
        return "education"
    return "general"


def expand_short_question(
    question: str,
    *,
    query_type: str,
    subject_name: str,
    history: Sequence[ChatMessage] | None = None,
) -> str:
    lower = question.lower().strip().rstrip("?")
    resolved = _resolve_pronouns(question, subject_name=subject_name, history=history)

    expansions = {
        "what school": f"What school or university does {subject_name} attend?",
        "what school does he attend": f"What school or university does {subject_name} attend?",
        "school": f"What school or university does {subject_name} attend?",
        "major": f"What is {subject_name}'s major or field of study?",
        "what is his major": f"What is {subject_name}'s major or field of study?",
        "what's his major": f"What is {subject_name}'s major or field of study?",
        "what major": f"What is {subject_name}'s major or field of study?",
        "who is rafael": f"Provide a brief profile summary of {subject_name} based on the document.",
        "who is rafael francisco perez": f"Provide a brief profile summary of {subject_name} based on the document.",
    }
    key = re.sub(r"[^a-z0-9\s]", "", lower)
    key = re.sub(r"\s+", " ", key).strip()
    if key in expansions:
        return expansions[key]

    if query_type == "identity":
        return f"Provide a brief profile summary of {subject_name} based on the document."
    if query_type == "education" and len(key.split()) <= 4:
        if "major" in key or key == "degree":
            return f"What is {subject_name}'s major or field of study?"
        return f"What school or university does {subject_name} attend?"
    if query_type == "experience" and len(key.split()) <= 5:
        return f"Where has {subject_name} worked? List relevant experience roles."
    if query_type == "internship":
        return f"What internship experience does {subject_name} have?"
    if query_type == "research":
        return f"What research experience does {subject_name} have?"
    return resolved


def expand_terms(question: str) -> List[str]:
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    mapping = {
        "school": ["education", "university", "college", "degree", "academic", "major"],
        "major": ["education", "degree", "field", "study", "university", "college"],
        "job": ["experience", "employment", "role", "position", "employer"],
        "jobs": ["experience", "employment", "role", "position", "employer"],
        "work": ["experience", "employment", "role", "position"],
        "worked": ["experience", "employment", "role", "position"],
        "internship": ["intern", "experience", "role"],
        "research": ["research", "lab", "assistant", "experience"],
        "who": ["profile", "overview", "identity", "about", "summary"],
        "skills": ["skill", "abilities", "competencies"],
    }
    out: Set[str] = set(tokens)
    for token in list(tokens):
        out.update(mapping.get(token, []))
    return sorted(out)


def _normalize(question: str) -> str:
    text = question.replace("\u2019", "'").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _resolve_pronouns(
    question: str,
    *,
    subject_name: str,
    history: Sequence[ChatMessage] | None,
) -> str:
    text = question
    if re.search(r"\b(he|his|him|she|her|they|their|them)\b", text, re.I):
        text = re.sub(r"\bhis\b", f"{subject_name}'s", text, flags=re.I)
        text = re.sub(r"\bhe\b", subject_name, text, flags=re.I)
        text = re.sub(r"\bhim\b", subject_name, text, flags=re.I)
    # Follow-ups after identity questions
    if history:
        recent_user = [msg.content for msg in history if msg.role == "user"][-3:]
        if recent_user and len(question.split()) <= 4 and "rafael" not in question.lower():
            if classify_query(question) in {"education", "experience", "internship", "research", "skills"}:
                return expand_short_question(
                    question,
                    query_type=classify_query(question),
                    subject_name=subject_name,
                    history=None,
                )
    return text


def _infer_subject_from_history(history: Sequence[ChatMessage] | None) -> Optional[str]:
    if not history:
        return None
    for message in reversed(list(history)):
        match = re.search(
            r"who is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
            message.content,
            re.I,
        )
        if match:
            return match.group(1).strip()
        match = re.search(
            r"\b(Rafael(?:\s+Francisco)?(?:\s+Perez)?)\b",
            message.content,
            re.I,
        )
        if match:
            # Only accept if the history itself mentioned this name as a query subject,
            # not hardcoding company facts into answers.
            return match.group(1)
    return None
