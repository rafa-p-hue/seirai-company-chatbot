"""Cross-turn follow-up resolution and conversational anchors.

Document-agnostic: extracts entities/topics from prior user + assistant turns.
Never hardcodes corpus-specific organization names, fees, or schedules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

from app.models.api import ChatMessage
from app.retrieval.procedure_context import ProcedureContext, build_procedure_context


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _previous_user_question(
    history: Sequence[ChatMessage] | None, current_question: str
) -> Optional[str]:
    current = _normalize(current_question).rstrip(" ?").lower()
    for message in reversed(list(history or [])):
        if message.role != "user":
            continue
        candidate = _normalize(message.content).strip()
        if not candidate or candidate.rstrip(" ?").lower() == current:
            continue
        return candidate
    return None


def _extract_procedural_topic(text: str) -> Optional[str]:
    from app.retrieval.query_understanding import extract_procedural_topic

    return extract_procedural_topic(text)


ELLIPTICAL_ATTRIBUTE_RE = re.compile(
    r"(?i)^\s*(?:and\s+)?"
    r"(?:"
    r"what\s+is\s+(?:the\s+)?(?:capacity|fee|cost|price|deadline|schedule|"
    r"address|location|phone|hours?|size|amount)|"
    r"when\s+is\s+(?:(?:the|that|it|this)\s+)?(?:\w+\s+){0,3}"
    r"(?:allowance|benefit|payment|fee|collected|paid|due|open)|"
    r"when\s+(?:is|are)\s+(?:they|it|that|this)\s+(?:paid|due|collected)|"
    r"how\s+much\s+(?:is\s+)?(?:it|that|this)|"
    r"where\s+is\s+(?:it|that|this)|"
    r"what\s+about\s+(?:it|that|this|the\s+\w+)"
    r")\??\s*$"
)

TITLE_CASE_ENTITY_RE = re.compile(
    r"\b("
    r"(?:[A-Z][A-Za-z0-9&'/-]+(?:\s+(?:of|and|the|for|at|in))?){0,2}"
    r"(?:\s+[A-Z][A-Za-z0-9&'/-]+){1,5}"
    r")\b"
)

PROGRAM_TOPIC_RE = re.compile(
    r"(?i)\b("
    r"child\s+allowance|childcare\s+(?:support|allowance)|child\s+benefit|"
    r"health\s+insurance|national\s+health\s+insurance|"
    r"resident(?:ial)?\s+(?:registration|parking)|"
    r"residence\s+certificate|family\s+register|"
    r"burnable\s+garbage|oversized\s+garbage|"
    r"evacuation\s+shelter|disaster\s+preparedness|"
    r"waste\s+(?:and\s+)?recycling|garbage\s+collection"
    r")\b"
)

ORG_HINT_RE = re.compile(
    r"(?i)\b("
    r"shelter|center|centre|gym|school|office|window|desk|counter|"
    r"hall|park|stadium|facility|site|station"
    r")\b"
)

STOP_ENTITIES = {
    "the",
    "this",
    "that",
    "when",
    "what",
    "which",
    "how",
    "status",
    "effective",
    "april",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "february",
    "june",
    "october",
    "january",
    "march",
    "may",
    "july",
    "august",
    "september",
    "november",
    "december",
}


@dataclass
class ConversationAnchors:
    organization: Optional[str] = None
    program: Optional[str] = None
    location: Optional[str] = None
    procedure: Optional[str] = None
    benefit: Optional[str] = None
    document: Optional[str] = None
    section: Optional[str] = None
    entity_phrase: Optional[str] = None
    prior_user_question: Optional[str] = None
    prior_assistant_answer: Optional[str] = None
    topic_terms: List[str] = field(default_factory=list)

    def primary_entity(self) -> Optional[str]:
        return (
            self.organization
            or self.entity_phrase
            or self.benefit
            or self.program
            or self.procedure
        )

    def to_diagnostics(self) -> dict:
        return {
            "organization": self.organization,
            "program": self.program,
            "location": self.location,
            "procedure": self.procedure,
            "benefit": self.benefit,
            "document": self.document,
            "section": self.section,
            "entity_phrase": self.entity_phrase,
            "primary_entity": self.primary_entity(),
            "prior_user_question": self.prior_user_question,
            "topic_terms": list(self.topic_terms),
        }


def is_elliptical_followup_question(question: str) -> bool:
    text = _normalize(question or "").strip()
    if not text:
        return False
    if ELLIPTICAL_ATTRIBUTE_RE.match(text):
        return True
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) <= 6 and re.search(
        r"(?i)\b(capacity|paid|fee|cost|when|where|how much|schedule)\b", text
    ):
        # Short attribute questions without an explicit subject noun phrase.
        has_named_subject = bool(
            re.search(
                r"(?i)\b(of|for|at|about)\s+(?:the\s+)?[a-z]{3,}",
                text,
            )
            and not re.search(r"(?i)\b(of|for)\s+(?:the\s+)?(?:capacity|fee)\b", text)
        )
        if not has_named_subject:
            return True
    return False


def previous_turn(
    history: Sequence[ChatMessage] | None,
    *,
    current_question: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """Return (prior_user_question, prior_assistant_answer) immediately before now."""
    messages = list(history or [])
    prior_user = _previous_user_question(messages, current_question)
    prior_assistant: Optional[str] = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role != "assistant":
            continue
        prior_assistant = _normalize(messages[index].content).strip() or None
        # Prefer the assistant reply that followed the prior user question.
        if prior_user:
            for earlier in range(index - 1, -1, -1):
                if messages[earlier].role == "user":
                    if _normalize(messages[earlier].content).strip() == prior_user:
                        return prior_user, prior_assistant
                    break
        return prior_user, prior_assistant
    return prior_user, prior_assistant


def extract_anchors_from_turn(
    *,
    user_question: Optional[str],
    assistant_answer: Optional[str] = None,
    document: Optional[str] = None,
    section: Optional[str] = None,
) -> ConversationAnchors:
    user_q = _normalize(user_question or "")
    answer = _normalize(assistant_answer or "")
    anchors = ConversationAnchors(
        prior_user_question=user_q or None,
        prior_assistant_answer=answer or None,
        document=document,
        section=section,
    )

    program_match = PROGRAM_TOPIC_RE.search(user_q) or PROGRAM_TOPIC_RE.search(answer)
    if program_match:
        phrase = re.sub(r"\s+", " ", program_match.group(1)).strip()
        anchors.program = phrase
        if re.search(r"(?i)\ballowance|benefit\b", phrase):
            anchors.benefit = phrase

    procedure = _extract_procedural_topic(user_q) or _extract_procedural_topic(answer)
    if procedure:
        anchors.procedure = procedure

    org = _pick_organization(answer) or _pick_organization(user_q)
    if org:
        anchors.organization = org
        anchors.entity_phrase = org
    elif anchors.program:
        anchors.entity_phrase = anchors.program
    elif anchors.procedure:
        anchors.entity_phrase = anchors.procedure

    # Location cue from answer near the organization.
    if anchors.organization and answer:
        loc = re.search(
            rf"(?i){re.escape(anchors.organization)}.{{0,80}}"
            rf"(?:at|in|near)\s+([A-Z][A-Za-z0-9 ,'-]{{3,60}})",
            answer,
        )
        if loc:
            anchors.location = loc.group(1).strip(" .,")

    anchors.topic_terms = _topic_terms(user_q, answer, anchors)
    return anchors


def anchors_from_history(
    history: Sequence[ChatMessage] | None,
    *,
    current_question: str = "",
) -> ConversationAnchors:
    prior_user, prior_assistant = previous_turn(
        history, current_question=current_question
    )
    return extract_anchors_from_turn(
        user_question=prior_user,
        assistant_answer=prior_assistant,
    )


def resolve_followup_question(
    question: str,
    *,
    history: Sequence[ChatMessage] | None = None,
    anchors: ConversationAnchors | None = None,
) -> str:
    """Rewrite elliptical follow-ups using prior user question + assistant answer.

    Retrieval should use the resolved form; generation should keep the original
    wording (caller responsibility).
    """
    normalized = _normalize(question)
    if not is_elliptical_followup_question(normalized):
        return normalized

    active = anchors or anchors_from_history(history, current_question=normalized)
    entity = active.primary_entity()
    prior = active.prior_user_question or ""
    lower = normalized.lower()

    if re.search(r"(?i)\bcapacity\b", lower):
        if entity:
            return f"What is the capacity of {entity}?"
        if prior:
            return f"What is the capacity of the previously identified subject from: {prior}?"
        return normalized

    if re.search(r"(?i)\bpaid|payment|pay\b", lower):
        benefit = active.benefit or active.program or entity
        if benefit:
            return f"When is {benefit} paid?"
        if prior:
            return f"When is that benefit from the previous question paid?"
        return normalized

    if re.search(r"(?i)\b(fee|cost|price|how much)\b", lower):
        target = entity or active.program
        if target:
            return f"How much does {target} cost?"
        return normalized

    if re.search(r"(?i)\bwhen\b", lower) and entity:
        return f"When is {entity} collected or due?"

    if re.search(r"(?i)\bwhere\b", lower) and entity:
        return f"Where is {entity} located?"

    if entity:
        # Generic attribute follow-up: attach the active subject.
        stripped = normalized.rstrip(" ?")
        return f"{stripped} for {entity}?"
    if prior:
        return f"{normalized.rstrip(' ?')} (regarding: {prior})?"
    return normalized


def build_context_from_history(
    *,
    question: str,
    history: Sequence[ChatMessage] | None,
) -> Tuple[ProcedureContext, ConversationAnchors]:
    """Build procedure continuity from prior turn only when the question is elliptical.

    Fully specified questions rebuild domain/procedure from the current wording so
    a prior topic cannot override retrieval.
    """
    anchors = anchors_from_history(history, current_question=question)
    elliptical = is_elliptical_followup_question(question)
    base_source = (
        (anchors.prior_user_question or question) if elliptical else question
    )
    context = build_procedure_context(base_source)
    if elliptical:
        if anchors.organization:
            context = replace(context, active_entity=anchors.organization)
        if anchors.procedure:
            context = replace(context, active_procedure=anchors.procedure)
        if anchors.document:
            context = replace(context, active_document=anchors.document)
        if anchors.section:
            context = replace(context, active_section=anchors.section)
        if anchors.primary_entity():
            # Lock continuity for singular follow-ups so retrieval stays on subject.
            context = replace(context, locked=True, allow_domain_switch=False)
    else:
        context = replace(
            context,
            locked=False,
            allow_domain_switch=True,
            original_question=question,
            active_document=None,
            active_section=None,
        )
    return context, anchors


def filter_chunks_to_active_entity(
    chunks: Sequence,
    anchors: ConversationAnchors,
    *,
    singular_followup: bool,
) -> List:
    """For singular follow-ups, keep only chunks about the active entity."""
    if not singular_followup:
        return list(chunks)
    entity = anchors.primary_entity()
    if not entity:
        return list(chunks)
    entity_l = entity.lower()
    key_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", entity_l)
        if len(token) > 2 and token not in STOP_ENTITIES
    ]
    if not key_tokens:
        return list(chunks)

    kept = []
    for chunk in chunks:
        blob = " ".join(
            part
            for part in (
                getattr(chunk, "content", None) or "",
                getattr(chunk, "section_title", None) or "",
                getattr(chunk, "document_name", None) or "",
                getattr(chunk, "organization", None) or "",
            )
            if part
        ).lower()
        if entity_l in blob or sum(1 for token in key_tokens if token in blob) >= max(
            1, (len(key_tokens) + 1) // 2
        ):
            kept.append(chunk)
    # Hard singular constraint: never fall back to unrelated entities.
    return kept


def _pick_organization(text: str) -> Optional[str]:
    if not text:
        return None
    candidates: List[str] = []
    for match in TITLE_CASE_ENTITY_RE.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip(" .,;:")
        if len(phrase.split()) < 2:
            continue
        if phrase.lower().split()[0] in STOP_ENTITIES:
            continue
        if ORG_HINT_RE.search(phrase) or re.search(
            r"(?i)\b(shelter|center|centre|gym|office|school|hall)\b",
            text[max(0, match.start() - 40) : match.end() + 40],
        ):
            candidates.append(phrase)
    if candidates:
        # Prefer phrases that look like facility names.
        ranked = sorted(
            candidates,
            key=lambda item: (
                1 if ORG_HINT_RE.search(item) else 0,
                len(item.split()),
            ),
            reverse=True,
        )
        return ranked[0]
    # Fallback: "X accepts pets" / "X is pet-friendly"
    named = re.search(
        r"([A-Z][A-Za-z0-9&'/-]+(?:\s+[A-Z][A-Za-z0-9&'/-]+){1,5})"
        r"\s+(?:accepts?\s+pets?|is\s+pet[- ]friendly|allows?\s+pets?)",
        text,
    )
    if named:
        return named.group(1).strip()
    return None


def _topic_terms(
    user_q: str, answer: str, anchors: ConversationAnchors
) -> List[str]:
    blob = f"{user_q} {answer} {anchors.primary_entity() or ''}"
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", blob.lower())
        if len(token) > 3 and token not in STOP_ENTITIES
    ]
    # Preserve order, unique.
    seen = set()
    out: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out[:24]
