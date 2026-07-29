"""Shared procedure continuity context for compound multi-file questions.

Document-agnostic: no city names, filenames, windows, or prices hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.ingestion.service_domain import (
    GENERAL,
    HEALTH_INSURANCE,
    RESIDENT_REGISTRATION,
    detect_service_domain,
    extract_query_service_domain,
    rank_service_domains,
)
from app.models.api import RetrievedChunk
from app.retrieval.query_understanding import extract_procedural_topic

PROCEDURE_LABELS = {
    RESIDENT_REGISTRATION: "move-in resident-registration",
    HEALTH_INSURANCE: "health-insurance enrollment",
    "certificate_fees": "certificate issuance and fees",
    "waste_recycling": "waste and recycling collection",
    "childcare_support": "childcare support / child allowance",
    "disaster_preparedness": "disaster preparedness",
}

# Explicit second-topic cues that may unlock a domain switch mid-question.
INSURANCE_TOPIC_RE = re.compile(
    r"(?i)\b("
    r"insurance|employer(?:'?s)?\s+(?:coverage|insurance)|nhi|"
    r"premiums?|medical\s+coverage|national\s+health|"
    r"enroll(?:ment)?\s+in\s+(?:health\s+)?insurance|"
    r"left\s+(?:my\s+)?employer|leave\s+(?:my\s+)?employer"
    r")\b"
)
INSURANCE_CONFLICT_RE = re.compile(
    r"(?i)\b("
    r"employer(?:'?s)?\s+insurance|national\s+health\s+insurance|\bnhi\b|"
    r"insurance\s+premiums?|premiums?\s+are\s+billed|"
    r"insurance\s+(?:and|&)\s+pension|"
    r"certificate\s+of\s+loss(?:\s+of\s+(?:eligibility|insurance))?|"
    r"nhi\s+card\s+is\s+mailed|enroll\s+in\s+(?:national\s+)?(?:health\s+)?insurance"
    r")\b"
)
MOVE_IN_QUERY_RE = re.compile(
    r"(?i)\b("
    r"moved?\s+to|moving\s+in|move-?in|new\s+address|"
    r"register(?:\s+my)?\s+address|resident\s+registration|"
    r"just\s+moved"
    r")\b"
)
MOVE_IN_HEADING_RE = re.compile(
    r"(?i)\b("
    r"moving\s+in|move-?in(?:\s+notification)?|resident\s+registration|"
    r"new\s+(?:address|resident)|citizen\s+affairs"
    r")\b"
)
RESIDENT_TOPIC_RE = re.compile(
    r"(?i)\b("
    r"register(?:ation)?(?:\s+my\s+address)?|moving\s+in|moved|"
    r"move-?in|resident\s+registration|change\s+of\s+address|"
    r"address\s+registration"
    r")\b"
)


def is_move_in_registration_question(question: str) -> bool:
    text = question or ""
    if not MOVE_IN_QUERY_RE.search(text):
        return False
    # Explicit insurance dual-topic questions are allowed to use insurance evidence.
    if INSURANCE_TOPIC_RE.search(text) and question_explicitly_spans_domains(text):
        return False
    return True


def evidence_dominated_by_insurance(content: str, *, section_title: str = "") -> bool:
    """True when a chunk is primarily about insurance enrollment, not address registration."""
    blob = f"{section_title}\n{content}"
    insurance_hits = len(INSURANCE_CONFLICT_RE.findall(blob))
    move_hits = len(
        re.findall(
            r"(?i)\b(move-?in\s+notification|citizen\s+affairs|moving-out\s+certificate|"
            r"resident\s+registration|new\s+address)\b",
            blob,
        )
    )
    if insurance_hits <= 0:
        return False
    return insurance_hits >= max(1, move_hits)


def reject_insurance_evidence_for_move_in(
    *,
    question: str,
    content: str,
    section_title: str = "",
    service_domain: str = "",
) -> bool:
    """Hard rule: move-in/address-registration questions reject insurance-dominated evidence."""
    if not is_move_in_registration_question(question):
        return False
    if (service_domain or "") == HEALTH_INSURANCE:
        return True
    return evidence_dominated_by_insurance(content, section_title=section_title)
ELLIPTICAL_FOLLOWUP_RE = re.compile(
    r"(?i)^\s*(?:and\s+)?(?:"
    r"what\s+do\s+i\s+bring|"
    r"what\s+(?:documents?|items?)\s+(?:do\s+i\s+)?(?:need|bring)|"
    r"where\s+do\s+i\s+go|"
    r"how\s+much|"
    r"when|"
    r"what\s+about\s+that|"
    r"and\s+what\s+do\s+i\s+bring"
    r")\??\s*$"
)


@dataclass
class ProcedureContext:
    active_domain: str = GENERAL
    active_procedure: Optional[str] = None
    active_document: Optional[str] = None
    active_section: Optional[str] = None
    active_entity: Optional[str] = None
    active_action: Optional[str] = None
    allow_domain_switch: bool = False
    original_question: str = ""
    locked: bool = True

    def procedure_label(self) -> str:
        if self.active_procedure:
            return self.active_procedure
        return PROCEDURE_LABELS.get(self.active_domain, "the current procedure")

    def updated_from_evidence(
        self, evidence: Sequence[RetrievedChunk]
    ) -> "ProcedureContext":
        if not evidence:
            return self
        top = evidence[0]
        domain = top.service_domain or self.active_domain
        if self.locked and not self.allow_domain_switch and self.active_domain != GENERAL:
            domain = self.active_domain
        return replace(
            self,
            active_domain=domain or self.active_domain,
            active_document=top.document_name or self.active_document,
            active_section=top.section_title or self.active_section,
            active_entity=top.organization or self.active_entity,
            active_action=self.active_action
            or _infer_action(top.section_title or "", top.content or ""),
        )

    def to_diagnostics(self) -> Dict[str, Any]:
        return {
            "active_domain": self.active_domain,
            "active_procedure": self.active_procedure,
            "active_document": self.active_document,
            "active_section": self.active_section,
            "active_entity": self.active_entity,
            "active_action": self.active_action,
            "allow_domain_switch": self.allow_domain_switch,
            "locked": self.locked,
        }


def build_procedure_context(original_question: str) -> ProcedureContext:
    """Detect primary domain/procedure from the full question before split."""
    question = (original_question or "").strip()
    domains = extract_explicit_query_domains(question)
    allow_switch = question_explicitly_spans_domains(question)
    if allow_switch and domains:
        # Prefer the first procedural topic in reading order for the initial lock state,
        # but keep switch enabled so later clauses can change domain.
        primary = domains[0]
    else:
        primary = domains[0] if domains else extract_query_service_domain(question)
    topic = extract_procedural_topic(question)
    if allow_switch:
        # Dual-topic questions do not share one procedure label across clauses.
        label = PROCEDURE_LABELS.get(primary) or topic
    else:
        label = topic or PROCEDURE_LABELS.get(primary)
    action = None
    if primary == RESIDENT_REGISTRATION:
        action = "register"
    elif primary == HEALTH_INSURANCE:
        action = "enroll"
    elif re.search(r"(?i)\bapply\b", question):
        action = "apply"
    return ProcedureContext(
        active_domain=primary,
        active_procedure=label,
        active_action=action,
        allow_domain_switch=allow_switch,
        original_question=question,
        locked=not allow_switch,
    )


def extract_explicit_query_domains(question: str) -> List[str]:
    """Ordered distinct domains mentioned strongly enough in the question."""
    ranked = rank_service_domains(text=question or "", filename="", headings=[])
    domains = [item.domain for item in ranked if item.score >= 0.35]
    if not domains:
        fallback = extract_query_service_domain(question)
        return [fallback] if fallback != GENERAL else []
    # Preserve specialty ordering for explicit dual-topic questions.
    if question_explicitly_spans_domains(question):
        ordered: List[str] = []
        if RESIDENT_TOPIC_RE.search(question or "") and RESIDENT_REGISTRATION in domains:
            ordered.append(RESIDENT_REGISTRATION)
        if INSURANCE_TOPIC_RE.search(question or "") and HEALTH_INSURANCE in domains:
            ordered.append(HEALTH_INSURANCE)
        for domain in domains:
            if domain not in ordered:
                ordered.append(domain)
        return ordered
    return domains


def question_explicitly_spans_domains(question: str) -> bool:
    """True only when the user clearly introduces two procedures."""
    text = question or ""
    has_resident = bool(RESIDENT_TOPIC_RE.search(text))
    has_insurance = bool(INSURANCE_TOPIC_RE.search(text))
    if not (has_resident and has_insurance):
        return False
    # "When must I register, and what do I bring?" mentions neither insurance.
    # Dual intent usually pairs move/register with leave employer / enroll.
    return bool(
        re.search(
            r"(?i)\b("
            r"and\s+(?:also\s+)?(?:left|leave|enroll)|"
            r"also\s+left|also\s+leave|"
            r"register(?:\s+my)?\s+address.+(?:enroll|insurance)|"
            r"(?:enroll|insurance).+register(?:\s+my)?\s+address|"
            r"moved.+(?:left|leave).+employer|"
            r"left.+(?:employer|insurance).+moved|"
            r"moved here and left"
            r")\b",
            text,
        )
    )


def assign_subquestion_domains(
    *,
    original_question: str,
    sub_questions: Sequence[str],
    context: ProcedureContext,
) -> List[str]:
    """Map each sub-question to a domain, preserving continuity by default."""
    if not sub_questions:
        return []
    if not context.allow_domain_switch:
        return [context.active_domain] * len(sub_questions)
    assigned: List[str] = []
    for sub in sub_questions:
        local = extract_query_service_domain(sub)
        if local == GENERAL:
            # Use sub + original cues without letting the other topic dominate.
            if re.search(r"(?i)\b(enroll|insurance|employer|nhi|premium)\b", sub):
                local = HEALTH_INSURANCE
            elif re.search(
                r"(?i)\b(register|address|moving|move-?in|bring|notification)\b", sub
            ):
                local = RESIDENT_REGISTRATION
            else:
                local = context.active_domain
        assigned.append(local)
    return assigned


def is_elliptical_followup(question: str) -> bool:
    return bool(ELLIPTICAL_FOLLOWUP_RE.match((question or "").strip()))


def conflicting_domain_allowed(
    *,
    active_domain: str,
    candidate_domain: Optional[str],
    original_question: str,
    allow_domain_switch: bool,
) -> bool:
    """Whether evidence from candidate_domain may be used under active_domain."""
    active = (active_domain or GENERAL).strip() or GENERAL
    candidate = (candidate_domain or "").strip()
    if not candidate or candidate == GENERAL or active == GENERAL:
        return True
    if candidate == active:
        return True
    if allow_domain_switch:
        return True
    # Hard reject health-insurance bleed into resident-registration continuity.
    if active == RESIDENT_REGISTRATION and candidate == HEALTH_INSURANCE:
        return bool(INSURANCE_TOPIC_RE.search(original_question or ""))
    if active == HEALTH_INSURANCE and candidate == RESIDENT_REGISTRATION:
        # Allow only when the original also asked about address registration.
        return bool(
            re.search(
                r"(?i)\b(register(?:\s+my)?\s+address|moving\s+in|moved here and)\b",
                original_question or "",
            )
        )
    return False


def continuity_score(
    *,
    context: ProcedureContext,
    payload: Dict[str, Any],
    content: str = "",
) -> Tuple[float, float, str]:
    """Return (continuity_boost, cross_domain_penalty, reason)."""
    if context.active_domain == GENERAL and not context.active_document:
        return 0.0, 0.0, "no_active_procedure"

    document = str(payload.get("document_name") or "")
    section = str(payload.get("section_title") or "")
    organization = str(payload.get("organization") or "")
    candidate_domain = str(payload.get("service_domain") or "").strip()
    if not candidate_domain or candidate_domain == GENERAL:
        candidate_domain = detect_service_domain(
            filename=document,
            headings=[section] if section else [],
            text=(content or str(payload.get("content") or ""))[:1200],
        )

    boost = 0.0
    penalty = 0.0
    reasons: List[str] = []

    if candidate_domain == context.active_domain and context.active_domain != GENERAL:
        boost += 1.6
        reasons.append("same_domain")
    elif candidate_domain and candidate_domain != GENERAL:
        if not conflicting_domain_allowed(
            active_domain=context.active_domain,
            candidate_domain=candidate_domain,
            original_question=context.original_question,
            allow_domain_switch=context.allow_domain_switch,
        ):
            penalty += 3.2
            reasons.append("cross_domain_reject")
        else:
            penalty += 1.55
            reasons.append("domain_mismatch")

    if context.active_document and document:
        if document.lower() == context.active_document.lower():
            boost += 1.25
            reasons.append("same_document")
        else:
            # Soft penalty for jumping docs while procedure is locked.
            if context.locked and context.active_domain != GENERAL:
                penalty += 0.45
                reasons.append("document_switch")

    if context.active_section and section:
        if section.strip().lower() == context.active_section.strip().lower():
            boost += 1.1
            reasons.append("same_section")

    if context.active_action:
        blob = f"{section}\n{content}\n{payload.get('content') or ''}".lower()
        if context.active_action.lower() in blob:
            boost += 0.35
            reasons.append("same_action")

    if context.active_entity and organization:
        if context.active_entity.lower() in organization.lower():
            boost += 0.35
            reasons.append("same_office")

    reason = "+".join(reasons) if reasons else "neutral"
    return boost, penalty, reason


def filter_evidence_to_procedure(
    evidence: Sequence[RetrievedChunk],
    context: ProcedureContext,
) -> List[RetrievedChunk]:
    """Keep checklist/procedure evidence inside the active domain/document."""
    if not evidence or context.active_domain == GENERAL:
        return list(evidence)
    kept: List[RetrievedChunk] = []
    for chunk in evidence:
        domain = chunk.service_domain
        if not domain or domain == GENERAL:
            domain = detect_service_domain(
                filename=chunk.document_name or "",
                headings=[chunk.section_title or ""],
                text=(chunk.content or "")[:1200],
            )
        if reject_insurance_evidence_for_move_in(
            question=context.original_question,
            content=chunk.content or "",
            section_title=chunk.section_title or "",
            service_domain=domain or "",
        ):
            continue
        if conflicting_domain_allowed(
            active_domain=context.active_domain,
            candidate_domain=domain,
            original_question=context.original_question,
            allow_domain_switch=context.allow_domain_switch,
        ) and (
            not context.locked
            or not context.active_document
            or not chunk.document_name
            or chunk.document_name.lower() == context.active_document.lower()
            or domain == context.active_domain
        ):
            # Prefer same domain; allow same-domain other docs.
            if domain == context.active_domain or not domain or domain == GENERAL:
                kept.append(chunk)
            elif context.allow_domain_switch:
                kept.append(chunk)
    if kept:
        return kept
    # Fall back to same-domain only.
    same = [
        chunk
        for chunk in evidence
        if (chunk.service_domain or "") == context.active_domain
        and not reject_insurance_evidence_for_move_in(
            question=context.original_question,
            content=chunk.content or "",
            section_title=chunk.section_title or "",
            service_domain=chunk.service_domain or "",
        )
    ]
    return same or list(evidence)


def _infer_action(section: str, content: str) -> Optional[str]:
    blob = f"{section}\n{content}".lower()
    for token in ("register", "enroll", "apply", "submit", "notify", "collect"):
        if token in blob:
            return token
    return None
