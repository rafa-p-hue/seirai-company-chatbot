"""Source-intent and answer-type grounding (document-agnostic).

Validates that retrieved evidence matches the *kind* of answer the user asked
for — fee vs process noise, salary-guide vs open-role listing, office location
vs contact details, candidate free-of-charge vs employer fee schedule.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models.api import RetrievedChunk

logger = logging.getLogger(__name__)

FEE_INDICATOR_RE = re.compile(
    r"(?i)\b("
    r"fee|fees|minimum\s+fee|percentage|percent|"
    r"first[- ]year\s+base\s+salary|first[- ]year\s+total\s+compensation|"
    r"markup|flat\s+fee|payable|invoice"
    r")\b|"
    r"\d{1,3}(?:\.\d+)?\s*%|"
    r"(?:SGD|USD|GBP|AUD|EUR|¥|\$)\s*[\d,]+"
)

FEE_AMOUNT_RE = re.compile(
    r"(?i)("
    r"\d{1,3}(?:\.\d+)?\s*%\s+of\s+(?:first[- ]year|base\s+salary|total\s+compensation)|"
    r"(?:placement\s+)?fee(?:\s*\([^)]*\))?\s*[:=]\s*\d{1,3}(?:\.\d+)?\s*%|"
    r"(?:minimum\s+fee|flat\s+fee)\s*[:=]?\s*(?:SGD|USD|GBP|AUD|EUR)?\s*[\d,]+|"
    r"(?:SGD|USD|GBP|AUD|EUR)\s*[\d,]+|"
    r"[$¥€£]\s*[\d,]+|"
    r"\d{1,3}(?:\.\d+)?\s*%\s+markup"
    r")"
)

FEE_PROCESS_NOISE_RE = re.compile(
    r"(?i)\b("
    r"replacement\s+guarantee|shortlist(?:ing)?|offer[- ]acceptance|"
    r"acceptance\s+rate|time[- ]to[- ]hire|business\s+days|"
    r"role\s+charter|market\s+mapping|onboarding\s+follow[- ]?up|"
    r"competency[- ]based\s+interview"
    r")\b"
)

CANDIDATE_AUDIENCE_RE = re.compile(
    r"(?i)\b("
    r"job\s+seekers?|candidates?|applicants?"
    r")\b"
)

CANDIDATE_FREE_RE = re.compile(
    r"(?i)\b("
    r"free\s+of\s+charge|completely\s+free|services?\s+are\s+free|"
    r"no\s+fee|no\s+charge|without\s+(?:a\s+)?(?:fee|charge)|"
    r"do\s+not\s+(?:have\s+to\s+)?pay|don'?t\s+(?:have\s+to\s+)?pay|"
    r"never\s+(?:have\s+to\s+)?pay|paid\s+entirely\s+by\s+(?:hiring\s+)?"
    r"(?:companies|employers|clients)|fees?\s+are\s+paid\s+(?:entirely\s+)?"
    r"by\s+(?:hiring\s+)?(?:companies|employers|clients)"
    r")\b"
)

SALARY_GUIDE_SOURCE_RE = re.compile(
    r"(?i)\b("
    r"salary\s+guide|compensation\s+guide|pay\s+guide|"
    r"salary\s+benchmark(?:ing)?(?:\s+report)?|benchmark(?:ing)?\s+report|"
    r"according\s+to\s+(?:the\s+)?(?:\d{4}\s+)?salary"
    r")\b"
)

OPEN_ROLE_SOURCE_RE = re.compile(
    r"(?i)\b("
    r"open\s+positions?|job\s+openings?|open\s+roles?|"
    r"(?:job|role|position)\s+in\s+[A-Z]|"
    r"posted\s+(?:job|role|opening)|currently\s+hiring"
    r")\b"
)

OPEN_POSITION_DOC_RE = re.compile(
    r"(?i)\b(open[_\s-]?positions?|job[_\s-]?listings?|job[_\s-]?openings?)\b"
)

SALARY_GUIDE_DOC_RE = re.compile(
    r"(?i)\b(salary[_\s-]?guide|compensation[_\s-]?guide|pay[_\s-]?"
    r"guide|salary[_\s-]?benchmark|benchmark(?:ing)?[_\s-]?report)\b"
)

JOB_LISTING_MARKERS_RE = re.compile(
    r"(?i)\b(job_id|posted_date|employment_type|remote_option|salary_range)\b"
)

OFFICE_PRESENCE_RE = re.compile(
    r"(?i)\b("
    r"office|offices|branch|branches|location|locations|"
    r"headquarters|head\s*office|hq|address|premises"
    r")\b"
)

CONTACT_ONLY_RE = re.compile(
    r"(?i)\b("
    r"email|e-?mail|phone|tel\.?|telephone|enquir(?:y|ies)|contact|"
    r"@|whatsapp|linkedin\.com"
    r")\b"
)

SENIORITY_RE = re.compile(
    r"(?i)\b(senior|junior|lead|principal|staff|associate|mid[- ]?level)\b"
)


def detect_source_intent(question: str) -> Optional[str]:
    """Named source type when the user explicitly scopes the answer."""
    q = question or ""
    if SALARY_GUIDE_SOURCE_RE.search(q):
        return "salary_guide"
    if re.search(
        r"(?i)\b(job\s+seeker|as\s+a\s+(?:job\s+)?seeker|as\s+a\s+candidate|"
        r"do\s+i\s+have\s+to\s+pay|have\s+to\s+pay\s+.+\s+anything|"
        r"pay\s+.+\s+anything\s+as\s+a)\b",
        q,
    ):
        return "candidate_fee"
    if re.search(
        r"(?i)\b("
        r"office\s+in|have\s+an?\s+office|offices?\s+in|"
        r"branch\s+in|located\s+in\s+[A-Z]|headquarters\s+in"
        r")\b",
        q,
    ):
        return "office_location"
    if OPEN_ROLE_SOURCE_RE.search(q) or re.search(
        r"(?i)\b(salary|pay)\s+range\s+for\s+(?:the\s+)?(?:.+?\s+)?"
        r"(?:job|role|position)\b",
        q,
    ):
        return "open_position"
    if re.search(
        r"(?i)\b("
        r"what\s+fee|charge\s+for|placement\s+fee|recruitment\s+fee|"
        r"how\s+much\s+(?:does|do|is).+\bfee|fee\s+schedule"
        r")\b",
        q,
    ):
        return "employer_fee"
    return None


def extract_requested_place(question: str) -> Optional[str]:
    """Place name from office/location questions (e.g. Tokyo)."""
    match = re.search(
        r"(?i)\b(?:office|branch|headquarters|hq|location|based)\s+in\s+"
        r"([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})\b",
        question or "",
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"(?i)\b(?:in|at)\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})\b\s*\??$",
        question or "",
    )
    if match:
        place = match.group(1).strip()
        if place.lower() not in {"the", "this", "that", "our", "your"}:
            return place
    return None


def extract_requested_role(question: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (seniority, role_family) for salary/role questions."""
    q = question or ""
    if detect_source_intent(q) in {"candidate_fee", "employer_fee", "office_location"}:
        return None, None
    seniority = None
    senior_match = SENIORITY_RE.search(q)
    if senior_match:
        seniority = senior_match.group(1).lower()

    role = None
    patterns = (
        r"(?i)(?:salary(?:\s+guide)?|earn|compensation|pay(?:\s+range)?)"
        r".{0,40}?\b(?:a|an|the)\s+((?:senior|junior|lead|principal|staff)\s+)?"
        r"([A-Za-z][A-Za-z0-9/+&. -]{2,60}?)(?:\s+earn|\s+make|\?|$)",
        r"(?i)\b(?:senior|junior|lead|principal|staff)\s+"
        r"([A-Za-z][A-Za-z0-9/+&. -]{2,50}?)"
        r"(?:\s+(?:earn|make|salary|in\b|\?|$))",
        r"(?i)\b(?:salary(?:\s+range)?|compensation|pay\s+range)\s+for\s+(?:the\s+)?"
        r"((?:senior|junior|lead|principal|staff)\s+)?"
        r"(.+?)\s+(?:job|role|position)",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        groups = [g for g in match.groups() if g]
        if not groups:
            continue
        blob = " ".join(g.strip() for g in groups).strip(" .?")
        blob = re.sub(r"(?i)^(a|an|the)\s+", "", blob).strip()
        sen = SENIORITY_RE.match(blob)
        if sen:
            seniority = seniority or sen.group(1).lower()
            role = SENIORITY_RE.sub("", blob, count=1).strip(" -")
        else:
            role = blob
        if role:
            role = re.sub(
                r"(?i)\s+(earn|make|in\s+\w+|job|role|position).*$",
                "",
                role,
            ).strip(" -")
            break
    return seniority, role or None


def content_has_fee_indicator(text: str) -> bool:
    return bool(FEE_INDICATOR_RE.search(text or ""))


def content_has_explicit_fee_amount(text: str) -> bool:
    return bool(FEE_AMOUNT_RE.search(text or ""))


def is_fee_process_noise_only(text: str) -> bool:
    """True when chunk is placement-process noise without an explicit fee amount."""
    body = text or ""
    if content_has_explicit_fee_amount(body):
        return False
    # Service headers are anchors for fee-block assembly, not noise.
    if re.search(r"(?i)^\s*service\s*:", body):
        return False
    if re.search(r"(?i)^\s*(?:fee|minimum\s+fee)\b", body):
        return False
    if FEE_PROCESS_NOISE_RE.search(body):
        return True
    # "permanent placements" prose without an amount is not a fee block.
    if re.search(r"(?i)\bpermanent\s+placements?\b", body) and not content_has_explicit_fee_amount(
        body
    ):
        return True
    return False


def is_employer_fee_block(text: str, *, service_phrases: Sequence[str] = ()) -> bool:
    """Explicit fee schedule row/block for a named employer service."""
    body = text or ""
    if is_fee_process_noise_only(body):
        return False
    if not content_has_explicit_fee_amount(body):
        return False
    if not content_has_fee_indicator(body):
        return False
    if service_phrases:
        blob = body.lower()
        if not any(p.lower() in blob for p in service_phrases if len(p) >= 4):
            # Allow pure amount rows that will be attached to a Service: header.
            if not re.search(r"(?i)^\s*(?:fee|minimum\s+fee)\b", body):
                return False
    # Prefer first-year salary context for percentage placement fees.
    if re.search(r"\d{1,3}(?:\.\d+)?\s*%", body) and re.search(
        r"(?i)\bpermanent\s+placement", " ".join(service_phrases)
    ):
        if not re.search(
            r"(?i)\b(first[- ]year|base\s+salary|minimum\s+fee)\b", body
        ) and not re.search(r"(?i)^\s*(?:fee|minimum\s+fee)\b", body):
            return False
    return True


def is_candidate_fee_evidence(text: str) -> bool:
    body = text or ""
    return bool(CANDIDATE_FREE_RE.search(body)) and (
        bool(CANDIDATE_AUDIENCE_RE.search(body))
        or bool(re.search(r"(?i)\b(job\s+seeker|candidate\s+faq|for\s+candidates)\b", body))
    )


def is_salary_guide_document(chunk: RetrievedChunk) -> bool:
    blob = " ".join(
        part
        for part in (
            chunk.document_name or "",
            chunk.section_title or "",
            (chunk.content or "")[:120],
        )
        if part
    )
    return bool(SALARY_GUIDE_DOC_RE.search(blob))


def is_open_position_document(chunk: RetrievedChunk) -> bool:
    blob = " ".join(
        part
        for part in (chunk.document_name or "", chunk.section_title or "")
        if part
    )
    if OPEN_POSITION_DOC_RE.search(blob):
        return True
    ctype = (chunk.content_type or "").lower()
    if ctype in {"structured_table_row", "table"} and JOB_LISTING_MARKERS_RE.search(
        chunk.content or ""
    ):
        return True
    return bool(JOB_LISTING_MARKERS_RE.search(chunk.content or ""))


def role_matches_evidence(
    text: str,
    *,
    seniority: Optional[str],
    role: Optional[str],
) -> bool:
    content = (text or "").lower()
    if role:
        tokens = [
            t
            for t in re.findall(r"[a-z0-9]+", role.lower())
            if len(t) > 2 and t not in {"the", "and", "for", "job", "role"}
        ]
        if tokens and not all(t in content for t in tokens):
            return False
    if seniority:
        if not re.search(rf"(?i)\b{re.escape(seniority)}\b", content):
            return False
        # Reject when a conflicting seniority is the only level present for the role.
        if seniority == "senior" and re.search(
            r"(?i)\bjunior\s+" + re.escape(role or "devops"), content
        ):
            return False
    return True


def is_office_location_evidence(text: str, place: Optional[str]) -> bool:
    body = text or ""
    if not place:
        return False
    if not re.search(rf"(?i)\b{re.escape(place)}\b", body):
        return False
    if not OFFICE_PRESENCE_RE.search(body):
        return False
    # Contact-only rows that happen to mention a city in an email signature
    # are not office-presence evidence unless they also assert an office list.
    if CONTACT_ONLY_RE.search(body) and not re.search(
        r"(?i)\b("
        r"offices?\s+in|other\s+offices|headquarters|branch|"
        r"office\s+(?:at|in|located)|have\s+an?\s+office"
        r")\b",
        body,
    ):
        # Allow "offices in Singapore, London…" style even with nearby contact.
        if not re.search(
            rf"(?i)\b(?:offices?|branches|headquarters).{{0,40}}\b{re.escape(place)}\b|"
            rf"\b{re.escape(place)}\b.{{0,40}}\b(?:office|branch|headquarters)\b",
            body,
        ):
            return False
    return True


def is_contact_details_only(text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return False
    if OFFICE_PRESENCE_RE.search(body) and re.search(
        r"(?i)\b(offices?\s+in|other\s+offices|headquarters|branch)\b", body
    ):
        return False
    return bool(CONTACT_ONLY_RE.search(body)) and not bool(
        re.search(
            r"(?i)\b(office|branch|headquarters|located\s+in|address\s*:)\b",
            body,
        )
    )


def filter_evidence_for_answer_grounding(
    evidence: Sequence[RetrievedChunk],
    question: str,
    *,
    fact_type: str = "",
) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    """Drop evidence that fails source-intent / answer-type checks."""
    intent = detect_source_intent(question)
    seniority, role = extract_requested_role(question)
    place = extract_requested_place(question)
    diagnostics: Dict[str, Any] = {
        "source_intent": intent,
        "requested_seniority": seniority,
        "requested_role": role,
        "requested_place": place,
        "kept": [],
        "rejected": [],
        "rejection_reason": None,
        "answer_type_ok": True,
        "entity_match_ok": True,
        "source_type_ok": True,
    }

    if not evidence:
        diagnostics["rejection_reason"] = "no_evidence"
        return [], diagnostics

    from app.retrieval.entity_validation import (
        extract_requested_service_phrases,
        service_phrases_for_matching,
    )

    phrases = service_phrases_for_matching(extract_requested_service_phrases(question))
    kept: List[RetrievedChunk] = []

    for chunk in evidence:
        content = chunk.content or ""
        blob = "\n".join(
            part
            for part in (content, chunk.section_title or "", chunk.document_name or "")
            if part
        )
        reason = None

        if intent == "employer_fee" or (
            fact_type == "price"
            and not intent
            and re.search(
                r"(?i)\b("
                r"employer|client|recruit(?:ment|ing)?|placement|staffing|"
                r"executive\s+search|permanent\s+placement"
                r")\b",
                question or "",
            )
            and re.search(r"(?i)\b(fee|charge\s+for|cost|price)\b", question or "")
        ):
            if is_fee_process_noise_only(blob):
                reason = "fee_process_noise_without_amount"
            elif not (
                is_employer_fee_block(blob, service_phrases=phrases)
                or (
                    re.search(r"(?i)^\s*service\s*:", content)
                    and any(p.lower() in content.lower() for p in phrases)
                )
                or (
                    re.search(r"(?i)^\s*(?:fee|minimum\s+fee)\b", content)
                    and content_has_explicit_fee_amount(content)
                )
            ):
                # Keep service header rows for sibling assembly; drop other non-fee.
                if not (
                    re.search(r"(?i)^\s*service\s*:", content)
                    and any(p.lower() in content.lower() for p in phrases)
                ):
                    reason = "missing_explicit_fee_block"

        elif intent == "candidate_fee":
            if not is_candidate_fee_evidence(blob):
                reason = "missing_candidate_free_of_charge_evidence"

        elif intent == "salary_guide":
            if is_open_position_document(chunk) and not is_salary_guide_document(chunk):
                reason = "open_position_not_salary_guide"
            elif not role_matches_evidence(
                blob, seniority=seniority, role=role
            ):
                reason = "role_or_seniority_mismatch"
            elif not (
                is_salary_guide_document(chunk)
                or SALARY_GUIDE_SOURCE_RE.search(blob)
                or (
                    content_has_explicit_fee_amount(blob)
                    and role_matches_evidence(blob, seniority=seniority, role=role)
                    and not is_open_position_document(chunk)
                )
            ):
                # Without a salary-guide doc, only accept non-listing rows that
                # explicitly match senior+role — still prefer not-found later.
                if is_open_position_document(chunk):
                    reason = "open_position_not_salary_guide"
                else:
                    reason = "not_salary_guide_evidence"

        elif intent == "open_position" or (
            fact_type == "price"
            and seniority is None
            and role
            and re.search(r"(?i)\b(job|role|position)\b", question or "")
        ):
            if not role_matches_evidence(blob, seniority=seniority, role=role):
                reason = "role_or_seniority_mismatch"

        elif intent == "office_location" or (
            fact_type == "location"
            and re.search(r"(?i)\boffice\b", question or "")
        ):
            if is_contact_details_only(blob):
                reason = "contact_details_not_office_location"
            elif not is_office_location_evidence(blob, place):
                reason = "missing_explicit_office_location"

        if reason:
            diagnostics["rejected"].append(
                {
                    "document_name": chunk.document_name,
                    "preview": content[:160],
                    "rejection_reason": reason,
                    "source_type": _infer_source_type_label(chunk),
                }
            )
            logger.info(
                "GROUNDING_REJECT doc=%s reason=%s preview=%r",
                chunk.document_name,
                reason,
                content[:120],
            )
            continue

        kept.append(chunk)
        diagnostics["kept"].append(
            {
                "document_name": chunk.document_name,
                "preview": content[:160],
                "source_type": _infer_source_type_label(chunk),
            }
        )

    if not kept:
        diagnostics["rejection_reason"] = (
            diagnostics["rejected"][0]["rejection_reason"]
            if diagnostics["rejected"]
            else "no_grounded_evidence"
        )
        diagnostics["answer_type_ok"] = False
        if any(
            r["rejection_reason"]
            in {
                "open_position_not_salary_guide",
                "not_salary_guide_evidence",
                "fee_process_noise_without_amount",
                "missing_explicit_fee_block",
                "contact_details_not_office_location",
                "missing_explicit_office_location",
                "missing_candidate_free_of_charge_evidence",
            }
            for r in diagnostics["rejected"]
        ):
            diagnostics["source_type_ok"] = False
        if any(
            r["rejection_reason"] == "role_or_seniority_mismatch"
            for r in diagnostics["rejected"]
        ):
            diagnostics["entity_match_ok"] = False

    # Salary-guide questions: if nothing survived from a guide doc, fail closed
    # even when open-position rows were the only candidates.
    if intent == "salary_guide":
        guide_kept = [c for c in kept if is_salary_guide_document(c)]
        if not guide_kept:
            for chunk in kept:
                diagnostics["rejected"].append(
                    {
                        "document_name": chunk.document_name,
                        "preview": (chunk.content or "")[:160],
                        "rejection_reason": "salary_guide_required_but_absent",
                        "source_type": _infer_source_type_label(chunk),
                    }
                )
            diagnostics["kept"] = []
            diagnostics["rejection_reason"] = "salary_guide_required_but_absent"
            diagnostics["source_type_ok"] = False
            diagnostics["answer_type_ok"] = False
            return [], diagnostics

    return kept, diagnostics


def _infer_source_type_label(chunk: RetrievedChunk) -> str:
    if is_salary_guide_document(chunk):
        return "salary_guide"
    if is_open_position_document(chunk):
        return "open_position"
    name = (chunk.document_name or "").lower()
    if "fee" in name or "service" in name:
        return "fee_schedule"
    if "faq" in name or "candidate" in name:
        return "candidate_faq"
    if "office" in name or "overview" in name or "location" in name:
        return "company_profile"
    return (chunk.content_type or "document") or "document"


def format_office_not_found_answer(
    place: Optional[str],
    *,
    contact_name: Optional[str] = None,
) -> str:
    """Grounded not-found for yes/no office-presence questions."""
    city = (place or "that city").strip()
    org = (contact_name or "").strip()
    if org and org.lower() not in {"the city", "city", "the organization"}:
        return (
            f"I could not find confirmation of a {city} office in the available "
            f"documents. You may want to contact {org} directly."
        )
    return (
        f"I could not find confirmation of a {city} office in the available "
        "documents. You may want to contact the organization directly."
    )


def synthesize_candidate_fee_answer(
    evidence: Sequence[RetrievedChunk],
) -> Optional[str]:
    """Deterministic free-for-candidates answer when evidence is explicit."""
    for chunk in evidence:
        text = chunk.content or ""
        if not is_candidate_fee_evidence(text):
            continue
        if re.search(r"(?i)\bno\b.{0,80}\bfree\b", text) or CANDIDATE_FREE_RE.search(
            text
        ):
            return (
                "No. Services are free for job seekers / candidates; "
                "fees are paid by hiring companies."
            )
    return None


def synthesize_office_presence_answer(
    evidence: Sequence[RetrievedChunk],
    *,
    place: Optional[str],
    question: str = "",
) -> Optional[str]:
    if not place:
        return None
    positive = [
        chunk
        for chunk in evidence
        if is_office_location_evidence(chunk.content or "", place)
    ]
    if not positive:
        return None
    # Yes/no questions.
    if re.search(r"(?i)\b(does|do|is|are|have|has)\b", question or ""):
        return f"Yes. Documents confirm an office presence in {place}."
    # Otherwise return the grounded snippet.
    text = re.sub(r"\s+", " ", (positive[0].content or "")).strip()
    return text[:320]
