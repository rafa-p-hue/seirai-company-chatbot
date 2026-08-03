"""Deterministic validation/composition for exhaustive checklist and fee answers."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import RetrievedChunk
from app.retrieval.hybrid_search import query_term_scores, tokenize
from app.retrieval.numeric_facts import CURRENCY_RE

LIST_MARKER_RE = re.compile(r"^\s*(?:[-•●▪◦·]|\d+[.)])\s+")
INLINE_BULLET_SPLIT_RE = re.compile(r"\s*(?:[-•●▪◦·]|\d+[.)])\s+")
CHECKLIST_CUE_RE = re.compile(
    r"(?i)\b("
    r"what (?:do|should|must) (?:i|we) bring|what is required|"
    r"what (?:documents?|items?|materials?) (?:do|should|must) (?:i|we) need|"
    r"what (?:do|should|must) (?:i|we) need(?:\s+to\s+(?:bring|submit|provide)|\s+for\b)|"
    r"which documents?|documents? (?:are )?(?:needed|required)|"
    r"required (?:items?|documents?|materials?)|"
    r"need to (?:bring|submit|provide)|"
    r"need for (?:the\s+)?(?:move-?in|registration|notification)"
    r")\b"
)
REQUIRED_CLAUSE_RE = re.compile(
    r"(?i)\b("
    r"must (?:bring|submit|provide|include)|"
    r"required (?:documents?|items?|materials?)|"
    r"bring(?: the following| your| a| an| the)?|"
    r"provide(?: the following| your| a| an| the)?|"
    r"submit(?: the following| your| a| an| the)?|"
    r"documents? (?:needed|required)|"
    r"(?:is|are) required|"
    r"you (?:will )?need"
    r")\b"
)
REQUIRED_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"bring(?: the following| your| a| an| the)?|"
    r"required (?:documents?|items?|materials?)|"
    r"must (?:bring|provide|submit|include|complete)|"
    r"(?:documents?|items?|materials?) (?:are )?(?:needed|required)|"
    r"(?:is|are) required|"
    r"need to (?:bring|provide|submit)|submit the following|"
    r"provide(?: the following| your| a| an| the)?"
    r")\b"
)
BRING_PROSE_RE = re.compile(
    r"(?i)^(?:please\s+)?(?:bring|provide|submit)\s+(?:your|the|a|an)\s+(.+)$"
)
IS_REQUIRED_PROSE_RE = re.compile(
    r"(?i)^(.+?)\s+(?:is|are)\s+required(?:\s+(.+))?$"
)
INTRO_RE = re.compile(
    r"(?i)^(?:required (?:documents?|items?|materials?)|"
    r"(?:please )?bring(?: the following)?|you (?:will )?need|"
    r"documents? required)\s*:?\s*$"
)
DEADLINE_ONLY_RE = re.compile(
    r"(?i)\b(within\s+\d+\s+(?:business\s+)?(?:days?|weeks?|months?)|"
    r"must\s+register|service window)\b"
)
DOCUMENT_CUE_RE = re.compile(
    r"(?i)\b("
    r"passports?|cards?|forms?|documents?|identification|id\b|bills?|consent|"
    r"licenses?|certificates?|photos?|proof|agreements?|applications?|"
    r"bring|submit|provide|complete|required items?"
    r")\b"
)
DEADLINE_LOCATION_CUE_RE = re.compile(
    r"(?i)\b("
    r"within\s+\d+\s+(?:business\s+)?(?:days?|weeks?|months?)|"
    r"must\s+register|service window|window\s+\d+|"
    r"citizen affairs|file a (?:move-?in )?notification"
    r")\b"
)


def strip_checklist_deadline_padding(answer: str) -> str:
    """Drop deadline/location-only sentences from checklist answers.

    Missing required items must not be replaced by repeating when/where to
    register. Document-bearing sentences are kept even if they also mention a
    window or deadline.
    """
    text = (answer or "").strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+|(?=\n\s*[-•])|\n+", text)
    kept: List[str] = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        has_doc = bool(DOCUMENT_OBJECT_RE.search(cleaned))
        has_deadline = bool(DEADLINE_LOCATION_CUE_RE.search(cleaned))
        if has_deadline and not has_doc:
            continue
        # Procedure sentences that slipped into checklist bullets.
        if PROCEDURE_ACTION_RE.search(cleaned) and not has_doc:
            continue
        # Long heading+deadline dumps without a concrete document name.
        if (
            re.search(r"(?i)\b(moving in|tennyu|move-?in notification)\b", cleaned)
            and has_deadline
            and not has_doc
        ):
            continue
        kept.append(cleaned)
    if not kept:
        return text
    if any(re.match(r"^[-•]", item) for item in kept) or "\n" in text:
        # Preserve bullet structure; keep intro lines intact.
        return "\n".join(kept)
    return " ".join(kept)


def missing_required_item_phrases(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> List[str]:
    """Human-readable required items missing from an answer."""
    from app.generation.answer_synthesis import (
        format_checklist_item_phrase,
        missing_checklist_items,
    )

    return [
        format_checklist_item_phrase(entry)
        for entry in missing_checklist_items(answer, evidence)
    ]


def is_checklist_question(question: str) -> bool:
    return bool(CHECKLIST_CUE_RE.search(question or ""))


def extract_required_items(
    evidence: Sequence[RetrievedChunk],
) -> List[Tuple[int, str]]:
    """Extract complete, explicit requirements while preserving conditions."""
    items: List[Tuple[int, str]] = []
    seen = set()
    for evidence_index, chunk in enumerate(evidence, start=1):
        text = (chunk.content or "").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Flatten inline bullet runs produced by PDF extractors.
        expanded: List[str] = []
        for line in lines:
            # Prefer fused bring/required prose expansion before inline numbered
            # splits so phrases like "Window 3. Bring your..." are not broken.
            prose_lines = _expand_prose_requirement_lines(line)
            if prose_lines:
                expanded.extend(prose_lines)
                continue
            if INLINE_BULLET_SPLIT_RE.search(line) and not LIST_MARKER_RE.match(line):
                parts = [
                    part.strip(" :")
                    for part in INLINE_BULLET_SPLIT_RE.split(line)
                    if part and part.strip(" :")
                ]
                if len(parts) >= 2:
                    for part in parts:
                        if INTRO_RE.match(part):
                            continue
                        expanded.append(f"• {part}")
                    continue
            expanded.append(line)
        lines = expanded
        is_list = chunk.content_type in {"list", "numbered_list"}
        context = "\n".join(
            value
            for value in (
                chunk.section_title or "",
                chunk.subsection_title or "",
                text,
            )
            if value
        )
        has_requirement_context = bool(REQUIRED_CONTEXT_RE.search(context))
        if is_list and not has_requirement_context:
            # Same-document sibling bullets often omit "bring/required" wording
            # after an earlier checklist intro was already selected. If ANY
            # evidence chunk for this pass already carries requirement context
            # in the same document, keep these list items.
            sibling_context = False
            doc_name = (chunk.document_name or "").strip().lower()
            for other in evidence:
                if (other.document_name or "").strip().lower() != doc_name:
                    continue
                other_blob = "\n".join(
                    value
                    for value in (
                        other.section_title or "",
                        other.subsection_title or "",
                        other.content or "",
                    )
                    if value
                )
                if REQUIRED_CONTEXT_RE.search(other_blob):
                    sibling_context = True
                    break
            if not sibling_context:
                # A bullet list is not inherently a checklist. Hours, closures,
                # contacts, and unrelated lists must not become required items.
                continue
            has_requirement_context = True
        bullet_items_present = any(LIST_MARKER_RE.match(line) for line in lines)
        for line in lines:
            marked = bool(LIST_MARKER_RE.match(line))
            cleaned = LIST_MARKER_RE.sub("", line).strip()
            if not cleaned or INTRO_RE.match(cleaned):
                continue
            if DEADLINE_ONLY_RE.search(cleaned) and not re.search(
                r"(?i)\b(passport|card|form|document|identification|bill|consent|"
                r"bring|provide|submit|agreement)\b",
                cleaned,
            ):
                continue
            # When an explicit bullet checklist exists under "Bring", skip
            # surrounding procedure-action sentences ("Submit a notification...").
            if (
                bullet_items_present
                and not marked
                and (
                    re.match(
                        r"(?i)^(?:submit|file|complete|register|apply|enroll)\b",
                        cleaned,
                    )
                    or PROCEDURE_ACTION_RE.search(cleaned)
                    or (
                        DEADLINE_LOCATION_CUE_RE.search(cleaned)
                        and not DOCUMENT_OBJECT_RE.search(cleaned)
                    )
                )
            ):
                continue
            if is_list or marked:
                candidate = cleaned
            elif REQUIRED_CLAUSE_RE.search(cleaned) and len(cleaned.split()) <= 40:
                candidate = cleaned
            else:
                continue
            if not _is_valid_required_item(candidate):
                continue
            normalized = re.sub(r"\W+", " ", candidate.lower()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                items.append((evidence_index, candidate))
    return items


def _expand_prose_requirement_lines(line: str) -> List[str]:
    """Split fused bring/required sentences into item lines when present."""
    text = (line or "").strip()
    if not text or LIST_MARKER_RE.match(text):
        return []
    if not REQUIRED_CONTEXT_RE.search(text):
        return []
    # Keep true bullet/list intros untouched.
    if INTRO_RE.match(text.rstrip(":")):
        return []
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part and part.strip()
    ]
    if len(sentences) <= 1 and not BRING_PROSE_RE.match(text.rstrip(".!?")):
        # Single short clause handled by the normal REQUIRED_CLAUSE path.
        if len(text.split()) <= 40:
            return []
    expanded: List[str] = []
    for sentence in sentences:
        cleaned = sentence.strip().rstrip(".!?")
        if not cleaned:
            continue
        if DEADLINE_ONLY_RE.search(cleaned) and not re.search(
            r"(?i)\b(passport|card|form|document|identification|bill|consent|"
            r"bring|provide|submit|agreement)\b",
            cleaned,
        ):
            continue
        bring = BRING_PROSE_RE.match(cleaned)
        if bring:
            for part in _split_requirement_list(bring.group(1)):
                expanded.append(f"• {part}")
            continue
        required = IS_REQUIRED_PROSE_RE.match(cleaned)
        if required:
            subject = required.group(1).strip()
            trailing = (required.group(2) or "").strip()
            item = f"{subject} {trailing}".strip() if trailing else subject
            # Skip deadline-only "registration is required within..."
            if DEADLINE_ONLY_RE.search(item) and not re.search(
                r"(?i)\b(passport|card|form|document|identification|bill|consent|"
                r"agreement)\b",
                item,
            ):
                continue
            expanded.append(f"• {item}")
            continue
    return expanded


def _split_requirement_list(body: str) -> List[str]:
    """Split comma/and lists while keeping 'X or Y' / 'X, or Y' alternatives intact."""
    text = re.sub(r"\s+", " ", (body or "").strip())
    if not text:
        return []
    # Do not split on the comma in "A, or B" — that is one alternative pair.
    # Lookahead must allow intervening whitespace (`, or` vs `,or`).
    parts = re.split(r",\s+and\s+|,\s+(?!\s*or\b)|(?<!,)\s+and\s+", text)
    items: List[str] = []
    for part in parts:
        cleaned = part.strip(" .;")
        if cleaned:
            items.append(cleaned)
    return items


DOCUMENT_OBJECT_RE = re.compile(
    r"(?i)\b("
    r"passports?|cards?|forms?|documents?|identification|id\b|bills?|consent|"
    r"licenses?|certificates?|photos?|proof|agreements?|applications?|"
    r"seal|my\s+number|notification\s+cards?"
    r")\b"
)
PROCEDURE_ACTION_RE = re.compile(
    r"(?i)\b("
    r"submit a (?:move-?in )?notification|file a (?:move-?in )?notification|"
    r"must (?:register|enroll|apply)|register within|enroll within|"
    r"beginning to live|new address"
    r")\b"
)


def isolate_checklist_passage(text: str) -> str:
    """Prefer the required-items portion of a fused procedure passage."""
    body = (text or "").strip()
    if not body:
        return body
    match = re.search(
        r"(?im)^(?:required (?:documents?|items?|materials?)\s*:?|"
        r"(?:please )?bring(?: the following)?\s*:?|"
        r"you (?:will )?need\s*:?|"
        r"documents? required\s*:?)\s*$",
        body,
    )
    if match:
        return body[match.start() :].strip()
    bring = re.search(
        r"(?i)\b(?:bring(?: the following)?|required (?:documents?|items?)|"
        r"documents? (?:needed|required))\b",
        body,
    )
    if bring and bring.start() > 0:
        return body[bring.start() :].strip()
    return body


def _is_valid_required_item(item: str) -> bool:
    """Reject list noise while preserving objects, alternatives, and conditions."""
    text = re.sub(r"\s+", " ", (item or "")).strip()
    if not text or not re.search(r"[A-Za-z]", text):
        return False
    if len(text.split()) > 45:
        return False
    if re.search(
        r"(?i)^(?:office|opening|business)?\s*hours?\b|"
        r"^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.*"
        r"(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|closed)\b|"
        r"^(?:closed|closure|page\s+\d+|contact information)\b|"
        r"^(?:guide|handbook|manual|brochure)\b",
        text,
    ):
        return False
    if re.fullmatch(
        r"(?i)[A-Z][A-Za-z0-9 /&-]{2,60}",
        text,
    ) and not DOCUMENT_OBJECT_RE.search(text):
        # Bare headings / titles are not checklist items.
        return False
    if re.fullmatch(
        r"(?i)(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)"
        r"(?:\s*[-–—]\s*\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))?)",
        text,
    ):
        return False
    if re.fullmatch(
        r"(?i)(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4}",
        text,
    ):
        return False
    if re.match(
        r"^\d{1,6}\s+\S+(?:\s+\S+){0,5}\s+"
        r"(?:st(?:reet)?|ave(?:nue)?|rd|road|blvd|drive|lane)\b",
        text,
        re.I,
    ):
        return False
    # Procedure / deadline sentences are not required documents.
    if PROCEDURE_ACTION_RE.search(text) and not DOCUMENT_OBJECT_RE.search(text):
        return False
    if DEADLINE_LOCATION_CUE_RE.search(text) and not DOCUMENT_OBJECT_RE.search(text):
        return False
    # Require an actual document/credential object — bare verbs like "submit"
    # must not turn registration instructions into checklist items.
    if not DOCUMENT_OBJECT_RE.search(text):
        return False
    return True


def checklist_answer(
    evidence: Sequence[RetrievedChunk],
) -> str:
    """Synthesize a natural-language checklist — never dump retrieval passages."""
    from app.generation.answer_synthesis import synthesize_checklist_answer

    return synthesize_checklist_answer(evidence)


def checklist_answer_is_complete(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> bool:
    from app.generation.answer_synthesis import (
        checklist_structure_is_covered,
        extract_checklist_structure,
        is_retrieval_dump_answer,
    )
    from app.generation.evidence_presentation import (
        answer_exposes_internal_field_keys,
        looks_like_answer_fragment,
    )

    if not answer or answer == FALLBACK_ANSWER:
        return False
    answer = strip_checklist_deadline_padding(answer)
    if not answer or answer == FALLBACK_ANSWER:
        return False
    if (
        answer_exposes_internal_field_keys(answer)
        or looks_like_answer_fragment(answer)
        or is_retrieval_dump_answer(answer)
    ):
        return False
    structure = extract_checklist_structure(evidence)
    if structure and checklist_structure_is_covered(answer, structure):
        return True
    # Legacy token coverage fallback for non-structured bullet answers.
    items = extract_required_items(evidence)
    if not items:
        return False
    answer_tokens = tokenize(answer)
    answer_l = (answer or "").lower()
    for _, item in items:
        distinctive = {
            token
            for token in tokenize(item)
            if token
            not in {
                "required",
                "bring",
                "provide",
                "submit",
                "document",
                "documents",
                "following",
                "your",
                "the",
                "and",
                "or",
                "for",
                "all",
            }
            and len(token) > 2
        }
        if not distinctive:
            continue
        if distinctive.issubset(answer_tokens):
            continue
        hits = sum(1 for token in distinctive if token in answer_tokens)
        if hits >= max(2, (len(distinctive) + 1) // 2) and any(
            cue in answer_l
            for cue in (
                "card",
                "passport",
                "certificate",
                "form",
                "number",
                "id",
                "permit",
            )
        ):
            continue
        compact_item = re.sub(r"\W+", " ", item.lower()).strip()
        compact_answer = re.sub(r"\W+", " ", answer_l).strip()
        words = compact_item.split()
        if len(words) >= 2:
            for size in (3, 2):
                for i in range(0, max(0, len(words) - size + 1)):
                    phrase = " ".join(words[i : i + size])
                    if phrase in compact_answer and phrase not in {
                        "for all",
                        "or passport",
                    }:
                        break
                else:
                    continue
                break
            else:
                return False
            continue
        return False
    return True


def relevant_price_evidence(
    question: str, evidence: Sequence[RetrievedChunk]
) -> List[Tuple[int, RetrievedChunk]]:
    from app.ingestion.document_status import question_wants_historical
    from app.generation.evidence_presentation import extract_currency_amounts
    from app.retrieval.numeric_facts import content_has_currency, PERCENT_RE

    salary_question = bool(
        re.search(r"(?i)\b(salary|salaries|compensation|pay\s+range)\b", question or "")
    )

    currency_chunks = [
        (index, chunk)
        for index, chunk in enumerate(evidence, start=1)
        if content_has_currency(chunk.content or "")
        or extract_currency_amounts(chunk.content or "")
        or PERCENT_RE.search(chunk.content or "")
        or (
            salary_question
            and re.search(
                r"(?i)\b(salary_range|salary|sgd|usd|gbp|aud|compensation)\b",
                chunk.content or "",
            )
        )
    ]
    if not currency_chunks:
        return []
    historical = question_wants_historical(question)
    requested_age = _extract_requested_age(question)

    # For salary/job questions, require strong title + location match when asked.
    requested_title = ""
    requested_location = ""
    salary_role = re.search(
        r"(?i)\b(?:salary(?:\s+range)?|compensation|pay\s+range)\s+for\s+(?:the\s+)?"
        r"(.+?)\s+(?:job|role|position)(?:\s+in\s+([A-Za-z][A-Za-z .'-]{1,40}))?",
        question or "",
    )
    if salary_role:
        requested_title = (salary_role.group(1) or "").strip(" .")
        requested_location = (salary_role.group(2) or "").strip(" .?")
    if salary_question and not requested_location:
        loc_only = re.search(
            r"(?i)\b(?:job|role|position)\s+in\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
            question or "",
        )
        if loc_only:
            requested_location = loc_only.group(1).strip()
    if salary_question and not requested_title:
        # Fallback: capture "DevOps Engineer" style phrases before "in <place>".
        loose = re.search(
            r"(?i)\b(?:the\s+)?([A-Z][A-Za-z0-9/+&. -]{2,60}?)\s+"
            r"(?:job|role|position)\s+in\s+",
            question or "",
        )
        if loose:
            requested_title = loose.group(1).strip()

    scored: List[Tuple[float, int, RetrievedChunk]] = []
    for index, chunk in currency_chunks:
        content = chunk.content or ""
        term_scores = query_term_scores(question, content)
        score = max(term_scores.values(), default=0.0)
        if chunk.content_type in {"table", "structured_table_row"}:
            score += 0.2
        status = (chunk.document_status or "unknown").lower()
        if historical:
            if status == "archived":
                score += 0.45
        else:
            if status == "current":
                score += 0.55
            elif status == "archived":
                score -= 0.8
        if requested_age is not None:
            age_score = _age_bracket_match_score(content, requested_age)
            score += age_score
        if salary_question:
            content_l = content.lower()
            if requested_title:
                title_tokens = [
                    t
                    for t in re.findall(r"[a-z0-9]+", requested_title.lower())
                    if len(t) > 2
                ]
                if title_tokens and all(t in content_l for t in title_tokens):
                    score += 1.2
                elif title_tokens:
                    score -= 1.5
            if requested_location:
                loc = requested_location.lower()
                if loc in content_l:
                    score += 0.8
                else:
                    score -= 1.0
        scored.append((score, index, chunk))
    best = max(score for score, _, _ in scored)
    selected = [
        (index, chunk)
        for score, index, chunk in scored
        if score > 0 and score >= best - 0.35
    ]
    if salary_question and requested_title:
        tight = []
        title_tokens = [
            t for t in re.findall(r"[a-z0-9]+", requested_title.lower()) if len(t) > 2
        ]
        for index, chunk in selected:
            content_l = (chunk.content or "").lower()
            if title_tokens and all(t in content_l for t in title_tokens):
                if not requested_location or requested_location.lower() in content_l:
                    tight.append((index, chunk))
        if tight:
            selected = tight
    if not historical:
        current_only = [
            item
            for item in selected
            if (item[1].document_status or "unknown").lower() != "archived"
        ]
        if current_only:
            selected = current_only
    if requested_age is not None:
        age_matched = [
            item
            for item in selected
            if _age_bracket_match_score(item[1].content or "", requested_age) >= 0.8
        ]
        if age_matched:
            selected = age_matched
    # Never fall back to an unrelated top currency row when nothing matched the
    # requested service terms — that path produced wrong Koseki/certificate fees.
    if not selected:
        return []
    # Extra entity gate: when the question names a service, drop currency-only
    # mismatches even if term scoring was noisy.
    from app.retrieval.entity_validation import (
        collect_named_service_fee_evidence,
        evidence_matches_requested_entity,
        extract_requested_service_phrases,
        service_phrases_for_matching,
    )

    if service_phrases_for_matching(extract_requested_service_phrases(question)):
        entity_ok = [
            item
            for item in selected
            if evidence_matches_requested_entity(
                item[1].content or "",
                question,
                section_title=item[1].section_title or "",
                document_name=item[1].document_name or "",
            )
        ]
        if entity_ok:
            return entity_ok
        # Fee schedules often split "Service: X" from "Fee: N%" across chunks.
        block = collect_named_service_fee_evidence(evidence, question)
        if not block:
            return []
        block_keys = {
            (c.document_name, (c.content or "")[:160]) for c in block
        }
        sibling_ok = [
            item
            for item in selected
            if (item[1].document_name, (item[1].content or "")[:160]) in block_keys
        ]
        if sibling_ok:
            return sibling_ok
        # Rebuild from the recovered block when selected rows lacked entity text.
        rebuilt: List[Tuple[int, RetrievedChunk]] = []
        for chunk in block:
            if not (
                content_has_currency(chunk.content or "")
                or extract_currency_amounts(chunk.content or "")
                or PERCENT_RE.search(chunk.content or "")
            ):
                continue
            # Preserve original 1-based indexes when possible.
            try:
                index = next(
                    i
                    for i, c in enumerate(evidence, start=1)
                    if c is chunk
                    or (
                        c.document_name == chunk.document_name
                        and (c.content or "")[:160] == (chunk.content or "")[:160]
                    )
                )
            except StopIteration:
                index = len(rebuilt) + 1
            rebuilt.append((index, chunk))
        return rebuilt
    return selected


def _extract_requested_age(question: str) -> Optional[int]:
    match = re.search(
        r"(?i)\b(\d{1,2})\s*[- ]?\s*year(?:s)?(?:\s*|-)?old\b|"
        r"\b(\d{1,2})\s*yo\b|"
        r"\bage(?:d)?\s+(\d{1,2})\b",
        question or "",
    )
    if not match:
        return None
    raw = match.group(1) or match.group(2) or match.group(3)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _age_bracket_match_score(content: str, age: int) -> float:
    """Score how well content's age ranges cover the requested age."""
    text = content or ""
    best = 0.0
    # Explicit "2-year-old" / "for a 2-year-old"
    if re.search(rf"(?i)\b{age}\s*[- ]?\s*year(?:s)?(?:\s*|-)?old\b", text):
        best = max(best, 1.2)
    # Ranges like "ages 3 to 12" / "3-12"
    for match in re.finditer(
        r"(?i)\b(?:aged?\s+)?(\d{1,2})\s*(?:to|-|–|—)\s*(\d{1,2})\b",
        text,
    ):
        low, high = int(match.group(1)), int(match.group(2))
        if low <= age <= high:
            best = max(best, 1.0)
    return best


def price_answer(question: str, evidence: Sequence[RetrievedChunk]) -> str:
    """Synthesize natural-language fee facts — never expose CSV field keys."""
    from app.generation.answer_synthesis import synthesize_fee_answer

    return synthesize_fee_answer(question, evidence)


def price_answer_is_complete(
    answer: str, question: str, evidence: Sequence[RetrievedChunk]
) -> bool:
    from app.generation.answer_synthesis import is_retrieval_dump_answer
    from app.generation.evidence_presentation import (
        answer_exposes_internal_field_keys,
        extract_currency_amounts,
        looks_like_answer_fragment,
    )

    if not answer or answer == FALLBACK_ANSWER:
        return False
    if (
        answer_exposes_internal_field_keys(answer or "")
        or is_retrieval_dump_answer(answer)
        or looks_like_answer_fragment(answer)
    ):
        return False
    if re.search(r"(?i)^applicable fees:\s*", answer or ""):
        # Legacy dump header — only OK if body is already clean prose without keys.
        if answer_exposes_internal_field_keys(answer):
            return False
    required = relevant_price_evidence(question, evidence)
    if not required:
        return False
    expected: set[str] = set()
    pct_expected: set[str] = set()
    for _, chunk in required:
        text = chunk.content or ""
        amounts = extract_currency_amounts(text)
        expected |= amounts
        if re.search(r"\d{1,3}(?:\.\d+)?\s*%", text) and not re.search(
            r"(?i)minimum\s+fee", text
        ):
            pct_expected |= {
                a for a in amounts if re.search(rf"(?i)\b{re.escape(a)}\s*%", text)
            } or amounts
    actual = extract_currency_amounts(answer or "")
    # Percentage placement fees: require the % fee in the answer; minimum fee
    # amounts are optional companions, not completeness blockers.
    if pct_expected and re.search(r"\d{1,3}(?:\.\d+)?\s*%", answer or ""):
        return bool(pct_expected & actual) or pct_expected.issubset(actual)
    if re.search(r"(?i)\bpermanent\s+placement\b", question or "") and not re.search(
        r"\d{1,3}(?:\.\d+)?\s*%", answer or ""
    ):
        return False
    return bool(expected) and expected.issubset(actual)
