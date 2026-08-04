"""Natural-language answer synthesis from normalized evidence.

Never returns raw retrieval dumps, snake_case field names, or mid-sentence
fragments as the user-facing answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.generation.evidence_presentation import (
    answer_exposes_internal_field_keys,
    format_structured_content_as_prose,
    looks_like_answer_fragment,
    looks_like_structured_record,
    prepare_evidence_for_generation,
)
from app.generation.evidence_validation import (
    _age_bracket_match_score,
    _extract_requested_age,
    extract_required_items,
    relevant_price_evidence,
)
from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import RetrievedChunk

logger = logging.getLogger(__name__)


DOCUMENT_OBJECT_HINT = re.compile(
    r"(?i)\b("
    r"passports?|cards?|forms?|documents?|identification|id\b|bills?|consent|"
    r"licenses?|certificates?|photos?|proof|agreements?|applications?|"
    r"seal|my\s+number"
    r")\b"
)


def extract_checklist_structure(
    evidence: Sequence[RetrievedChunk],
) -> List[Dict[str, Any]]:
    """Typed structured required-item list for prompts and completeness checks.

    Internal shape (never shown to users):
      {
        "type": "identity_document" | "certificate" | "form" |
                "household_document" | "required_document",
        "required": True,
        "alternatives": ["item A", "item B"],  # length 1 when no alternative
        "condition": "when ..." | None,
        "scope": "all household members" | None,
        "item": "<primary label>",            # backward-compatible alias
        "alternative": "<alt or None>",       # backward-compatible alias
        "evidence_index": 1,
      }
    """
    structured: List[Dict[str, Any]] = []
    for evidence_index, item in extract_required_items(evidence):
        text = re.sub(r"\s+", " ", item).strip(" •-")
        if not text:
            continue
        text = re.sub(r"(?i)^or\s+", "", text).strip()
        alternatives: List[str] = []
        condition = None
        scope = None

        household_alt = re.search(
            r"(?i)^(.+?),\s*or\s+(.+?)\s+(for all(?:\s+household)?\s+members.*)$",
            text,
        ) or re.search(
            r"(?i)^(.+?)\s+or\s+(.+?)\s+(for all(?:\s+household)?\s+members.*)$",
            text,
        )
        if household_alt and len(household_alt.group(1).split()) <= 12:
            alternatives = [
                household_alt.group(1).strip(" ,"),
                household_alt.group(2).strip(" ,"),
            ]
            scope = _clean_scope_text(household_alt.group(3))
            text = alternatives[0]
        else:
            alt_match = re.search(
                r"(?i)^(.+?),\s*or\s+(.+?)(?:\s+(when|if|only if)\s+(.+))?$",
                text,
            )
            if not alt_match:
                alt_match = re.search(
                    r"(?i)^(.+?)\s+or\s+(.+?)(?:\s+(when|if|only if)\s+(.+))?$",
                    text,
                )
            if alt_match and len(alt_match.group(1).split()) <= 12:
                alternatives = [
                    alt_match.group(1).strip(" ,"),
                    alt_match.group(2).strip(" ,"),
                ]
                text = alternatives[0]
                if alt_match.lastindex and alt_match.lastindex >= 4 and alt_match.group(4):
                    condition = f"{alt_match.group(3)} {alt_match.group(4)}".strip()
            else:
                alternatives = [text]

        if not condition and not scope:
            probe = alternatives[0] if alternatives else text
            cond_match = re.search(
                r"(?i)^(.+?)(?:,\s*)?(only if|when|if|for)\s+(.+)$",
                probe,
            )
            if cond_match:
                trailing = cond_match.group(3).strip()
                if re.search(r"(?i)\b(household|all members)\b", trailing) and re.match(
                    r"(?i)^for\b", cond_match.group(2)
                ):
                    scope = _clean_scope_text(
                        f"{cond_match.group(2)} {trailing}"
                    )
                    if not re.search(r"(?i)\bhousehold\b", probe):
                        alternatives[0] = cond_match.group(1).strip(" ,")
                elif len(trailing.split()) <= 14:
                    alternatives[0] = cond_match.group(1).strip(" ,")
                    condition = f"{cond_match.group(2)} {trailing}".strip()

        # Household scope embedded in the item text without an alt split.
        if not scope:
            scope_match = re.search(
                r"(?i)^(.*?)\s*(?:,\s*)?(for all(?:\s+household)?\s+members.*)$",
                alternatives[0],
            )
            if scope_match and DOCUMENT_OBJECT_HINT.search(scope_match.group(1) or ""):
                # Only peel scope when the left side still looks like a document.
                left = scope_match.group(1).strip(" ,")
                if left and len(left.split()) <= 12:
                    alternatives[0] = left
                    scope = _clean_scope_text(scope_match.group(2))

        alternatives = [a for a in alternatives if a]
        for i, alt in enumerate(alternatives):
            cleaned = alt.strip(" ,")
            if cleaned and cleaned[0].islower() and i == 0:
                cleaned = cleaned[0].upper() + cleaned[1:]
            alternatives[i] = cleaned
        if not alternatives:
            continue

        primary = alternatives[0]
        item_type = _classify_checklist_type(
            primary=primary,
            alternatives=alternatives,
            condition=condition,
            scope=scope,
        )
        structured.append(
            {
                "type": item_type,
                "required": True,
                "alternatives": alternatives,
                "condition": condition,
                "scope": scope,
                # Backward-compatible fields used by older helpers/tests.
                "item": primary,
                "alternative": alternatives[1] if len(alternatives) > 1 else None,
                "evidence_index": evidence_index,
            }
        )
    return structured


def _clean_scope_text(raw: str | None) -> str | None:
    """Keep household scope; drop glued office-hours / weekday noise."""
    text = re.sub(r"\s+", " ", (raw or "")).strip(" ,.")
    if not text:
        return None
    text = re.split(
        r"(?i)\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"office hours?|closed)\b",
        text,
        maxsplit=1,
    )[0].strip(" ,.")
    return text or None


def _classify_checklist_type(
    *,
    primary: str,
    alternatives: Sequence[str],
    condition: Optional[str],
    scope: Optional[str],
) -> str:
    blob = " ".join(
        part
        for part in (
            primary,
            " ".join(alternatives),
            condition or "",
            scope or "",
        )
        if part
    )
    if scope or re.search(r"(?i)\b(household|all members)\b", blob):
        return "household_document"
    if re.search(r"(?i)\b(certificates?|moving-?out)\b", blob):
        return "certificate"
    if re.search(r"(?i)\b(forms?|consent|application)\b", blob):
        return "form"
    if re.search(
        r"(?i)\b(passports?|cards?|identification|identity|id\b|my\s+number)\b",
        blob,
    ):
        return "identity_document"
    return "required_document"


def synthesize_checklist_answer(
    evidence: Sequence[RetrievedChunk],
) -> str:
    """Clean checklist prose/bullets — never raw passage fragments."""
    items = extract_checklist_structure(evidence)
    if not items:
        return FALLBACK_ANSWER
    lines = ["Bring the following:"]
    for entry in items:
        piece = format_checklist_item_phrase(entry)
        if piece and piece[0].islower():
            piece = piece[0].upper() + piece[1:]
        lines.append(f"- {piece} [{entry['evidence_index']}]")
    office_line = _checklist_office_line(evidence)
    if office_line:
        lines.append(office_line)
    answer = "\n".join(lines)
    if answer_exposes_internal_field_keys(answer) or looks_like_answer_fragment(answer):
        return FALLBACK_ANSWER
    # Never leave deadline/procedure padding in the user-facing checklist.
    from app.generation.evidence_validation import strip_checklist_deadline_padding

    return strip_checklist_deadline_padding(answer)


def _checklist_office_line(evidence: Sequence[RetrievedChunk]) -> Optional[str]:
    """Preserve submit/office location from the same procedure section."""
    office_re = re.compile(
        r"(?i)\b("
        r"submit(?:\s+the\s+[\w-]+)?\s+(?:notification|application|form)?\s*"
        r"at\s+(?:the\s+)?[^.\n]{5,80}|"
        r"at\s+(?:the\s+)?(?:citizen|resident|insurance|welfare|city)\s+"
        r"(?:services?\s+)?(?:window|desk|counter|office)[^.\n]{0,40}"
        r")\b"
    )
    for index, chunk in enumerate(evidence, start=1):
        blob = "\n".join(
            part
            for part in (
                chunk.content or "",
                chunk.section_title or "",
                getattr(chunk, "responsible_office", None) or "",
            )
            if part
        )
        match = office_re.search(blob)
        if not match:
            continue
        phrase = re.sub(r"\s+", " ", match.group(0)).strip(" .,")
        if not phrase:
            continue
        if not phrase[0].isupper():
            phrase = phrase[0].upper() + phrase[1:]
        if not phrase.endswith((".", "!", "?")):
            phrase += "."
        return f"{phrase} [{index}]"
    return None


def format_checklist_item_phrase(entry: Dict[str, Any]) -> str:
    alts = entry.get("alternatives")
    if isinstance(alts, list) and alts:
        piece = " or ".join(str(a).strip() for a in alts if str(a).strip())
    else:
        piece = str(entry.get("item") or "").strip()
        if entry.get("alternative"):
            piece = f"{piece} or {entry['alternative']}"
    condition = entry.get("condition")
    scope = entry.get("scope")
    if condition:
        cond = str(condition)
        if cond.lower() not in piece.lower():
            if re.match(r"(?i)^(when|if|only if|for)\b", cond):
                piece = f"{piece}, {cond}"
            else:
                piece = f"{piece} ({cond})"
    if scope:
        scope_text = str(scope)
        if scope_text.lower() not in piece.lower():
            if re.match(r"(?i)^for\b", scope_text):
                piece = f"{piece}, {scope_text}"
            else:
                piece = f"{piece} ({scope_text})"
    return re.sub(r"(?i)^(a|an|the)\s+", "", piece).strip()


def missing_checklist_items(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> List[Dict[str, Any]]:
    """Return structured required items not covered by the answer."""
    structure = extract_checklist_structure(evidence)
    if not structure:
        return []
    missing: List[Dict[str, Any]] = []
    for entry in structure:
        if not checklist_structure_is_covered(answer, [entry]):
            missing.append(entry)
    return missing


def checklist_regeneration_instruction(
    evidence: Sequence[RetrievedChunk], answer: str
) -> str:
    """Explicit regeneration instruction listing every missing required item."""
    missing = missing_checklist_items(answer, evidence)
    structure = extract_checklist_structure(evidence)
    targets = missing or structure
    lines = [format_checklist_item_phrase(entry) for entry in targets if entry]
    lines = [line for line in lines if line]
    if not lines:
        return (
            "Include every explicitly required document or item from the evidence. "
            "Preserve alternatives, conditions, and household-wide requirements. "
            "Include each required item exactly once. "
            "Do not replace missing items with deadline or location information."
        )
    return (
        "Your previous answer was incomplete. Include ALL of these required items "
        "exactly once (paraphrase is fine; keep exact document names, alternatives, "
        "conditions, and household scope). Do not omit any item, and do not replace "
        "missing items with deadline or location information:\n"
        + "\n".join(f"- {line}" for line in lines)
    )


def synthesize_fee_answer(
    question: str, evidence: Sequence[RetrievedChunk]
) -> str:
    """Natural-language fee facts — never snake_case CSV dumps."""
    selected = relevant_price_evidence(question, evidence)
    if not selected:
        return FALLBACK_ANSWER
    requested_age = _extract_requested_age(question)
    service_label = _fee_service_label(question, evidence)
    lines: List[str] = []
    seen = set()
    for evidence_index, chunk in selected:
        raw = chunk.content or ""
        if looks_like_structured_record(raw, chunk):
            prose = format_structured_content_as_prose(raw, chunk)
        else:
            prose = format_structured_content_as_prose(raw, chunk)
            if prose == raw and answer_exposes_internal_field_keys(raw):
                continue
            if not looks_like_structured_record(raw, chunk):
                # Plain currency sentence already in evidence.
                prose = re.sub(r"\s+", " ", raw).strip()
        # Age-specific benefit: keep only the matching bracket sentence.
        if requested_age is not None and prose:
            sentences = [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+|\n+", prose)
                if part.strip()
            ]
            age_hits = [
                sent
                for sent in sentences
                if _age_bracket_match_score(sent, requested_age) >= 0.8
                and re.search(r"(?i)\b(?:¥|yen|\d[\d,]*)\b", sent)
            ]
            if age_hits:
                prose = age_hits[0]
        prose = re.sub(r"\s+", " ", prose).strip()
        if not prose or answer_exposes_internal_field_keys(prose):
            continue
        if looks_like_answer_fragment(prose):
            continue
        fingerprint = re.sub(r"\W+", " ", prose.lower()).strip()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        if not prose.endswith((".", "!", "?")):
            prose += "."
        lines.append(f"{prose} [{evidence_index}]")
    if not lines:
        return FALLBACK_ANSWER
    combined = _compose_named_fee_answer(
        service_label,
        lines,
        minimum_fee=_minimum_fee_from_evidence(evidence),
    )
    if combined:
        return combined
    if len(lines) == 1:
        return lines[0]
    return "\n".join(f"- {line}" for line in lines)


def _minimum_fee_from_evidence(evidence: Sequence[RetrievedChunk]) -> str:
    for chunk in evidence:
        match = re.search(
            r"(?i)(minimum\s+fee\s*:\s*(?:SGD|USD|EUR|GBP|AUD)?\s*[\d,]+)",
            chunk.content or "",
        )
        if match:
            return match.group(1).strip()
    return ""


def _fee_service_label(
    question: str, evidence: Sequence[RetrievedChunk]
) -> str:
    """Best service/product name for a fee answer (from evidence, else question)."""
    from app.retrieval.entity_validation import (
        extract_requested_service_phrases,
        service_phrases_for_matching,
    )

    phrases = service_phrases_for_matching(extract_requested_service_phrases(question))
    for chunk in evidence:
        text = chunk.content or ""
        match = re.search(r"(?i)\bservice\s*:\s*([^.\n]+)", text)
        if match:
            label = re.sub(r"\s+", " ", match.group(1)).strip(" .;")
            label = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
            if not label:
                continue
            if not phrases or any(
                p.lower() in label.lower() or label.lower() in p.lower() for p in phrases
            ):
                return label
    for phrase in phrases:
        if len(phrase) >= 4:
            return phrase
    return ""


def _compose_named_fee_answer(
    service_label: str,
    lines: Sequence[str],
    *,
    minimum_fee: str = "",
) -> str:
    """Build one grounded sentence that names the service (entity-gate safe)."""
    if not lines:
        return ""
    blob = " ".join(lines)
    # Strip citation markers while collecting first cite.
    cites = re.findall(r"\[(\d+)\]", blob)
    cite = f" [{cites[0]}]" if cites else ""
    plain = re.sub(r"\s*\[\d+\]", "", blob)
    plain = re.sub(r"\s+", " ", plain).strip(" -")
    if minimum_fee and minimum_fee.lower() not in plain.lower():
        plain = f"{plain} {minimum_fee}".strip()
    if not plain:
        return ""
    if service_label and service_label.lower() not in plain.lower():
        # Prefer "The permanent placement fee is 22% … Minimum fee: SGD 9,000."
        # Prefer an explicit percentage-of-salary fee when present.
        pct_match = re.search(
            r"(?i)\d{1,3}(?:\.\d+)?\s*%\s+of\s+(?:the\s+)?"
            r"(?:candidate'?s\s+)?(?:first[- ]year\s+)?(?:base\s+salary|total\s+compensation)"
            r"[^.\[\]]*",
            plain,
        )
        if not pct_match:
            pct_match = re.search(r"(?i)\d{1,3}(?:\.\d+)?\s*%\s+of\s+first[- ]year[^.\[\]]*", plain)
        min_match = re.search(
            r"(?i)(minimum\s+fee\s*:\s*(?:SGD|USD|EUR|GBP|AUD)?\s*[\d,]+|"
            r"minimum(?:\s+fee)?\s+(?:of\s+)?(?:SGD|USD|EUR|GBP|AUD)?\s*[\d,]+)",
            plain,
        )
        if pct_match:
            amount = pct_match.group(0).strip(" .;")
            sentence = f"The {service_label} fee is {amount}"
            if min_match and min_match.group(1).lower() not in sentence.lower():
                sentence = f"{sentence}. {min_match.group(1).rstrip('.')}"
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            return f"{sentence}{cite}"
        # Minimum-fee money alone is not the placement fee (need an explicit %).
        if re.search(r"(?i)\bpermanent\s+placement|placement\s+fee\b", service_label or ""):
            if not re.search(r"\d{1,3}(?:\.\d+)?\s*%", plain):
                return ""
        # Preserve multiple labeled fees for the same service instead of
        # collapsing the answer to the first currency amount.
        counter_match = re.search(
            r"(?i)\bcounter\s+fee\s*:\s*"
            r"((?:SGD|USD|EUR|GBP|AUD)\s*[\d,]+|¥[\d,]+|\$[\d,]+|[\d,]+\s*yen)",
            plain,
        )
        kiosk_match = re.search(
            r"(?i)\b(?:convenience[- ]store\s+)?kiosk\s+fee\s*:\s*"
            r"((?:SGD|USD|EUR|GBP|AUD)\s*[\d,]+|¥[\d,]+|\$[\d,]+|[\d,]+\s*yen)",
            plain,
        )
        if counter_match and kiosk_match:
            sentence = (
                f"The {service_label} counter fee is {counter_match.group(1)}, "
             f"and the convenience-store kiosk fee is {kiosk_match.group(1)}."
            )
            return f"{sentence}{cite}"

        fee_match = re.search(
            r"(?i)(?:fee[^.:]*:\s*)?((?:SGD|USD|EUR|GBP|AUD)\s*[\d,]+|¥[\d,]+|\$[\d,]+)",
            plain,
        )
        # Never promote a "Minimum fee: SGD …" row as the primary fee amount.
        if fee_match and re.search(r"(?i)minimum\s+fee", plain):
            if not re.search(r"\d{1,3}(?:\.\d+)?\s*%", plain):
                return ""
        if fee_match:
            amount = fee_match.group(1).strip(" .;")
            sentence = f"The {service_label} fee is {amount}"
            if min_match and min_match.group(1).lower() not in sentence.lower():
                sentence = f"{sentence}. {min_match.group(1).rstrip('.')}"
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            return f"{sentence}{cite}"
        # Fallback: prefix the service name onto the first fact.
        first = re.sub(r"\s*\[\d+\]", "", lines[0]).strip()
        if minimum_fee and minimum_fee.lower() not in first.lower():
            first = f"{first.rstrip('.')} {minimum_fee}."
        return f"For {service_label}: {first}{cite}"
    if len(lines) == 1:
        answer = lines[0]
        if minimum_fee and minimum_fee.lower() not in answer.lower():
            answer = re.sub(r"\s*\[\d+\]\s*$", "", answer).rstrip(".")
            cite_m = re.search(r"\[(\d+)\]", lines[0])
            cite_s = f" [{cite_m.group(1)}]" if cite_m else ""
            return f"{answer}. {minimum_fee}.{cite_s}"
        return answer
    return ""


def checklist_structure_is_covered(
    answer: str, structure: Sequence[Dict[str, Any]]
) -> bool:
    if not structure:
        return False
    answer_l = re.sub(r"\W+", " ", (answer or "").lower())
    for entry in structure:
        alts = entry.get("alternatives")
        if isinstance(alts, list) and alts:
            item = str(alts[0])
            alternative = str(alts[1]) if len(alts) > 1 else None
        else:
            item = entry.get("item") or ""
            alternative = entry.get("alternative")
        tokens = [
            t
            for t in re.findall(r"[a-z0-9]{3,}", item.lower())
            if t
            not in {
                "the",
                "and",
                "for",
                "all",
                "with",
                "your",
                "from",
                "when",
                "moving",
                "another",
            }
        ]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in answer_l)
        if hits < max(1, (len(tokens) + 1) // 2):
            compact = re.sub(r"\W+", " ", item.lower()).strip()
            words = compact.split()
            found = False
            for size in (3, 2):
                for i in range(0, max(0, len(words) - size + 1)):
                    phrase = " ".join(words[i : i + size])
                    if phrase in answer_l:
                        found = True
                        break
                if found:
                    break
            if not found:
                return False
        if alternative:
            alt = alternative.lower()
            alt_tokens = re.findall(r"[a-z0-9]{4,}", alt)
            if alt_tokens and not any(t in answer_l for t in alt_tokens[:3]):
                if " or " not in f" {answer_l} ":
                    return False
        condition = entry.get("condition") or ""
        scope = entry.get("scope") or ""
        if condition:
            cond_l = condition.lower()
            if re.search(r"(?i)\bonly if\b", cond_l):
                if "only if" not in answer_l and "if " not in f" {answer_l} ":
                    return False
            elif re.search(r"(?i)^when\b", cond_l):
                when_tokens = [
                    t
                    for t in re.findall(r"[a-z0-9]{4,}", cond_l)
                    if t not in {"when", "from", "another"}
                ]
                if when_tokens and not any(t in answer_l for t in when_tokens[:2]):
                    return False
        if scope or re.search(r"(?i)\bhousehold\b", condition):
            if (
                "household" not in answer_l
                and "all members" not in answer_l
                and "each member" not in answer_l
            ):
                return False
    return True


def is_retrieval_dump_answer(answer: str) -> bool:
    """Detect answers that are clearly pasted retrieval / record dumps."""
    text = (answer or "").strip()
    if not text:
        return True
    if answer_exposes_internal_field_keys(text):
        return True
    if re.search(
        r"(?i)^applicable fees:\s*[-•].*(certificate_or_service|fee_jpy|where_to_apply)",
        text,
    ):
        return True
    if re.search(
        r"(?i)\b(certificate_or_service|fee_jpy|where_to_apply)\s*:",
        text,
    ):
        return True
    # Heading glued to body without punctuation.
    if re.match(
        r"(?i)^(?:moving in|move-in notification|national health)[^(]{0,40}\)\s+If\b",
        text,
    ):
        return True
    if looks_like_answer_fragment(text):
        return True
    return False


def normalize_evidence_bundle(
    evidence: Sequence[RetrievedChunk],
    *,
    question: str = "",
) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    """Full evidence normalization + diagnostics for generation."""
    from app.generation.evidence_presentation import (
        merge_procedure_evidence,
        repair_passage_text,
    )

    diagnostics: Dict[str, Any] = {
        "selected_chunk_ids": [
            getattr(c, "chunk_id", None) or f"{c.document_name}:{c.page_number}:{c.score}"
            for c in evidence
        ],
        "chunk_boundary_flags": [],
        "merged_siblings": [],
        "checklist_structure": [],
        "fee_rows_normalized": [],
    }
    for chunk in evidence:
        text = chunk.content or ""
        diagnostics["chunk_boundary_flags"].append(
            {
                "document_name": chunk.document_name,
                "section_title": chunk.section_title,
                "starts_on_boundary": bool(
                    text and (text[0].isupper() or text[0] in "•-\"“'")
                ),
                "ends_on_boundary": bool(
                    text.rstrip().endswith((".", "!", "?", ":", ";"))
                    or text.rstrip().endswith(("•",))
                ),
                "preview": " ".join(text.split())[:120],
            }
        )

    merged = merge_procedure_evidence(evidence)
    if len(merged) < len(evidence):
        diagnostics["merged_siblings"] = [
            {
                "document_name": c.document_name,
                "section_title": c.section_title,
                "length": len(c.content or ""),
            }
            for c in merged
        ]

    prepared: List[RetrievedChunk] = []
    for chunk in merged:
        content = repair_passage_text(
            chunk.content or "",
            section_title=chunk.section_title,
        )
        if looks_like_structured_record(content, chunk):
            prose = format_structured_content_as_prose(content, chunk)
            diagnostics["fee_rows_normalized"].append(
                {
                    "document_name": chunk.document_name,
                    "row_number": chunk.row_number,
                    "internal_preview": " ".join((chunk.content or "").split())[:160],
                    "normalized": prose,
                }
            )
            content = prose
        prepared.append(chunk.model_copy(update={"content": content}))

    # Second pass via shared prepare for consistency.
    prepared = prepare_evidence_for_generation(prepared, question=question)
    diagnostics["checklist_structure"] = extract_checklist_structure(prepared)
    diagnostics["final_clean_evidence"] = [
        {
            "document_name": c.document_name,
            "section_title": c.section_title,
            "content": (c.content or "")[:400],
        }
        for c in prepared
    ]
    return prepared, diagnostics
