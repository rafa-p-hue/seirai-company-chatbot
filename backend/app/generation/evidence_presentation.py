"""Convert structured retrieval evidence into clean, user-facing prose.

Internal snake_case field names stay in metadata only. Chunk content sent to
the LLM and deterministic answer paths uses natural-language facts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models.api import RetrievedChunk

LABELED_PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_]*(?:[ \t]+[A-Za-z][A-Za-z0-9_]*){0,6})\s*:\s*(?P<value>[^;|\n]+)"
)
SNAKE_KEY_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
KNOWN_INTERNAL_KEYS = {
    "certificate_or_service",
    "fee_jpy",
    "where_to_apply",
    "notes",
    "effective_date",
    "document_status",
    "service_domain",
    "row_number",
    "content_type",
    "column_labels",
    "fee_usd",
    "fee_amount",
    "valid_from",
    "valid_to",
}

FEE_SUBJECT_KEYS = {
    "certificate_or_service",
    "certificate",
    "service",
    "item",
    "product",
    "name",
    "title",
}
FEE_AMOUNT_KEYS = {
    "fee_jpy",
    "fee_usd",
    "fee_eur",
    "fee_gbp",
    "fee",
    "price",
    "cost",
    "amount",
    "fee_amount",
    "minimum_fee",
}
JOB_LISTING_KEYS = {
    "job_id",
    "salary_range",
    "salary",
    "compensation",
    "employment_type",
    "remote_option",
    "posted_date",
    "practice",
}
FEE_PLACE_KEYS = {
    "where_to_apply",
    "location",
    "office",
    "counter",
    "window",
    "place",
}
FEE_NOTE_KEYS = {"notes", "note", "condition", "conditions", "remarks"}
FEE_DATE_KEYS = {"effective_date", "effective", "valid_from", "as_of"}


def humanize_field_label(key: str) -> str:
    text = re.sub(r"[_\-]+", " ", (key or "").strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # Keep short currency codes uppercase.
    parts = []
    for part in text.split(" "):
        if part.upper() in {"JPY", "USD", "EUR", "GBP", "NHI"}:
            parts.append(part.upper())
        else:
            parts.append(part.capitalize())
    return " ".join(parts)


def parse_labeled_fields(content: str) -> Dict[str, str]:
    """Parse `key: value; key: value` or newline-separated labeled rows."""
    text = (content or "").strip()
    if not text:
        return {}
    fields: Dict[str, str] = {}
    # Prefer semicolon-separated CSV human_text, else line-based.
    candidates = []
    if ";" in text and LABELED_PAIR_RE.search(text):
        candidates = [part.strip() for part in text.split(";") if part.strip()]
    else:
        candidates = [line.strip() for line in text.splitlines() if line.strip()]
        if len(candidates) == 1:
            single = candidates[0]
            # Split compact table rows: "Counter fee: ¥300 Kiosk fee: ¥200"
            split_hits = list(
                re.finditer(
                    r"(?=(?:^|(?<=\s))([A-Za-z][A-Za-z0-9_/ -]{1,40})\s*:)",
                    single,
                )
            )
            if len(split_hits) >= 2:
                parts = []
                for i, hit in enumerate(split_hits):
                    start = hit.start(1)
                    end = (
                        split_hits[i + 1].start(1)
                        if i + 1 < len(split_hits)
                        else len(single)
                    )
                    parts.append(single[start:end].strip(" ;"))
                candidates = [p for p in parts if p]
            elif "|" in single:
                candidates = [
                    part.strip() for part in single.split("|") if part.strip()
                ]
    for part in candidates:
        match = LABELED_PAIR_RE.match(part)
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip(" .;")
        if key and value:
            fields[key] = value
    return fields


def looks_like_structured_record(content: str, chunk: Optional[RetrievedChunk] = None) -> bool:
    if chunk is not None:
        ctype = (chunk.content_type or "").lower()
        ftype = (getattr(chunk, "file_type", None) or "").lower()
        if ctype in {"table", "structured_table_row"}:
            return True
        if ftype == "csv" or chunk.row_number is not None:
            return True
        if chunk.table_data:
            return True
        # key_value only when the payload still carries internal/snake keys.
        if ctype == "key_value":
            fields = parse_labeled_fields(content or "")
            if any(
                "_" in key or key.lower() in KNOWN_INTERNAL_KEYS for key in fields
            ):
                return True
    fields = parse_labeled_fields(content or "")
    if not fields:
        return False
    snake = sum(
        1 for key in fields if "_" in key or key.lower() in KNOWN_INTERNAL_KEYS
    )
    # Require real internal keys — do not rewrite plain PDF "Counter fee: 300 yen".
    return snake >= 1 and (len(fields) >= 1)


def _format_money(value: str, key: str = "") -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if re.search(r"[$¥€£]", raw):
        return raw
    number = re.sub(r"[^\d.]", "", raw)
    if not number:
        return raw
    key_l = (key or "").lower()
    if "jpy" in key_l or "yen" in key_l or key_l.endswith("_jpy"):
        return f"¥{number}"
    if "usd" in key_l or "dollar" in key_l:
        return f"${number}"
    if "eur" in key_l:
        return f"€{number}"
    if "gbp" in key_l or "pound" in key_l:
        return f"£{number}"
    # Preserve bare numeric amounts without inventing a currency glyph.
    return number


def format_fee_fields_as_prose(
    fields: Dict[str, str],
    *,
    effective_date: Optional[str] = None,
    document_status: Optional[str] = None,
) -> str:
    """Turn fee CSV/table fields into one natural-language fact sentence."""
    if not fields:
        return ""
    normalized = {
        re.sub(r"[\s-]+", "_", k.strip().lower()): (k, v) for k, v in fields.items()
    }

    subject = ""
    for key in FEE_SUBJECT_KEYS:
        if key in normalized:
            subject = normalized[key][1]
            break
    if not subject:
        for key, (original, value) in normalized.items():
            if key not in FEE_AMOUNT_KEYS | FEE_PLACE_KEYS | FEE_NOTE_KEYS | FEE_DATE_KEYS | {
                "status",
                "document_status",
                "version",
            } and "fee" not in key and "price" not in key and "cost" not in key:
                subject = value
                break

    fee_parts: List[str] = []
    for key, (original, value) in normalized.items():
        if key in FEE_AMOUNT_KEYS or re.search(r"(?:fee|price|cost|amount)", key):
            money = _format_money(value, key)
            if not money:
                continue
            label = humanize_field_label(original).lower()
            # Avoid "fee jpy: ¥350" — prefer bare money when label is generic.
            if label in {"fee", "fee jpy", "fee usd", "price", "cost", "amount"}:
                fee_parts.append(money)
            else:
                fee_parts.append(f"{label} {money}")

    place = ""
    for key in FEE_PLACE_KEYS:
        if key in normalized:
            place = normalized[key][1]
            break

    notes = ""
    for key in FEE_NOTE_KEYS:
        if key in normalized:
            notes = normalized[key][1]
            break

    date = effective_date or ""
    for key in FEE_DATE_KEYS:
        if key in normalized:
            date = normalized[key][1]
            break

    per = ""
    if notes and re.search(r"(?i)\bper\s+(copy|page|document|certificate|item)\b", notes):
        match = re.search(r"(?i)\bper\s+(?:copy|page|document|certificate|item)\b", notes)
        if match:
            per = match.group(0).lower()

    parts: List[str] = []
    if subject and fee_parts:
        money_clause = "; ".join(fee_parts)
        if per and len(fee_parts) == 1:
            parts.append(f"{subject} costs {fee_parts[0]} {per}")
        elif len(fee_parts) == 1:
            parts.append(f"{subject} costs {fee_parts[0]}")
        else:
            parts.append(f"{subject}: {money_clause}")
    elif fee_parts:
        parts.append("; ".join(fee_parts))
    elif subject:
        parts.append(subject)

    if place:
        parts.append(f"at {place}" if parts else place)

    sentence = " ".join(parts).strip()
    if not sentence:
        # Never fall back to snake_case labels in user-facing prose.
        generic = []
        for k, v in fields.items():
            if not v or k.lower() in {"status", "document_status"}:
                continue
            if "_" in k or k.lower() in KNOWN_INTERNAL_KEYS:
                continue
            generic.append(f"{humanize_field_label(k)}: {v}")
        sentence = "; ".join(generic)

    extras: List[str] = []
    if notes:
        note_clean = notes
        if per:
            note_clean = re.sub(
                r"(?i)\bper\s+(?:copy|page|document|certificate|item)\.?",
                "",
                note_clean,
            ).strip(" .;")
        if not date:
            date_match = re.search(
                r"(?i)effective(?:\s+date)?\s*:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
                notes,
            )
            if date_match:
                date = date_match.group(1).strip()
                note_clean = (notes[: date_match.start()] + notes[date_match.end() :]).strip(
                    " .;"
                )
        if note_clean and note_clean.lower() not in sentence.lower():
            extras.append(note_clean.rstrip(".") + ".")
    if date and date.lower() not in sentence.lower():
        extras.append(f"Effective {date}.")
    if (document_status or "").lower() == "archived":
        extras.append("Historical/archived fee schedule.")

    if extras:
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        sentence = f"{sentence} {' '.join(extras)}"
    elif sentence and not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence.strip()


def format_job_listing_as_prose(fields: Dict[str, Any]) -> str:
    """Turn open-position / job CSV rows into natural salary/role prose."""
    if not fields:
        return ""
    normalized = {
        re.sub(r"[\s-]+", "_", str(k).strip().lower()): str(v).strip()
        for k, v in fields.items()
        if str(v).strip()
    }
    title = normalized.get("title") or normalized.get("role") or normalized.get("position")
    location = normalized.get("location") or normalized.get("city") or normalized.get("office")
    salary = (
        normalized.get("salary_range")
        or normalized.get("salary")
        or normalized.get("compensation")
        or normalized.get("pay_range")
    )
    employment = normalized.get("employment_type") or normalized.get("type")
    parts: List[str] = []
    if title and salary and location:
        parts.append(f"The {title} role in {location} has a salary range of {salary}")
    elif title and salary:
        parts.append(f"The {title} role has a salary range of {salary}")
    elif title and location:
        parts.append(f"{title} in {location}")
    elif salary:
        parts.append(f"Salary range: {salary}")
    elif title:
        parts.append(title)
    if employment and parts:
        parts[0] = f"{parts[0]} ({employment})"
    sentence = parts[0] if parts else ""
    if sentence and not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def format_structured_content_as_prose(
    content: str,
    chunk: Optional[RetrievedChunk] = None,
) -> str:
    """Convert structured/CSV labeled content into clean prose for generation."""
    text = (content or "").strip()
    if not text:
        return ""
    fields = parse_labeled_fields(text)
    if chunk and chunk.table_data and isinstance(chunk.table_data, dict):
        labels = chunk.table_data.get("column_labels") or []
        cells = chunk.table_data.get("cells") or []
        if labels and cells and len(labels) == len(cells):
            from_table = {
                str(label): str(cell)
                for label, cell in zip(labels, cells)
                if str(cell).strip()
            }
            if from_table:
                fields = from_table
    if not fields:
        return text

    keys_l = {re.sub(r"[\s-]+", "_", k.lower()) for k in fields}

    # Job / open-position rows — never route through certificate-fee prose.
    if keys_l & JOB_LISTING_KEYS or (
        "title" in keys_l
        and "location" in keys_l
        and ("salary_range" in keys_l or "salary" in keys_l or "employment_type" in keys_l)
    ):
        return format_job_listing_as_prose(fields)

    # Fee-shaped records require an actual fee/amount field — a bare "title"
    # key (common in job CSVs) must not enter the fee formatter.
    if keys_l & FEE_AMOUNT_KEYS or (
        keys_l & FEE_SUBJECT_KEYS
        and any(
            re.search(r"(?i)\b(fee|price|cost|¥|yen|sgd|usd|gbp|aud)\b", str(v))
            for v in fields.values()
        )
        and "salary_range" not in keys_l
    ):
        return format_fee_fields_as_prose(
            fields,
            effective_date=getattr(chunk, "effective_date", None) if chunk else None,
            document_status=getattr(chunk, "document_status", None) if chunk else None,
        )

    # Generic structured record → "Label: value" with humanized labels.
    parts = [
        f"{humanize_field_label(key)}: {value}"
        for key, value in fields.items()
        if value
        and re.sub(r"[\s-]+", "_", key.lower())
        not in {"status", "document_status", "version"}
    ]
    return "; ".join(parts) if parts else text


def _chunk_sort_key(chunk: RetrievedChunk) -> Tuple[int, int, int]:
    page = chunk.page_number if chunk.page_number is not None else 10**9
    slide = chunk.slide_number if chunk.slide_number is not None else 10**9
    row = chunk.row_number if chunk.row_number is not None else 10**9
    return (page, slide, row)


def _strip_overlap(previous: str, nxt: str) -> str:
    """Remove accidental overlap when concatenating adjacent chunks."""
    left = (previous or "").rstrip()
    right = (nxt or "").lstrip()
    if not left or not right:
        return right
    max_window = min(len(left), len(right), 240)
    for size in range(max_window, 24, -1):
        if left[-size:].lower() == right[:size].lower():
            return right[size:].lstrip()
    # Word-overlap fallback at the join.
    left_tail = " ".join(left.split()[-12:])
    right_words = right.split()
    for n in range(min(12, len(right_words)), 2, -1):
        candidate = " ".join(right_words[:n])
        if left_tail.lower().endswith(candidate.lower()):
            return " ".join(right_words[n:]).lstrip()
    return right


def repair_passage_text(
    text: str,
    *,
    section_title: Optional[str] = None,
) -> str:
    """Repair mid-sentence starts/ends and strip glued section headings."""
    body = (text or "").strip()
    if not body:
        return ""
    if section_title:
        title = section_title.strip()
        if title:
            # Drop exact heading prefix glued to the body.
            body = re.sub(
                rf"(?is)^\s*{re.escape(title)}\s*[:\-]?\s*",
                "",
                body,
                count=1,
            ).strip()
            # "Moving In (Tennyu Todoke) If you move..." without separator.
            compact_title = re.sub(r"\s+", " ", title).strip()
            if body.lower().startswith(compact_title.lower()) and len(body) > len(
                compact_title
            ):
                remainder = body[len(compact_title) :].lstrip(" :-")
                if remainder and remainder[0].isupper():
                    body = remainder
    # Mid-sentence start → keep from the next sentence boundary.
    # Only lowercase openings are true continuations. Do NOT treat capitalized
    # "To/For/With..." as fragments — those are valid sentence starts.
    if body and body[0].islower():
        match = re.search(r"(?<=[.!?])\s+(?=[A-Z\"“])", body)
        if match:
            body = body[match.end() :].lstrip()
        else:
            cap = re.search(r"\b([A-Z][A-Za-z].*)", body)
            if cap and len(cap.group(1)) > len(body) * 0.5:
                body = cap.group(1)
    # Mid-sentence end → drop dangling trailing clause after last terminator.
    stripped = body.rstrip()
    if stripped and not stripped.endswith((".", "!", "?", ":", ";", "•")):
        last_stop = max(stripped.rfind(". "), stripped.rfind("! "), stripped.rfind("? "))
        if last_stop >= 40 and len(stripped) - last_stop < 80:
            dangling = stripped[last_stop + 1 :].strip()
            if re.search(
                r"(?i)\b(?:and|or|with|for|to|of|the|a|an|bring|your)\s*$",
                dangling,
            ) or dangling[0:1].islower():
                body = stripped[: last_stop + 1].strip()
    return re.sub(r"[ \t]+\n", "\n", body).strip()


def _is_atomic_structured_row(chunk: RetrievedChunk) -> bool:
    """CSV/table rows must stay isolated — never fuse distinct jobs/fee lines."""
    ctype = (chunk.content_type or "").lower()
    if ctype in {"table", "structured_table_row"}:
        return True
    if chunk.row_number is not None:
        return True
    ftype = (getattr(chunk, "file_type", None) or "").lower()
    if ftype == "csv":
        return True
    return False


def expand_sentence_boundaries(
    evidence: Sequence[RetrievedChunk],
) -> List[RetrievedChunk]:
    """Use sibling chunks in the same document to complete mid-sentence edges."""
    if len(evidence) <= 1:
        return list(evidence)

    def _starts_mid(text: str) -> bool:
        body = (text or "").strip()
        if not body:
            return False
        # Labeled fields / CSV keys ("job_id:", "salary_range:") are complete
        # records, not unfinished sentence continuations.
        if re.match(r"^[A-Za-z][A-Za-z0-9_ /-]{0,48}:\s*\S", body):
            return False
        # Only lowercase openings are unfinished sentences from a prior chunk.
        return body[0].islower()

    def _ends_mid(text: str) -> bool:
        body = (text or "").rstrip()
        if not body:
            return False
        if body.endswith((".", "!", "?", ":", ";", "•")):
            return False
        return bool(
            re.search(
                r"(?i)\b(?:and|or|with|for|to|of|the|a|an|bring|your|landing|passport)\s*$",
                body,
            )
        )

    by_doc: Dict[str, List[RetrievedChunk]] = {}
    doc_order: List[str] = []
    for chunk in evidence:
        key = (chunk.document_name or "").strip().lower()
        if key not in by_doc:
            by_doc[key] = []
            doc_order.append(key)
        by_doc[key].append(chunk)

    expanded: List[RetrievedChunk] = []
    for key in doc_order:
        ordered = sorted(by_doc[key], key=_chunk_sort_key)
        i = 0
        while i < len(ordered):
            primary = ordered[i]
            fused = (primary.content or "").strip()
            j = i
            # Atomic table/CSV rows are never fused with siblings.
            if _is_atomic_structured_row(primary):
                fused = repair_passage_text(
                    fused, section_title=primary.section_title
                )
                expanded.append(primary.model_copy(update={"content": fused}))
                i = j + 1
                continue
            # Pull following siblings while either side is mid-sentence.
            while j + 1 < len(ordered):
                nxt = ordered[j + 1]
                nxt_text = (nxt.content or "").strip()
                if not nxt_text:
                    break
                if _is_atomic_structured_row(nxt):
                    break
                list_continuation = bool(
                    re.match(r"^\s*[-•●▪◦·]", nxt_text)
                    or re.match(
                        r"^\s*[-•●▪◦·]",
                        fused.splitlines()[-1] if fused else "",
                    )
                )
                # PPTX/PDF slide/page list bullets on different pages are separate
                # sections — do not glue Alert Levels into Shelters just because
                # both start with "•". Require same page/slide or same section.
                same_section = bool(
                    (primary.section_title or "").strip()
                    and (primary.section_title or "").strip().lower()
                    == (nxt.section_title or "").strip().lower()
                )
                same_page = (
                    primary.page_number is not None
                    and nxt.page_number is not None
                    and primary.page_number == nxt.page_number
                )
                same_slide = (
                    getattr(primary, "slide_number", None) is not None
                    and getattr(nxt, "slide_number", None) is not None
                    and primary.slide_number == nxt.slide_number
                )
                if list_continuation and not (same_section or same_page or same_slide):
                    list_continuation = False
                if not (
                    _ends_mid(fused) or _starts_mid(nxt_text) or list_continuation
                ):
                    break
                fused = f"{fused} {_strip_overlap(fused, nxt_text)}".strip()
                j += 1
            fused = repair_passage_text(
                fused, section_title=primary.section_title
            )
            expanded.append(primary.model_copy(update={"content": fused}))
            i = j + 1
    return expanded


def merge_procedure_evidence(
    evidence: Sequence[RetrievedChunk],
) -> List[RetrievedChunk]:
    """Merge adjacent same-document/section/procedure chunks in source order."""
    if len(evidence) <= 1:
        return list(evidence)

    groups: Dict[Tuple[str, str], List[RetrievedChunk]] = {}
    order: List[Tuple[str, str]] = []
    for chunk in evidence:
        key = (
            (chunk.document_name or "").strip().lower(),
            (chunk.section_title or "").strip().lower(),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(chunk)

    merged: List[RetrievedChunk] = []
    for key in order:
        chunks = sorted(groups[key], key=_chunk_sort_key)
        if len(chunks) == 1:
            merged.append(chunks[0])
            continue
        if not key[1] or any(_is_atomic_structured_row(c) for c in chunks):
            # No shared section, or distinct CSV/table rows — keep separate.
            merged.extend(chunks)
            continue
        seen_text = set()
        parts: List[str] = []
        for chunk in chunks:
            body = (chunk.content or "").strip()
            if not body:
                continue
            fingerprint = re.sub(r"\W+", " ", body.lower()).strip()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            if parts:
                body = _strip_overlap(parts[-1], body)
            if body:
                parts.append(body)
        if not parts:
            merged.extend(chunks)
            continue
        primary = chunks[0]
        fused = repair_passage_text(
            "\n".join(parts),
            section_title=primary.section_title,
        )
        merged.append(primary.model_copy(update={"content": fused}))
    return merged


def prepare_evidence_for_generation(
    evidence: Sequence[RetrievedChunk],
    *,
    question: str = "",
) -> List[RetrievedChunk]:
    """Merge procedure passages, repair boundaries, rewrite structured records."""
    del question  # reserved for future domain-specific expansion cues
    expanded = expand_sentence_boundaries(evidence)
    merged = merge_procedure_evidence(expanded)
    prepared: List[RetrievedChunk] = []
    for chunk in merged:
        content = repair_passage_text(
            chunk.content or "",
            section_title=chunk.section_title,
        )
        if looks_like_structured_record(content, chunk):
            content = format_structured_content_as_prose(content, chunk)
        prepared.append(chunk.model_copy(update={"content": content}))
    return prepared


def answer_exposes_internal_field_keys(answer: str) -> bool:
    """True when user-facing text leaks snake_case metadata keys."""
    text = answer or ""
    if not text:
        return False
    # Explicit known keys or any snake_case label used as `key:`.
    if re.search(
        r"(?i)\b("
        + "|".join(re.escape(k) for k in sorted(KNOWN_INTERNAL_KEYS))
        + r")\s*:",
        text,
    ):
        return True
    for match in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\s*:", text):
        key = match.group(1)
        if key in KNOWN_INTERNAL_KEYS or key.count("_") >= 1:
            return True
    return False


def looks_like_answer_fragment(answer: str) -> bool:
    """Reject mid-sentence starts/ends and broken concatenations."""
    text = (answer or "").strip()
    if not text:
        return True
    # Strip citation markers for structural checks.
    plain = re.sub(r"\[\d+\]", "", text).strip()
    plain = re.sub(r"^Required items:\s*", "", plain, flags=re.I)
    plain = re.sub(r"^Applicable fees:\s*", "", plain, flags=re.I)
    # Bullet lists are allowed; check each bullet.
    bullets = [
        re.sub(r"^[-•*]\s*", "", line).strip()
        for line in plain.splitlines()
        if line.strip()
    ]
    if len(bullets) > 1:
        for bullet in bullets:
            if _is_fragment_clause(bullet):
                return True
        return False
    return _is_fragment_clause(plain)


def _is_fragment_clause(text: str) -> bool:
    clause = (text or "").strip()
    if not clause:
        return True
    # Begins mid-sentence / lowercase continuation.
    if re.match(
        r"^(?:and|or|but|because|which|that|who|when|where|with|without|to|of|for|in|on|at|by)\b",
        clause,
        re.I,
    ) and not re.match(r"(?i)^(?:and|or)\s+[A-Z0-9¥$€£]", clause):
        # "and My Number..." as a bullet starting with and is OK if capitalized after.
        if clause[0].islower():
            return True
    if clause[0].islower() and not clause.startswith(("•", "-", "¥", "$", "€", "£")):
        return True
    # Ends mid-sentence with dangling joiners / truncated words.
    if re.search(r"(?i)\b(?:and|or|with|for|to|of|the|a|an)\s*$", clause):
        return True
    if re.search(r"\s(?:fro|wi|th|yo|ho|me)$", clause):  # truncated tokens
        return True
    # Concatenation without punctuation / spacing issues.
    if re.search(r"[a-z]\s*[A-Z][a-z]+ [A-Z][a-z]+.*[a-z]\s*[A-Z]", clause) and "." not in clause:
        # Heuristic only for long jammed prose.
        if len(clause) > 160 and clause.count(".") == 0 and "\n" not in clause:
            return True
    return False


def extract_currency_amounts(text: str) -> set[str]:
    """Normalize fee amounts for completeness checks (350, 250, …)."""
    amounts: set[str] = set()
    for match in re.finditer(
        r"(?i)(?:[$¥€£]\s*(\d[\d,]*(?:\.\d+)?)|"
        r"(?:fee(?:[_\s-]?(?:jpy|usd|eur|gbp|amount|yen))?|price|cost|amount|salary_range|salary)"
        r"\s*[:=]\s*[^\d]*(\d[\d,]*(?:\.\d+)?)|"
        r"(?:USD|JPY|EUR|GBP|SGD|AUD)\s*(\d[\d,]*(?:\.\d+)?)|"
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:USD|JPY|EUR|GBP|SGD|AUD|dollars?|yen|euros?|pounds?)|"
        r"(\d{1,3}(?:\.\d+)?)\s*%)",
        text or "",
    ):
        number = next((g for g in match.groups() if g), None)
        if number:
            amounts.add(number.replace(",", "").lstrip("0") or "0")
    return amounts
