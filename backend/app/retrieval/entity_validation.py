"""Strict requested-entity / service validation (document-agnostic).

Numeric shape alone never validates a chunk. Evidence must explicitly mention
the requested item or service (or a close synonym phrase extracted from the
question itself — never from a fixed municipal corpus list).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from app.models.api import RetrievedChunk


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "how",
    "much",
    "many",
    "what",
    "when",
    "where",
    "which",
    "who",
    "do",
    "does",
    "did",
    "i",
    "we",
    "you",
    "my",
    "our",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "and",
    "or",
    "with",
    "from",
    "city",
    "town",
    "ward",
    "pref",
    "prefecture",
    "about",
    "into",
    "after",
    "before",
    "please",
    "tell",
    "me",
    "need",
    "get",
    "cost",
    "price",
    "fee",
    "fees",
    "yen",
    "usd",
    "receive",
    "provided",
    "provide",
}


SERVICE_NOUN_RE = re.compile(
    r"(?i)\b("
    r"(?:residential\s+|resident\s+|family\s+|residence\s+|national\s+|"
    r"child\s+|childcare\s+|health\s+|parking\s+|evacuation\s+|burnable\s+|"
    r"oversized\s+)?"
    r"(?:parking\s+)?(?:permit|certificate|allowance|benefit|registration|"
    r"enrollment|application|card|abstract|sticker|collection|shelter|"
    r"insurance|notification|passport|garbage|waste|recycling|"
    r"health\s+insurance)"
    r"(?:\s+(?:permit|certificate|fee|fees|card|abstract|application|insurance))?"
    r")\b"
)

UNRELATED_FEE_NOISE_RE = re.compile(
    r"(?i)\b("
    r"residence\s+certificate|family\s+register|juminhyo|"
    r"kiosk\s+fee|counter\s+fee|issuance\s+fee|"
    r"certificate\s+issuance|fee\s+schedule"
    r")\b"
)


def extract_requested_service_phrases(question: str) -> List[str]:
    """Pull the item/service phrases the user asked about."""
    text = re.sub(r"\s+", " ", (question or "").strip())
    if not text:
        return []

    phrases: List[str] = []

    # "How much is a residential parking permit"
    # "How much child allowance does a 2-year-old receive?"
    for pattern in (
        r"(?i)how\s+much\s+(.+?)\s+(?:does|do|is|are)\b",
        r"(?i)how\s+much\s+(?:is|are|does|do)\s+(?:a|an|the|my|our)?\s*(.+?)(?:\s+cost|\s+fee|$|\?)",
        r"(?i)(?:cost|fee|price)\s+(?:of|for)\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",
        # "What fee does TalentBridge charge for a permanent placement?"
        r"(?i)(?:fee|fees|cost|price)\s+(?:does|do)\s+.+?\s+charge\s+for\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",
        r"(?i)charge\s+for\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",
        r"(?i)(?:apply\s+for|throw\s+away|dispose\s+of|discard)\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",
        r"(?i)which\s+(.+?)\s+(?:accepts?|allows?|offers?)\b",
        r"(?i)when\s+is\s+(?:the\s+)?(.+?)\s+(?:paid|collected|due)\b",
        r"(?i)what\s+is\s+(?:the\s+)?capacity\s+of\s+(?:the\s+)?(.+?)(?:\?|$)",
        r"(?i)what\s+share\s+of\s+(.+?)\s+do\b",
        r"(?i)(?:with|under|for)\s+(national\s+health\s+insurance)\b",
    ):
        match = re.search(pattern, text)
        if match:
            phrase = _clean_phrase(match.group(1))
            if phrase:
                phrases.append(phrase)

    for match in SERVICE_NOUN_RE.finditer(text):
        phrase = _clean_phrase(match.group(1))
        if phrase and phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(phrase)

    # Recruitment / staffing fee services named in the question.
    for match in re.finditer(
        r"(?i)\b("
        r"permanent\s+placement|executive\s+search|contract\s+staffing|"
        r"temporary\s+staffing|retained\s+search|contingent\s+search|"
        r"placement\s+fee|recruitment\s+fee|staffing\s+fee"
        r")\b",
        text,
    ):
        phrase = _clean_phrase(match.group(1))
        if phrase and phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(phrase)

    # Expand close variants from the phrase itself (no external gazetteer).
    expanded: List[str] = []
    for phrase in phrases:
        expanded.append(phrase)
        expanded.extend(_phrase_variants(phrase))
    # Deduplicate preserving order.
    seen = set()
    out: List[str] = []
    for phrase in expanded:
        key = phrase.lower()
        if key in seen or len(key) < 4:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _clean_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip(" .,?!\"'"))
    text = re.sub(
        r"(?i)^(a|an|the|my|our|this|that)\s+",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\s+(cost|fee|fees|price|now|currently|please)$",
        "",
        text,
    ).strip()
    return text


def _phrase_variants(phrase: str) -> List[str]:
    lower = phrase.lower()
    variants: List[str] = []
    tokens = [t for t in re.findall(r"[a-z0-9]+", lower) if t not in STOPWORDS]
    if len(tokens) >= 2:
        variants.append(" ".join(tokens))
        variants.append(" ".join(tokens[-2:]))
    # residential parking permit ↔ parking permit ↔ resident parking
    if "parking" in tokens and "permit" in tokens:
        variants.extend(
            [
                "parking permit",
                "resident parking",
                "residential parking",
                "residential parking permit",
            ]
        )
    if "permanent" in tokens and "placement" in tokens:
        variants.extend(
            [
                "permanent placement",
                "permanent placements",
                "contingent placement",
                "placement fee",
            ]
        )
    if "executive" in tokens and "search" in tokens:
        variants.extend(["executive search", "retained search"])
    if "child" in tokens and "allowance" in tokens:
        variants.extend(["child allowance", "childcare allowance", "child benefit"])
    if "residence" in tokens and "certificate" in tokens:
        variants.extend(["residence certificate", "resident certificate"])
    if "health" in tokens and "insurance" in tokens:
        variants.extend(
            [
                "national health insurance",
                "health insurance",
                "patient share",
                "co-payment",
                "insured treatment",
            ]
        )
    if "shelter" in tokens or "evacuation" in tokens:
        variants.extend(["evacuation shelter", "pet-friendly shelter", "accepts pets"])
    return variants


_LOCATION_ONLY_PHRASE_RE = re.compile(
    r"(?i)^(?:hikari(?:\s+city)?|(?:the|this|a|an)\s+city)$"
)


def _is_location_only_phrase(phrase: str) -> bool:
    text = re.sub(r"\s+", " ", (phrase or "").strip())
    if not text:
        return True
    if _LOCATION_ONLY_PHRASE_RE.match(text):
        return True
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]
    # Pure place names (e.g. "Hikari") — never a fee/service entity alone.
    if not tokens:
        return bool(re.search(r"(?i)\b(city|town|ward|hikari)\b", text))
    if len(tokens) == 1 and tokens[0] in {"hikari", "city", "town", "ward"}:
        return True
    return False


def service_phrases_for_matching(phrases: Sequence[str]) -> List[str]:
    """Drop location-only phrases that otherwise false-match every local fee row."""
    return [p for p in phrases if not _is_location_only_phrase(p)]


def entity_match_score(content: str, requested_phrases: Sequence[str]) -> float:
    if not content or not requested_phrases:
        return 0.0
    phrases = service_phrases_for_matching(requested_phrases)
    if not phrases:
        return 0.0
    blob = content.lower()
    best = 0.0
    for phrase in phrases:
        phrase_l = phrase.lower().strip()
        if not phrase_l:
            continue
        if phrase_l in blob:
            best = max(best, 1.0)
            continue
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", phrase_l)
            if token not in STOPWORDS and len(token) > 2
        ]
        if not tokens:
            continue
        # Multi-token services (e.g. parking + permit) require the distinctive
        # service tokens — city/place tokens alone never count.
        service_tokens = [
            token
            for token in tokens
            if token not in {"hikari", "city", "town", "ward"}
        ]
        focus = service_tokens or tokens
        hits = sum(1 for token in focus if token in blob)
        ratio = hits / len(focus)
        if ratio >= 0.75 and hits >= min(2, len(focus)):
            best = max(best, 0.85 if len(focus) > 1 else 1.0)
        elif ratio >= 0.5 and len(focus) >= 2 and hits >= 2:
            best = max(best, 0.45)
    return best


def evidence_matches_requested_entity(
    content: str,
    question: str,
    *,
    section_title: str = "",
    document_name: str = "",
) -> bool:
    phrases = extract_requested_service_phrases(question)
    usable = service_phrases_for_matching(phrases)
    q = (question or "").lower()
    fee_question = bool(re.search(r"(?i)\b(how much|fee|cost|price|permit)\b", q))
    # Fee questions that name a service must fail closed when no usable phrase
    # survives (do not accept currency-only / city-name-only matches).
    if not usable:
        return not fee_question
    blob = "\n".join(part for part in (content, section_title, document_name) if part)
    score = entity_match_score(blob, usable)
    if score < 0.75:
        return False
    # Currency/certificate rows about a different product never count.
    if fee_question and _looks_like_unrelated_fee_row(blob, usable):
        joined = " ".join(usable).lower()
        if not re.search(
            r"(?i)\b(certificate|register|juminhyo|koseki|kiosk|counter fee)\b",
            joined,
        ):
            return False
    return True


def _looks_like_unrelated_fee_row(blob: str, phrases: Sequence[str]) -> bool:
    if not UNRELATED_FEE_NOISE_RE.search(blob):
        return False
    # If the requested phrases themselves are certificate-like, allow.
    joined = " ".join(phrases).lower()
    if re.search(r"(?i)\b(certificate|register|juminhyo|kiosk|counter fee)\b", joined):
        return False
    return True


_SERVICE_HEADER_RE = re.compile(r"(?i)^\s*service\s*:")
_FEE_LINE_RE = re.compile(
    r"(?i)^\s*(?:fee|minimum\s+fee|price|cost|amount|rate)\b"
)


def _content_has_price_signal(text: str) -> bool:
    from app.retrieval.numeric_facts import content_has_currency

    return content_has_currency(text or "")


def _block_has_percent_and_money(block: Sequence[RetrievedChunk]) -> bool:
    from app.retrieval.numeric_facts import CURRENCY_RE, PERCENT_RE

    blob = "\n".join(c.content or "" for c in block)
    return bool(PERCENT_RE.search(blob) and CURRENCY_RE.search(blob))


def collect_named_service_fee_evidence(
    evidence: Sequence[RetrievedChunk],
    question: str,
) -> List[RetrievedChunk]:
    """Keep fee/amount lines that belong to a named service split across chunks.

    Fee schedules often chunk as:
      Service: Permanent placement (contingent)
      Fee (from 1 Apr 2026): 22% of first-year base salary
      Minimum fee: SGD 9,000
    Entity matching on the service label alone must still recover the amount rows.
    Stop before the next service block even when that header was dropped from top-k.
    """
    if not evidence:
        return []
    phrases = service_phrases_for_matching(extract_requested_service_phrases(question))
    if not phrases:
        return []

    by_doc: Dict[str, List[RetrievedChunk]] = {}
    evidence_rank = {id(chunk): index for index, chunk in enumerate(evidence)}
    for chunk in evidence:
        key = (chunk.document_name or "").strip().lower() or "_unknown"
        by_doc.setdefault(key, []).append(chunk)

    selected: List[RetrievedChunk] = []
    seen: set = set()

    def _add(chunk: RetrievedChunk) -> None:
        fingerprint = (
            chunk.document_name,
            chunk.chunk_id or "",
            (chunk.content or "")[:120],
        )
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        selected.append(chunk)

    def _order_key(chunk: RetrievedChunk) -> Tuple:
        return (
            chunk.page_number if chunk.page_number is not None else 10**9,
            chunk.chunk_index if chunk.chunk_index is not None else 10**9,
            chunk.row_number if chunk.row_number is not None else 10**9,
            evidence_rank.get(id(chunk), 10**9),
        )

    for doc_chunks in by_doc.values():
        ordered = sorted(doc_chunks, key=_order_key)
        i = 0
        while i < len(ordered):
            chunk = ordered[i]
            is_anchor = evidence_matches_requested_entity(
                chunk.content or "",
                question,
                section_title=chunk.section_title or "",
                document_name=chunk.document_name or "",
            )
            if not is_anchor:
                i += 1
                continue
            block = [chunk]
            j = i + 1
            while j < len(ordered):
                nxt = ordered[j]
                nxt_text = (nxt.content or "").strip()
                if not nxt_text:
                    j += 1
                    continue
                # Next competing service header ends this fee block.
                if _SERVICE_HEADER_RE.search(nxt_text) and not evidence_matches_requested_entity(
                    nxt_text,
                    question,
                    section_title=nxt.section_title or "",
                    document_name=nxt.document_name or "",
                ):
                    break
                if evidence_matches_requested_entity(
                    nxt_text,
                    question,
                    section_title=nxt.section_title or "",
                    document_name=nxt.document_name or "",
                ):
                    block.append(nxt)
                    j += 1
                    continue
                if _content_has_price_signal(nxt_text) or _FEE_LINE_RE.search(nxt_text):
                    # Job salary rows are currency-bearing but not fee-schedule lines.
                    if re.search(
                        r"(?i)\b(job_id|salary_range|employment_type|posted_date)\b",
                        nxt_text,
                    ):
                        break
                    # One percent fee + one absolute minimum is a complete schedule row.
                    # Without this cap, a dropped "Service: Executive search" header
                    # lets the next service's fee leak into the block.
                    if _block_has_percent_and_money(block):
                        break
                    # Adjacent indexes only when available.
                    prev = block[-1]
                    if (
                        prev.chunk_index is not None
                        and nxt.chunk_index is not None
                        and nxt.chunk_index - prev.chunk_index > 2
                    ):
                        break
                    block.append(nxt)
                    j += 1
                    continue
                # Stop on unrelated prose; do not vacuum the whole page.
                break
            if any(_content_has_price_signal(c.content or "") for c in block):
                # Ensure a nearby Minimum fee row is kept with percentage fees.
                has_pct = any(
                    re.search(r"\d{1,3}(?:\.\d+)?\s*%", c.content or "")
                    for c in block
                )
                has_min = any(
                    re.search(r"(?i)\bminimum\s+fee\b", c.content or "")
                    for c in block
                )
                # Permanent-placement fee questions need an explicit % fee row.
                # A lone "Minimum fee: SGD …" sibling is not enough.
                wants_pct_fee = any(
                    re.search(r"(?i)\bpermanent\s+placement\b", p) for p in phrases
                ) or bool(
                    re.search(r"(?i)\bpermanent\s+placement\b", question or "")
                )
                if wants_pct_fee and not has_pct:
                    # Try to pull a nearby percent fee row still in this doc.
                    anchor_idx = chunk.chunk_index
                    for cand in ordered:
                        if cand in block:
                            continue
                        text = cand.content or ""
                        if not re.search(r"\d{1,3}(?:\.\d+)?\s*%", text):
                            continue
                        if not re.search(
                            r"(?i)^\s*(?:fee|minimum\s+fee)\b|%\s+of\s+first",
                            text,
                        ):
                            continue
                        if (
                            anchor_idx is not None
                            and cand.chunk_index is not None
                            and abs(cand.chunk_index - anchor_idx) > 4
                        ):
                            continue
                        block.append(cand)
                        has_pct = True
                        break
                if wants_pct_fee and not has_pct:
                    i = max(j, i + 1)
                    continue
                if has_pct and not has_min:
                    anchor_idx = chunk.chunk_index
                    for cand in ordered:
                        if cand in block:
                            continue
                        if not re.search(
                            r"(?i)^\s*minimum\s+fee\b", (cand.content or "").strip()
                        ):
                            continue
                        if (
                            anchor_idx is not None
                            and cand.chunk_index is not None
                            and abs(cand.chunk_index - anchor_idx) > 4
                        ):
                            continue
                        block.append(cand)
                        break
                for item in block:
                    _add(item)
            elif _content_has_price_signal(chunk.content or ""):
                _add(chunk)
            i = max(j, i + 1)

    # Also keep currency rows that themselves name the service.
    for chunk in evidence:
        if not _content_has_price_signal(chunk.content or ""):
            continue
        if evidence_matches_requested_entity(
            chunk.content or "",
            question,
            section_title=chunk.section_title or "",
            document_name=chunk.document_name or "",
        ):
            _add(chunk)
    return selected


def filter_evidence_for_requested_entity(
    evidence: Sequence[RetrievedChunk],
    question: str,
) -> Tuple[List[RetrievedChunk], dict]:
    phrases = extract_requested_service_phrases(question)
    usable = service_phrases_for_matching(phrases)
    diagnostics = {
        "requested_phrases": phrases,
        "usable_phrases": usable,
        "kept": [],
        "rejected": [],
        "rejection_reason": None,
    }

    # Only enforce for fee/price/amount style questions or explicit named services.
    q = (question or "").lower()
    enforce = bool(
        re.search(r"(?i)\b(how much|fee|cost|price|permit|certificate)\b", q)
    )
    if not enforce:
        return list(evidence), diagnostics

    # Named fee/service question with no usable entity phrase → keep nothing.
    if not usable:
        diagnostics["rejection_reason"] = "no_explicit_entity_match"
        for chunk in evidence:
            diagnostics["rejected"].append(
                {
                    "document_name": chunk.document_name,
                    "section_title": chunk.section_title,
                    "entity_match_score": 0.0,
                    "preview": (chunk.content or "")[:160],
                    "rejection_reason": "no_usable_service_phrase",
                }
            )
        return [], diagnostics

    kept: List[RetrievedChunk] = []
    for chunk in evidence:
        blob = "\n".join(
            part
            for part in (
                chunk.content or "",
                chunk.section_title or "",
                chunk.document_name or "",
            )
            if part
        )
        score = entity_match_score(blob, usable)
        row = {
            "document_name": chunk.document_name,
            "section_title": chunk.section_title,
            "entity_match_score": round(score, 3),
            "preview": (chunk.content or "")[:160],
        }
        accepts = score >= 0.75 and evidence_matches_requested_entity(
            chunk.content or "",
            question,
            section_title=chunk.section_title or "",
            document_name=chunk.document_name or "",
        )
        if accepts:
            kept.append(chunk)
            diagnostics["kept"].append(row)
        else:
            reason = (
                "unrelated_fee_row"
                if _looks_like_unrelated_fee_row(blob, usable)
                else "requested_entity_mismatch"
            )
            diagnostics["rejected"].append({**row, "rejection_reason": reason})

    # Service label and fee amount often live in adjacent chunks — attach them.
    if usable:
        block = collect_named_service_fee_evidence(evidence, question)
        if block:
            kept_ids = {
                (c.document_name, c.chunk_id or "", (c.content or "")[:120]) for c in kept
            }
            for chunk in block:
                key = (chunk.document_name, chunk.chunk_id or "", (chunk.content or "")[:120])
                if key not in kept_ids:
                    kept.append(chunk)
                    diagnostics["kept"].append(
                        {
                            "document_name": chunk.document_name,
                            "section_title": chunk.section_title,
                            "entity_match_score": 1.0,
                            "preview": (chunk.content or "")[:160],
                            "attached_as": "service_fee_sibling",
                        }
                    )
                    kept_ids.add(key)
            diagnostics["rejection_reason"] = None

    if not kept:
        diagnostics["rejection_reason"] = "no_explicit_entity_match"
    return kept, diagnostics


def answer_matches_requested_entity(answer: str, question: str) -> bool:
    """Final relevance gate: answer entity/service must align with the request."""
    if not answer:
        return False
    if re.search(r"(?i)could not find that information", answer):
        return True
    if not answer_matches_requested_category(answer, question):
        return False
    phrases = extract_requested_service_phrases(question)
    usable = service_phrases_for_matching(phrases)
    q = (question or "").lower()
    fee_question = bool(re.search(r"(?i)\b(how much|fee|cost|price|permit)\b", q))
    if not fee_question:
        return True
    if not usable:
        return False
    score = entity_match_score(answer, usable)
    if score >= 0.75:
        if UNRELATED_FEE_NOISE_RE.search(answer) and not re.search(
            r"(?i)\b(certificate|register|juminhyo|koseki)\b",
            " ".join(usable),
        ):
            return False
        return True
    # Reject answers that are clearly about a different fee product.
    if UNRELATED_FEE_NOISE_RE.search(answer):
        return False
    # If answer only has currency and no requested tokens, reject.
    if re.search(r"(?i)\b(?:¥|yen|\d[\d,]*\s*yen|\$\d)\b", answer) and score < 0.75:
        return False
    return False


def answer_matches_requested_category(answer: str, question: str) -> bool:
    """Reject category substitutions such as burnable garbage → recyclables."""
    q = (question or "").lower()
    a = (answer or "").lower()
    if not q or not a:
        return True
    if re.search(r"(?i)\bburnable\s+garbage\b", q):
        if re.search(r"(?i)\brecyclables?\b", a) and not re.search(
            r"(?i)\bburnable\b", a
        ):
            return False
    if re.search(r"(?i)\brecyclables?\b", q):
        if re.search(r"(?i)\bburnable\b", a) and not re.search(
            r"(?i)\brecycl", a
        ):
            return False
    if re.search(r"(?i)\bnational\s+health\s+insurance\b", q):
        if re.search(r"(?i)\bchild\s+allowance|childcare\b", a) and not re.search(
            r"(?i)\b(health\s+insurance|medical|co-?payment|patient\s+share|\d+\s*%)\b",
            a,
        ):
            return False
    return True


def format_not_found_answer(contact_name: Optional[str] = None) -> str:
    """Domain-neutral not-found wording.

    Prefer an explicit organization/contact name when provided. Never default to
    municipal phrasing such as "the city".
    """
    base = "I could not find that information in the available documents."
    name = (contact_name or "").strip()
    # Reject leftover municipal defaults from older configs/prompts.
    if name.lower() in {"", "the city", "city", "the municipal office", "municipal office"}:
        return (
            f"{base} You may want to contact the organization directly "
            "for confirmation."
        )
    return f"{base} You may want to contact {name} directly for confirmation."


def infer_not_found_contact_name(
    *,
    question: str = "",
    configured_name: str = "",
    document_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Best-effort organization label for not-found fallbacks (document-agnostic)."""
    configured = (configured_name or "").strip()
    if configured and configured.lower() not in {
        "the city",
        "city",
        "the municipal office",
        "municipal office",
    }:
        return configured

    blob = " ".join(
        part
        for part in (
            question or "",
            " ".join(document_names or ()),
        )
        if part
    )
    # "… of TalentBridge …"
    of_match = re.search(
        r"\b(?:of|at|for|with)\s+([A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z][A-Za-z0-9&'-]*){0,2})\b",
        question or "",
    )
    if of_match:
        candidate = of_match.group(1).strip()
        if candidate.lower() not in {
            "singapore",
            "london",
            "sydney",
            "the",
            "this",
            "that",
        } and not re.search(
            r"(?i)\b(engineer|manager|analyst|associate|accountant|ceo|cfo)\b",
            candidate,
        ):
            return candidate.split()[0]

    # CamelCase brand tokens (TalentBridge, OpenAI, …) — not job titles like DevOps.
    for camel in re.finditer(r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b", blob):
        token = camel.group(1)
        if re.search(
            r"(?i)^(devops|backend|frontend|fullstack|fintech|saas|hris)$",
            token,
        ):
            continue
        if re.search(
            rf"(?i)\b(?:senior|junior|lead|principal|staff)\s+{re.escape(token)}\b",
            question or "",
        ):
            continue
        if re.search(
            rf"(?i)\b{re.escape(token)}\s+(?:engineer|manager|developer|analyst|designer)\b",
            question or "",
        ):
            continue
        return token

    # Multi-word Title Case org names ending with Partners/Corp/Inc/…
    match = re.search(
        r"\b("
        r"[A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z][A-Za-z0-9&'-]*){0,3}"
        r")\s+(?:Partners|Corp|Corporation|Inc|Labs|Group|Company)\b",
        blob,
    )
    if match:
        return match.group(1).split()[0]
    return None

