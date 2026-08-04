"""Assemble complete checklist evidence from same-section sibling chunks.

Retrieval often returns only the first requirement fragment (e.g. residence
card / passport). Later bullets in the same section live in neighboring
chunks that lack "bring/required" wording, so they are dropped. This module
loads those siblings in source order until the requirement list is complete.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LIST_MARKER_START_RE = re.compile(r"^\s*(?:[-•●▪◦·]|\d+[.)])\s+")
CONTINUATION_END_RE = re.compile(
    r"(?i)(?:,\s*(?:and\s+|or\s+)?|\band\b\s*|\bor\b\s*|"
    r"bring(?: the following)?\s*:?)\s*$"
)
REQUIREMENT_INTRO_RE = re.compile(
    r"(?i)\b("
    r"bring(?: the following)?|"
    r"required (?:documents?|items?|materials?)|"
    r"documents? (?:needed|required)|"
    r"you (?:will )?need"
    r")\b"
)
DOCUMENT_OBJECT_RE = re.compile(
    r"(?i)\b("
    r"passports?|cards?|forms?|documents?|identification|id\b|bills?|consent|"
    r"licenses?|certificates?|photos?|proof|agreements?|applications?|"
    r"seal|my\s+number|notification\s+cards?"
    r")\b"
)
UNRELATED_SECTION_RE = re.compile(
    r"(?i)\b("
    r"office hours?|opening hours?|business hours?|closures?|closed|"
    r"contact information|phone numbers?|email addresses?|directions?|"
    r"certificates and fees|fee schedule"
    r")\b"
)


def requirement_list_looks_incomplete(text: str) -> bool:
    """True when a chunk looks like a truncated required-item list."""
    body = (text or "").strip()
    if not body:
        return True
    if CONTINUATION_END_RE.search(body):
        return True
    if REQUIREMENT_INTRO_RE.search(body):
        bullets = [
            line
            for line in body.splitlines()
            if LIST_MARKER_START_RE.match(line)
            and DOCUMENT_OBJECT_RE.search(line)
        ]
        # Intro present with zero/one item is usually a split list.
        if len(bullets) <= 1:
            return True
    # Bare continuation bullet(s) without intro.
    if LIST_MARKER_START_RE.match(body) and DOCUMENT_OBJECT_RE.search(body):
        if not REQUIREMENT_INTRO_RE.search(body):
            return True
    return False


def is_requirement_sibling_payload(payload: Dict[str, Any]) -> bool:
    """Same-section neighbor that can complete a checklist."""
    content = str(payload.get("content") or "")
    heading = str(payload.get("section_title") or "")
    subsection = str(payload.get("subsection_title") or "")
    ctype = str(payload.get("content_type") or "")
    blob = f"{heading}\n{subsection}\n{content}"
    if ctype == "heading" and len(content.split()) <= 6:
        return False
    if UNRELATED_SECTION_RE.search(heading) or UNRELATED_SECTION_RE.search(subsection):
        return False
    if UNRELATED_SECTION_RE.search(content) and not DOCUMENT_OBJECT_RE.search(content):
        return False
    if REQUIREMENT_INTRO_RE.search(blob):
        return True
    if DOCUMENT_OBJECT_RE.search(content) and (
        LIST_MARKER_START_RE.match(content.strip())
        or ctype in {"list", "numbered_list", "key_value"}
        or REQUIREMENT_INTRO_RE.search(heading)
        # Mid-sentence continuation of a bring list ("permission), your Moving-Out…").
        or content[:1].islower()
        or content.startswith((")", ",", "and ", "or "))
    ):
        return True
    return False


def _payload_key(payload: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(payload.get("document_id") or payload.get("document_name") or "")
        .strip()
        .lower(),
        str(payload.get("section_title") or "").strip().lower(),
        int(payload.get("chunk_index") or 0),
    )


def _item_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return item.get("payload") or {}


def section_chunks_diagnostic(
    payloads: Sequence[Dict[str, Any]],
    *,
    section_title: Optional[str] = None,
    document_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ordered section chunk diagnostics with synthetic offsets and neighbor ids."""
    rows = []
    for payload in payloads:
        if document_name and str(payload.get("document_name") or "").lower() != document_name.lower():
            continue
        if section_title and str(payload.get("section_title") or "").strip().lower() != (
            section_title or ""
        ).strip().lower():
            # Also accept nested heading text inside content for fused sections.
            content = str(payload.get("content") or "")
            if section_title and section_title.lower() not in content.lower():
                if str(payload.get("section_title") or "").strip().lower() != (
                    section_title or ""
                ).strip().lower():
                    continue
        rows.append(payload)
    rows = sorted(rows, key=lambda p: int(p.get("chunk_index") or 0))
    offset = 0
    out: List[Dict[str, Any]] = []
    for index, payload in enumerate(rows):
        content = str(payload.get("content") or "")
        start = offset
        end = offset + len(content)
        offset = end + 1
        prev_id = rows[index - 1].get("chunk_id") if index > 0 else None
        next_id = rows[index + 1].get("chunk_id") if index + 1 < len(rows) else None
        out.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "document_name": payload.get("document_name"),
                "section_title": payload.get("section_title"),
                "chunk_index": int(payload.get("chunk_index") or 0),
                "content_type": payload.get("content_type"),
                "start_offset": start,
                "end_offset": end,
                "token_count": len(content.split()),
                "clean_text": content,
                "previous_chunk_id": prev_id,
                "next_chunk_id": next_id,
            }
        )
    return out


def expand_checklist_siblings(
    selected_items: Sequence[Dict[str, Any]],
    all_payloads: Sequence[Dict[str, Any]],
    *,
    limit: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Include same-section adjacent requirement chunks in source order.

    Returns (expanded_items, diagnostics).
    """
    diagnostics: Dict[str, Any] = {
        "selected_before": [],
        "incomplete_selected": False,
        "siblings_added": [],
        "section_chunks": [],
        "merged_payload_previews": [],
    }
    if not selected_items:
        return list(selected_items), diagnostics

    selected_payloads = [_item_payload(item) for item in selected_items]
    diagnostics["selected_before"] = [
        {
            "chunk_id": p.get("chunk_id"),
            "chunk_index": p.get("chunk_index"),
            "section_title": p.get("section_title"),
            "preview": " ".join(str(p.get("content") or "").split())[:180],
            "incomplete": requirement_list_looks_incomplete(str(p.get("content") or "")),
        }
        for p in selected_payloads
    ]
    diagnostics["incomplete_selected"] = any(
        row["incomplete"] for row in diagnostics["selected_before"]
    )

    # Anchor on the strongest requirement / selected section.
    anchors: List[Dict[str, Any]] = []
    for payload in selected_payloads:
        if not payload:
            continue
        anchors.append(payload)
    if not anchors:
        return list(selected_items), diagnostics

    # Expand each anchor's document+section by walking chunk_index neighbors.
    by_doc_section: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for payload in all_payloads:
        key = (
            str(payload.get("document_id") or payload.get("document_name") or "")
            .strip()
            .lower(),
            str(payload.get("section_title") or "").strip().lower(),
        )
        by_doc_section.setdefault(key, []).append(payload)
    for key in by_doc_section:
        by_doc_section[key].sort(key=lambda p: int(p.get("chunk_index") or 0))

    selected_ids = {
        str(p.get("chunk_id") or id(p)) for p in selected_payloads if p is not None
    }
    added_payloads: List[Dict[str, Any]] = []

    for anchor in anchors:
        doc_key = str(
            anchor.get("document_id") or anchor.get("document_name") or ""
        ).strip().lower()
        section_key = str(anchor.get("section_title") or "").strip().lower()
        neighbors = by_doc_section.get((doc_key, section_key), [])
        # When the selected requirement is truncated, also walk same-document
        # neighbors by chunk_index — PDF section titles often drift ("City Hall
        # Hours") and hide the true continuation bullet.
        same_doc = [
            p
            for p in all_payloads
            if str(p.get("document_id") or p.get("document_name") or "")
            .strip()
            .lower()
            == doc_key
        ]
        same_doc.sort(key=lambda p: int(p.get("chunk_index") or 0))
        if not neighbors and doc_key:
            neighbors = same_doc
        elif diagnostics["incomplete_selected"] and same_doc:
            # Merge unique same-doc payloads into the neighbor walk order.
            seen_n = {str(p.get("chunk_id") or id(p)) for p in neighbors}
            for payload in same_doc:
                cid = str(payload.get("chunk_id") or id(payload))
                if cid not in seen_n:
                    neighbors.append(payload)
                    seen_n.add(cid)
            neighbors.sort(key=lambda p: int(p.get("chunk_index") or 0))

        diagnostics["section_chunks"] = section_chunks_diagnostic(
            [
                p
                for p in (neighbors or [])
                if is_requirement_sibling_payload(p)
                or str(p.get("chunk_id")) == str(anchor.get("chunk_id"))
            ],
            section_title=anchor.get("section_title"),
            document_name=anchor.get("document_name"),
        )

        if not neighbors:
            continue

        # Find anchor position; walk backward/forward while siblings continue.
        anchor_index = None
        anchor_chunk_id = anchor.get("chunk_id")
        anchor_chunk_index = int(anchor.get("chunk_index") or -1)
        for i, payload in enumerate(neighbors):
            if anchor_chunk_id and payload.get("chunk_id") == anchor_chunk_id:
                anchor_index = i
                break
            if int(payload.get("chunk_index") or -2) == anchor_chunk_index:
                anchor_index = i
                break
        if anchor_index is None:
            # Still pull requirement siblings from the section.
            for payload in neighbors:
                cid = str(payload.get("chunk_id") or id(payload))
                if cid in selected_ids:
                    continue
                if is_requirement_sibling_payload(payload):
                    added_payloads.append(payload)
                    selected_ids.add(cid)
            continue

        # Walk backward for leading intro / prior list parts.
        for i in range(anchor_index - 1, -1, -1):
            payload = neighbors[i]
            if not is_requirement_sibling_payload(payload):
                break
            cid = str(payload.get("chunk_id") or id(payload))
            if cid in selected_ids:
                continue
            added_payloads.append(payload)
            selected_ids.add(cid)

        # Walk forward; skip non-requirement noise (hours/fees) instead of
        # hard-stopping, so trailing document bullets still merge.
        for i in range(anchor_index + 1, len(neighbors)):
            payload = neighbors[i]
            if not is_requirement_sibling_payload(payload):
                content = str(payload.get("content") or "")
                # Hard-stop only on clear section breaks / unrelated domains.
                if re.search(
                    r"(?i)\b(office\s+hours|fee\s+schedule|contact\s+us|"
                    r"moving\s+out|garbage|recycling)\b",
                    f"{payload.get('section_title') or ''}\n{content[:80]}",
                ):
                    break
                continue
            cid = str(payload.get("chunk_id") or id(payload))
            if cid in selected_ids:
                continue
            added_payloads.append(payload)
            selected_ids.add(cid)
            preview = "\n".join(
                str(p.get("content") or "")
                for p in ([anchor] + added_payloads)
            )
            if not requirement_list_looks_incomplete(preview):
                nxt = neighbors[i]
                if not LIST_MARKER_START_RE.match(str(nxt.get("content") or "").strip()):
                    pass

    diagnostics["siblings_added"] = [
        {
            "chunk_id": p.get("chunk_id"),
            "chunk_index": p.get("chunk_index"),
            "section_title": p.get("section_title"),
            "preview": " ".join(str(p.get("content") or "").split())[:180],
        }
        for p in added_payloads
    ]

    if not added_payloads:
        # Even without new siblings, return selected ordered by chunk_index.
        ordered = sorted(
            selected_items,
            key=lambda item: int(_item_payload(item).get("chunk_index") or 0),
        )
        diagnostics["merged_payload_previews"] = [
            " ".join(str(_item_payload(item).get("content") or "").split())[:220]
            for item in ordered
        ]
        return ordered[:limit], diagnostics

    # Build expanded item list: selected + siblings as retrieval-shaped dicts.
    expanded: List[Dict[str, Any]] = list(selected_items)
    for payload in added_payloads:
        expanded.append(
            {
                "payload": payload,
                "dense": 0.0,
                "lexical": 0.0,
                "combined": float(payload.get("score") or 0.0),
                "rerank": float(payload.get("score") or 0.0),
                "checklist_sibling": True,
            }
        )
    expanded = sorted(
        expanded,
        key=lambda item: int(_item_payload(item).get("chunk_index") or 0),
    )
    # Dedupe by chunk_id.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in expanded:
        payload = _item_payload(item)
        cid = str(payload.get("chunk_id") or id(payload))
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(item)
        if len(deduped) >= limit:
            break

    diagnostics["merged_payload_previews"] = [
        " ".join(str(_item_payload(item).get("content") or "").split())[:220]
        for item in deduped
    ]
    logger.info(
        "checklist_sibling_assembly selected=%s added=%s incomplete=%s",
        len(selected_items),
        len(added_payloads),
        diagnostics["incomplete_selected"],
    )
    return deduped, diagnostics


def merged_checklist_text(items: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for item in items:
        content = str(_item_payload(item).get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)
