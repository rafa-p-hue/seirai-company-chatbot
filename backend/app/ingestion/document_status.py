"""Generic document status / archive / version detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from app.models.api import DocumentStatus, ExtractedPage

ARCHIVE_FILENAME_RE = re.compile(
    r"(?i)\b(archived?|old|previous|legacy|obsolete|superseded|deprecated)\b"
)
DRAFT_FILENAME_RE = re.compile(r"(?i)\b(draft|wip|temporary|temp)\b")
CURRENT_FILENAME_RE = re.compile(r"(?i)\b(current|latest|active|live)\b")
STATUS_VALUE_RE = re.compile(
    r"(?i)\b(status|document_status|record_status)\b"
)
EFFECTIVE_KEYS = {
    "effective_date",
    "effective",
    "valid_from",
    "validfrom",
    "as_of",
    "asof",
}
VERSION_KEYS = {"version", "ver", "revision", "rev"}
STATUS_KEYS = {"status", "document_status", "record_status", "state"}
VALID_TO_KEYS = {"valid_to", "validto", "expires", "expiry", "end_date"}
HISTORICAL_QUESTION_RE = re.compile(
    r"(?i)\b("
    r"old|previous|former|archived|legacy|obsolete|histor(?:y|ical)|"
    r"used to|before the change|prior|"
    r"in\s+(?:19|20)\d{2}|during\s+(?:19|20)\d{2}|"
    r"as\s+of\s+(?:19|20)\d{2}|fee\s+in\s+(?:19|20)\d{2}"
    r")\b"
)


@dataclass
class DocumentLifecycle:
    document_status: DocumentStatus = "unknown"
    effective_date: Optional[str] = None
    version: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    metadata: Dict[str, Any] | None = None


def detect_document_lifecycle(
    *,
    filename: str,
    pages: Sequence[ExtractedPage] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DocumentLifecycle:
    meta: Dict[str, Any] = dict(metadata or {})
    for page in pages or []:
        for key, value in (page.metadata or {}).items():
            meta.setdefault(key, value)
        for row in page.table_rows:
            if row.effective_date and "effective_date" not in meta:
                meta["effective_date"] = row.effective_date

    status = _status_from_meta(meta)
    if status == "unknown":
        if ARCHIVE_FILENAME_RE.search(filename or ""):
            status = "archived"
        elif DRAFT_FILENAME_RE.search(filename or ""):
            status = "draft"
        elif CURRENT_FILENAME_RE.search(filename or ""):
            status = "current"

    blob = "\n".join((page.text or "")[:400] for page in (pages or [])[:3])
    if status == "unknown":
        if re.search(r"(?i)\b(archived|obsolete|superseded|no longer valid)\b", blob):
            status = "archived"
        elif re.search(r"(?i)\b(draft|not final)\b", blob):
            status = "draft"
        elif re.search(r"(?i)\b(current|in effect|effective)\b", blob):
            status = "current"

    effective = _first_meta(meta, EFFECTIVE_KEYS) or _search_labeled(blob, "effective date")
    version = _first_meta(meta, VERSION_KEYS) or _search_labeled(blob, "version")
    valid_from = _first_meta(meta, {"valid_from", "validfrom"})
    valid_to = _first_meta(meta, VALID_TO_KEYS)
    if valid_to and status == "unknown":
        status = "archived"

    return DocumentLifecycle(
        document_status=status,
        effective_date=effective,
        version=version,
        valid_from=valid_from,
        valid_to=valid_to,
        metadata=meta,
    )


def question_wants_historical(question: str) -> bool:
    return bool(HISTORICAL_QUESTION_RE.search(question or ""))


def version_rank_boost(
    *,
    question: str,
    document_status: Optional[str],
    effective_date: Optional[str] = None,
) -> float:
    """Soft boost/penalty for current vs archived sources."""
    status = (document_status or "unknown").lower()
    historical = question_wants_historical(question)
    boost = 0.0
    if historical:
        if status == "archived":
            boost += 0.55
        elif status == "current":
            boost -= 0.15
    else:
        if status == "current":
            boost += 0.55
        elif status == "archived":
            boost -= 0.85
        elif status == "draft":
            boost -= 0.35
    if effective_date and not historical and status != "archived":
        boost += 0.08
    return boost


def _status_from_meta(meta: Mapping[str, Any]) -> DocumentStatus:
    for key, value in meta.items():
        if key.lower().replace(" ", "_") not in STATUS_KEYS and not STATUS_VALUE_RE.search(
            str(key)
        ):
            continue
        text = str(value or "").strip().lower()
        if text in {"archived", "archive", "old", "legacy", "obsolete", "superseded"}:
            return "archived"
        if text in {"draft", "wip"}:
            return "draft"
        if text in {"current", "active", "live", "latest"}:
            return "current"
    return "unknown"


def _first_meta(meta: Mapping[str, Any], keys: Set[str]) -> Optional[str]:
    for key, value in meta.items():
        normalized = re.sub(r"[\s-]+", "_", str(key).strip().lower())
        if normalized in keys and value not in (None, ""):
            return str(value).strip()
    return None


def _search_labeled(text: str, label: str) -> Optional[str]:
    match = re.search(
        rf"(?im)^{re.escape(label)}\s*:\s*(.+)$",
        text or "",
    )
    if match:
        return match.group(1).strip()
    return None
