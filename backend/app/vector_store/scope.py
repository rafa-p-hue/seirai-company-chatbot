"""Document scope helpers for company vs chat-scoped vectors."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def normalize_document_scope(value: Any) -> str:
    """Legacy/missing payloads default to company scope."""
    if value in {"company", "chat"}:
        return str(value)
    return "company"


def normalize_session_id(value: Any) -> str:
    """Empty string represents company / no-session for dedup keys."""
    if value is None:
        return ""
    text = str(value).strip()
    return text


def dedup_key(
    company_id: str,
    content_hash: str,
    *,
    document_scope: Any = "company",
    session_id: Any = None,
) -> tuple[str, str, str, str]:
    """Content-hash uniqueness within (company, scope, session)."""
    return (
        str(company_id),
        normalize_document_scope(document_scope),
        normalize_session_id(session_id),
        str(content_hash),
    )


def payload_dedup_key(payload: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return dedup_key(
        str(payload.get("company_id") or ""),
        str(payload.get("content_hash") or ""),
        document_scope=payload.get("document_scope"),
        session_id=payload.get("session_id"),
    )


def payload_in_scope(
    payload: Mapping[str, Any],
    *,
    company_id: str,
    session_id: Optional[str] = None,
    include_company_docs: bool = True,
    document_scope: Optional[str] = None,
) -> bool:
    """Return whether a stored payload is visible under the given filter.

    - document_scope=\"company\": admin/list default — only company docs.
    - session_id set: chat retrieval — that session's chat docs (+ company if allowed).
    - neither: company-only (stateless chat / retrieve must not see chat uploads).
    """
    if payload.get("company_id") != company_id:
        return False

    scope = normalize_document_scope(payload.get("document_scope"))
    payload_session = normalize_session_id(payload.get("session_id"))

    if document_scope == "company":
        return scope == "company"

    if document_scope == "chat":
        if scope != "chat":
            return False
        if session_id is None:
            return True
        return payload_session == normalize_session_id(session_id)

    # Retrieval / listing without an explicit document_scope filter.
    if session_id:
        if scope == "chat":
            return payload_session == normalize_session_id(session_id)
        return bool(include_company_docs) and scope == "company"

    return scope == "company"


def chunk_payload_fields(chunk: Any) -> Dict[str, Any]:
    """Serialize scope fields for vector payloads."""
    scope = normalize_document_scope(getattr(chunk, "document_scope", None))
    session = getattr(chunk, "session_id", None)
    return {
        "document_scope": scope,
        "session_id": session if scope == "chat" else None,
    }
