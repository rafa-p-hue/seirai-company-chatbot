"""Recover in-memory vector indexes after process restart."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import DefaultDict, List, Tuple

from sqlalchemy import select

from app.config import get_settings
from app.database import get_session_factory
from app.models.chat import ChatDocumentAttachment

logger = logging.getLogger(__name__)


async def recover_memory_index_if_empty() -> None:
    """Rebuild chat-scoped embeddings from on-disk uploads when memory store is empty.

    Chat attachment rows and upload files survive restarts; VECTOR_STORE=memory
    does not. Without recovery, the UI still shows Ready while retrieval returns
    zero candidates.
    """
    settings = get_settings()
    if settings.vector_store.lower().strip() not in {"memory", "inmemory", "local"}:
        return

    from app.dependencies import get_document_service, get_vector_store

    store = get_vector_store()
    points = getattr(store, "_points", None)
    if isinstance(points, dict) and points:
        return

    database = get_session_factory(settings.database_url)()
    try:
        attachments = list(
            database.scalars(
                select(ChatDocumentAttachment).where(
                    ChatDocumentAttachment.deleted_at.is_(None)
                )
            ).all()
        )
    finally:
        database.close()

    if not attachments:
        logger.info("Memory vector store empty; no chat attachments to recover.")
        return

    grouped: DefaultDict[Tuple[str, str], List[ChatDocumentAttachment]] = defaultdict(
        list
    )
    for attachment in attachments:
        grouped[(attachment.company_id, attachment.session_id)].append(attachment)

    service = get_document_service()
    logger.warning(
        "Memory vector store empty after restart; rebuilding %s chat session(s) "
        "from on-disk uploads.",
        len(grouped),
    )
    for (company_id, session_id), rows in grouped.items():
        try:
            status = await service.rebuild_session_index(
                company_id=company_id,
                session_id=session_id,
                attachments=rows,
            )
            logger.info(
                "Recovered session %s (%s): %s indexed chunks",
                session_id,
                company_id,
                status.get("indexed_chunk_count"),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to recover vector index for session %s (%s)",
                session_id,
                company_id,
            )
