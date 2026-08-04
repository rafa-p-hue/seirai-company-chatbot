"""Persistent chat orchestration around the existing grounded RAG service."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from app.config import get_settings
from app.models.api import (
    ChatMessage as RAGChatMessage,
    ChatRequest as RAGChatRequest,
    UploadDocumentResponse,
)

logger = logging.getLogger(__name__)
from app.models.chat import ChatDocumentAttachment, ChatMessage, ChatSession
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import (
    ChatAttachmentRead,
    ChatMessageCreate,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionRead,
    ChatSessionRename,
    ChatTurnRead,
)
from app.services.chat_service import ChatService as RAGChatService
from app.services.document_service import DocumentService


class ChatSessionNotFoundError(LookupError):
    pass


class ChatAttachmentNotFoundError(LookupError):
    pass


class ChatHistoryService:
    def __init__(
        self,
        *,
        repository: ChatRepository,
        rag_service: RAGChatService,
        document_service: Optional[DocumentService] = None,
    ) -> None:
        self.repository = repository
        self.rag_service = rag_service
        self.document_service = document_service

    def create_session(self, request: ChatSessionCreate) -> ChatSessionRead:
        session = self.repository.create_session(request.title)
        self.repository.commit()
        return ChatSessionRead.model_validate(session)

    def list_sessions(self) -> List[ChatSessionRead]:
        return [
            ChatSessionRead.model_validate(session)
            for session in self.repository.list_sessions()
        ]

    def get_session(self, session_id: str) -> ChatSessionDetail:
        session = self._require_session(session_id, include_messages=True)
        return self._session_detail(session)

    def rename_session(
        self, session_id: str, request: ChatSessionRename
    ) -> ChatSessionRead:
        session = self._require_session(session_id)
        self.repository.rename_session(session, request.title)
        self.repository.commit()
        return ChatSessionRead.model_validate(session)

    async def delete_session(self, session_id: str) -> None:
        session = self._require_session(session_id)
        attachments = self.repository.list_attachments(session_id)
        if self.document_service is not None:
            for attachment in attachments:
                await self.document_service.delete_document(
                    attachment.company_id, attachment.document_id
                )
        self.repository.soft_delete_attachments_for_session(session_id)
        self.repository.soft_delete_session(session)
        self.repository.commit()

    async def add_message(
        self, session_id: str, request: ChatMessageCreate
    ) -> ChatTurnRead:
        session = self._require_session(session_id)
        settings = get_settings()
        recent = self.repository.recent_messages(
            session_id,
            limit=settings.chat_history_message_limit,
        )
        is_first_user_message = self.repository.user_message_count(session_id) == 0

        try:
            user = self.repository.add_message(
                session_id=session_id,
                role="user",
                content=request.content,
            )
            rag_response = await self.rag_service.chat(
                RAGChatRequest(
                    company_id=request.company_id,
                    question=request.content,
                    conversation_id=session_id,
                    history=[
                        RAGChatMessage(role=message.role, content=message.content)
                        for message in recent
                    ],
                    top_k=request.top_k,
                    session_id=session_id,
                    include_company_docs=request.include_company_docs,
                )
            )
            # Dev-only side-channel capture for live pipeline diagnosis.
            # Does not alter retrieval, ranking, validation, or the API response.
            if settings.is_development:
                try:
                    debug_dir = Path(__file__).resolve().parents[2] / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    capture = {
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "endpoint": (
                            f"POST /api/chat/sessions/{session_id}/messages"
                        ),
                        "session_id": session_id,
                        "company_id": request.company_id,
                        "top_k": request.top_k,
                        "include_company_docs": request.include_company_docs,
                        "original_query": request.content,
                        "history_len": len(recent),
                        "answer": rag_response.answer,
                        "sources": [
                            source.model_dump(mode="json")
                            for source in rag_response.sources
                        ],
                        "diagnostics": rag_response.diagnostics,
                    }
                    capture_path = debug_dir / "session_message_captures.jsonl"
                    with capture_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(capture, ensure_ascii=False, default=str)
                            + "\n"
                        )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to write session message diagnostics")
            citations_json = json.dumps(
                [
                    source.model_dump(mode="json")
                    for source in rag_response.sources
                ],
                ensure_ascii=False,
            )
            assistant = self.repository.add_message(
                session_id=session_id,
                role="assistant",
                content=rag_response.answer,
                citations_json=citations_json,
            )
            # Guarantee assistant sorts after its user when clocks share a second.
            if assistant.created_at <= user.created_at:
                assistant.created_at = user.created_at + timedelta(microseconds=1000)
                self.repository.database.flush()
            if is_first_user_message and session.title == "New Chat":
                session.title = _title_from_message(request.content)
            self.repository.touch_session(session)
            self.repository.commit()
            return ChatTurnRead(
                user=_message_read(user),
                assistant=_message_read(assistant),
            )
        except Exception:
            self.repository.rollback()
            raise

    def list_attachments(self, session_id: str) -> List[ChatAttachmentRead]:
        self._require_session(session_id)
        return [
            _attachment_read(attachment)
            for attachment in self.repository.list_attachments(session_id)
        ]

    async def upload_attachment(
        self,
        session_id: str,
        *,
        company_id: str,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> UploadDocumentResponse:
        if self.document_service is None:
            raise RuntimeError("Document service is not configured.")
        session = self._require_session(session_id)
        try:
            response = await self.document_service.upload_document(
                company_id=company_id,
                filename=filename,
                data=data,
                session_id=session_id,
                document_scope="chat",
                content_type=content_type,
            )
            self.repository.add_attachment(
                session_id=session_id,
                company_id=company_id,
                document_id=response.document.document_id,
                document_name=response.document.document_name,
                status=response.document.status,
            )
            self.repository.touch_session(session)
            self.repository.commit()
            # Chat UI only needs document metadata — omit bulky page/chunk dumps.
            return response.model_copy(update={"pages": [], "chunks": []})
        except Exception:
            self.repository.rollback()
            raise

    async def delete_attachment(self, session_id: str, document_id: str) -> int:
        if self.document_service is None:
            raise RuntimeError("Document service is not configured.")
        self._require_session(session_id)
        attachment = self.repository.get_attachment(session_id, document_id)
        if attachment is None:
            raise ChatAttachmentNotFoundError(document_id)
        deleted = await self.document_service.delete_document(
            attachment.company_id, attachment.document_id
        )
        self.repository.soft_delete_attachment(attachment)
        self.repository.commit()
        return deleted

    def register_attachment(
        self,
        *,
        session_id: str,
        company_id: str,
        document_id: str,
        document_name: str,
        status: str = "ready",
    ) -> ChatAttachmentRead:
        """Register an attachment row after a successful scoped upload."""
        session = self._require_session(session_id)
        existing = self.repository.get_attachment(session_id, document_id)
        if existing is not None:
            return _attachment_read(existing)
        attachment = self.repository.add_attachment(
            session_id=session_id,
            company_id=company_id,
            document_id=document_id,
            document_name=document_name,
            status=status,
        )
        self.repository.touch_session(session)
        self.repository.commit()
        return _attachment_read(attachment)

    def _require_session(
        self, session_id: str, *, include_messages: bool = False
    ) -> ChatSession:
        session = self.repository.get_session(
            session_id, include_messages=include_messages
        )
        if session is None:
            raise ChatSessionNotFoundError(session_id)
        return session

    @staticmethod
    def _session_detail(session: ChatSession) -> ChatSessionDetail:
        messages = sorted(
            list(session.messages or []),
            key=lambda message: (message.created_at, message.id),
        )
        attachments = [
            attachment
            for attachment in getattr(session, "attachments", []) or []
            if getattr(attachment, "deleted_at", None) is None
        ]
        attachments.sort(key=lambda item: (item.created_at, item.id))
        return ChatSessionDetail(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            messages=[_message_read(message) for message in messages],
            attachments=[_attachment_read(attachment) for attachment in attachments],
        )


def _title_from_message(content: str) -> str:
    title = re.sub(r"\s+", " ", (content or "").splitlines()[0]).strip()
    title = title.strip("\"'` ")
    if len(title) <= 60:
        return title or "New Chat"
    return f"{title[:57].rstrip()}..."


def _message_read(message: ChatMessage) -> ChatMessageRead:
    citations = []
    if message.citations_json:
        try:
            parsed = json.loads(message.citations_json)
            if isinstance(parsed, list):
                citations = [item for item in parsed if isinstance(item, dict)]
        except (TypeError, ValueError):
            citations = []
    return ChatMessageRead(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        citations=citations,
        created_at=message.created_at,
    )


def _attachment_read(attachment: ChatDocumentAttachment) -> ChatAttachmentRead:
    return ChatAttachmentRead(
        id=attachment.id,
        session_id=attachment.session_id,
        company_id=attachment.company_id,
        document_id=attachment.document_id,
        document_name=attachment.document_name,
        filename=attachment.document_name,
        status=attachment.status,
        created_at=attachment.created_at,
    )
