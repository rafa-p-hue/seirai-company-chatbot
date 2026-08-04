"""Database queries for chat sessions and messages."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from app.models.chat import ChatDocumentAttachment, ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, database: Session) -> None:
        self.database = database

    def create_session(self, title: str = "New Chat") -> ChatSession:
        session = ChatSession(title=title)
        self.database.add(session)
        self.database.flush()
        return session

    def list_sessions(self) -> List[ChatSession]:
        statement = (
            select(ChatSession)
            .where(ChatSession.deleted_at.is_(None))
            .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        )
        return list(self.database.scalars(statement).all())

    def get_session(
        self, session_id: str, *, include_messages: bool = False
    ) -> Optional[ChatSession]:
        statement: Select = select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.deleted_at.is_(None),
        )
        if include_messages:
            statement = statement.options(
                selectinload(ChatSession.messages),
                selectinload(ChatSession.attachments),
            )
        return self.database.scalar(statement)

    def rename_session(self, session: ChatSession, title: str) -> ChatSession:
        session.title = title
        session.updated_at = datetime.utcnow()
        self.database.flush()
        return session

    def soft_delete_session(self, session: ChatSession) -> None:
        now = datetime.utcnow()
        session.deleted_at = now
        session.updated_at = now
        self.database.flush()

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        citations_json: Optional[str] = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            citations_json=citations_json,
        )
        self.database.add(message)
        self.database.flush()
        return message

    def recent_messages(
        self, session_id: str, *, limit: int = 10
    ) -> List[ChatMessage]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        messages = list(self.database.scalars(statement).all())
        messages.reverse()
        return messages

    def user_message_count(self, session_id: str) -> int:
        statement = select(func.count(ChatMessage.id)).where(
            ChatMessage.session_id == session_id,
            ChatMessage.role == "user",
        )
        return int(self.database.scalar(statement) or 0)

    def touch_session(self, session: ChatSession) -> None:
        session.updated_at = datetime.utcnow()
        self.database.flush()

    def add_attachment(
        self,
        *,
        session_id: str,
        company_id: str,
        document_id: str,
        document_name: str,
        status: str = "ready",
    ) -> ChatDocumentAttachment:
        attachment = ChatDocumentAttachment(
            session_id=session_id,
            company_id=company_id,
            document_id=document_id,
            document_name=document_name,
            status=status,
        )
        self.database.add(attachment)
        self.database.flush()
        return attachment

    def list_attachments(
        self, session_id: str, *, include_deleted: bool = False
    ) -> List[ChatDocumentAttachment]:
        statement = select(ChatDocumentAttachment).where(
            ChatDocumentAttachment.session_id == session_id
        )
        if not include_deleted:
            statement = statement.where(ChatDocumentAttachment.deleted_at.is_(None))
        statement = statement.order_by(
            ChatDocumentAttachment.created_at.desc(),
            ChatDocumentAttachment.id.desc(),
        )
        return list(self.database.scalars(statement).all())

    def get_attachment(
        self,
        session_id: str,
        document_id: str,
        *,
        include_deleted: bool = False,
    ) -> Optional[ChatDocumentAttachment]:
        statement = select(ChatDocumentAttachment).where(
            ChatDocumentAttachment.session_id == session_id,
            ChatDocumentAttachment.document_id == document_id,
        )
        if not include_deleted:
            statement = statement.where(ChatDocumentAttachment.deleted_at.is_(None))
        return self.database.scalar(statement)

    def soft_delete_attachment(self, attachment: ChatDocumentAttachment) -> None:
        now = datetime.utcnow()
        attachment.deleted_at = now
        attachment.status = "deleted"
        self.database.flush()

    def soft_delete_attachments_for_session(self, session_id: str) -> List[ChatDocumentAttachment]:
        attachments = self.list_attachments(session_id, include_deleted=False)
        for attachment in attachments:
            self.soft_delete_attachment(attachment)
        return attachments

    def commit(self) -> None:
        self.database.commit()

    def rollback(self) -> None:
        self.database.rollback()
