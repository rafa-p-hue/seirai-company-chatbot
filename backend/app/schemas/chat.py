"""Request and response schemas for persistent chat history."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class ChatSessionCreate(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=60)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            raise ValueError("Title cannot be empty.")
        return cleaned


class ChatSessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=60)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            raise ValueError("Title cannot be empty.")
        return cleaned


class ChatMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    company_id: str = Field(default="seirai", min_length=1, max_length=64)
    top_k: int = Field(default=5, ge=1, le=20)
    include_company_docs: bool = True

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        return cleaned

    @field_validator("company_id")
    @classmethod
    def clean_company_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9_-]{1,64}", cleaned):
            raise ValueError("Invalid company_id.")
        return cleaned


class ChatMessageRead(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class ChatTurnRead(BaseModel):
    """User + assistant pair created by POST .../messages."""

    user: ChatMessageRead
    assistant: ChatMessageRead


class ChatAttachmentRead(BaseModel):
    id: str
    session_id: str
    company_id: str
    document_id: str
    document_name: str
    filename: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionRead(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionDetail(ChatSessionRead):
    messages: List[ChatMessageRead] = Field(default_factory=list)
    attachments: List[ChatAttachmentRead] = Field(default_factory=list)
