"""API regression tests for SQLite-backed chat history."""

from __future__ import annotations

from typing import List

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.api import ChatRequest, ChatResponse, CitationSource, SourceType
from app.repositories.chat_repository import ChatRepository
from app.services.chat_history_service import ChatHistoryService


class RecordingRAGService:
    def __init__(self) -> None:
        self.requests: List[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request.model_copy(deep=True))
        return ChatResponse(
            answer=f"Grounded answer for: {request.question} [1]",
            sources=[
                CitationSource(
                    number=1,
                    document_name="guide.pdf",
                    page_number=2,
                    source_type=SourceType.pdf,
                )
            ],
            conversation_id=request.conversation_id,
        )


@pytest.fixture()
def history_client(monkeypatch, tmp_path):
    database_path = tmp_path / "history.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "test")

    from app import dependencies
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_provider.cache_clear()

    rag = RecordingRAGService()
    app = create_app()

    def override_history_service(
        database: Session = Depends(get_db),
    ) -> ChatHistoryService:
        return ChatHistoryService(
            repository=ChatRepository(database),
            rag_service=rag,  # type: ignore[arg-type]
        )

    app.dependency_overrides[
        dependencies.get_chat_history_service
    ] = override_history_service

    with TestClient(app) as client:
        yield client, rag

    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_provider.cache_clear()


def _create_session(client: TestClient) -> dict:
    response = client.post("/api/chat/sessions")
    assert response.status_code == 201, response.text
    return response.json()


def test_create_and_list_sessions(history_client):
    client, _rag = history_client
    created = _create_session(client)
    assert created["title"] == "New Chat"

    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    assert [session["id"] for session in response.json()] == [created["id"]]


def test_add_message_reopen_and_auto_title(history_client):
    client, rag = history_client
    session = _create_session(client)
    content = "When must I register after moving into the city?"

    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"company_id": "seirai", "content": content},
    )
    assert response.status_code == 201, response.text
    turn = response.json()
    assert turn["user"]["role"] == "user"
    assert turn["user"]["content"] == content
    assert turn["assistant"]["role"] == "assistant"
    assert turn["assistant"]["citations"][0]["document_name"] == "guide.pdf"
    assert rag.requests[0].history == []
    assert rag.requests[0].conversation_id == session["id"]
    assert rag.requests[0].session_id == session["id"]
    assert rag.requests[0].include_company_docs is True

    reopened = client.get(f"/api/chat/sessions/{session['id']}")
    assert reopened.status_code == 200
    body = reopened.json()
    assert body["title"] == content
    assert [message["role"] for message in body["messages"]] == [
        "user",
        "assistant",
    ]
    assert body["messages"][0]["content"] == content
    assert body["messages"][1]["citations"][0]["page_number"] == 2
    # Session detail must be chronological ascending.
    assert body["messages"][0]["created_at"] <= body["messages"][1]["created_at"]


def test_rename_soft_delete_and_hide_from_recents(history_client):
    client, _rag = history_client
    session = _create_session(client)

    renamed = client.patch(
        f"/api/chat/sessions/{session['id']}",
        json={"title": "Resident registration"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Resident registration"

    deleted = client.delete(f"/api/chat/sessions/{session['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/chat/sessions").json() == []
    assert client.get(f"/api/chat/sessions/{session['id']}").status_code == 404


def test_session_messages_are_isolated(history_client):
    client, rag = history_client
    first = _create_session(client)
    second = _create_session(client)

    client.post(
        f"/api/chat/sessions/{first['id']}/messages",
        json={"content": "Private first-session topic"},
    )
    response = client.post(
        f"/api/chat/sessions/{second['id']}/messages",
        json={"content": "Independent second-session topic"},
    )
    assert response.status_code == 201
    assert rag.requests[-1].history == []

    second_detail = client.get(f"/api/chat/sessions/{second['id']}").json()
    assert all(
        "Private first-session" not in message["content"]
        for message in second_detail["messages"]
    )


def test_follow_up_history_is_passed_to_existing_rag_pipeline(history_client):
    client, rag = history_client
    session = _create_session(client)
    first_question = "When must I register after moving?"

    client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": first_question},
    )
    follow_up = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "What do I bring?"},
    )
    assert follow_up.status_code == 201

    request = rag.requests[-1]
    assert request.question == "What do I bring?"
    assert [(item.role, item.content) for item in request.history] == [
        ("user", first_question),
        ("assistant", f"Grounded answer for: {first_question} [1]"),
    ]
