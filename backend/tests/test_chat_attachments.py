"""Chat-scoped document attachments: upload, list isolation, retrieval."""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    database_path = tmp_path / "attachments.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.chdir(tmp_path)

    from app import dependencies
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_provider.cache_clear()

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_provider.cache_clear()


def _pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in text.splitlines() or [text]:
        page.insert_text((72, y), line[:110], fontsize=11)
        y += 16
    data = doc.tobytes()
    doc.close()
    return data


def _create_session(client: TestClient) -> dict:
    response = client.post("/api/chat/sessions")
    assert response.status_code == 201, response.text
    return response.json()


def _upload_company(client: TestClient, company_id: str, filename: str, text: str):
    response = client.post(
        "/api/documents/upload",
        data={"company_id": company_id},
        files={"file": (filename, _pdf_bytes(text), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_chat_attachment(
    client: TestClient,
    session_id: str,
    company_id: str,
    filename: str,
    text: str,
    *,
    via: str = "nested",
):
    if via == "nested":
        response = client.post(
            f"/api/chat/sessions/{session_id}/attachments",
            data={"company_id": company_id},
            files={"file": (filename, _pdf_bytes(text), "application/pdf")},
        )
        assert response.status_code == 201, response.text
    else:
        response = client.post(
            "/api/documents/upload",
            data={
                "company_id": company_id,
                "session_id": session_id,
                "document_scope": "chat",
            },
            files={"file": (filename, _pdf_bytes(text), "application/pdf")},
        )
        assert response.status_code == 200, response.text
    return response.json()


def test_company_upload_still_works(client: TestClient):
    company_id = "attach-co"
    upload = _upload_company(
        client,
        company_id,
        "company-handbook.pdf",
        "Acme Company refund window is fourteen calendar days.",
    )
    assert upload["document"]["document_scope"] == "company"
    assert upload["document"]["session_id"] is None
    assert upload["document"]["embedding_count"] > 0

    listed = client.get("/api/documents", params={"company_id": company_id})
    assert listed.status_code == 200
    docs = listed.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["document_id"] == upload["document"]["document_id"]

    chat = client.post(
        "/api/chat",
        json={
            "company_id": company_id,
            "question": "What is the refund window?",
        },
    )
    assert chat.status_code == 200
    assert "fourteen" in chat.json()["answer"].lower()


def test_chat_upload_not_in_admin_document_list(client: TestClient):
    company_id = "attach-admin"
    session = _create_session(client)
    company = _upload_company(
        client,
        company_id,
        "company.pdf",
        "Company policy mentions neon lighthouse docking fees.",
    )
    chat_doc = _upload_chat_attachment(
        client,
        session["id"],
        company_id,
        "private-note.pdf",
        "Secret chat note about violet comet passport stamps.",
        via="nested",
    )
    assert chat_doc["document"]["document_scope"] == "chat"
    assert chat_doc["document"]["session_id"] == session["id"]

    listed = client.get("/api/documents", params={"company_id": company_id})
    assert listed.status_code == 200
    ids = {doc["document_id"] for doc in listed.json()["documents"]}
    assert company["document"]["document_id"] in ids
    assert chat_doc["document"]["document_id"] not in ids

    attachments = client.get(f"/api/chat/sessions/{session['id']}/attachments")
    assert attachments.status_code == 200
    rows = attachments.json()
    assert len(rows) == 1
    assert rows[0]["document_id"] == chat_doc["document"]["document_id"]
    assert rows[0]["document_name"] == "private-note.pdf"


def test_session_chat_retrieves_its_attached_doc(client: TestClient):
    company_id = "attach-retrieve"
    session = _create_session(client)
    unique_fact = "Zephyr mango warranty lasts ninety-seven moons."
    _upload_chat_attachment(
        client,
        session["id"],
        company_id,
        "warranty.pdf",
        unique_fact,
        via="form",
    )

    # Stateless company chat must not see chat-scoped docs.
    company_chat = client.post(
        "/api/chat",
        json={"company_id": company_id, "question": "How long is the zephyr mango warranty?"},
    )
    assert company_chat.status_code == 200
    assert "ninety-seven" not in company_chat.json()["answer"].lower()

    # Session message path should retrieve the attachment.
    response = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={
            "company_id": company_id,
            "content": "How long is the zephyr mango warranty?",
        },
    )
    assert response.status_code == 201, response.text
    answer = response.json()["assistant"]["content"].lower()
    assert "ninety-seven" in answer


def test_sessions_do_not_leak_attachments(client: TestClient):
    company_id = "attach-leak"
    session_a = _create_session(client)
    session_b = _create_session(client)
    secret = "Session Alpha owns the crimson violin serial VX-4411."
    upload = _upload_chat_attachment(
        client,
        session_a["id"],
        company_id,
        "alpha-secret.pdf",
        secret,
    )
    doc_id = upload["document"]["document_id"]

    # Session B cannot list A's attachment.
    listed_b = client.get(f"/api/chat/sessions/{session_b['id']}/attachments")
    assert listed_b.status_code == 200
    assert listed_b.json() == []

    # Session B chat should not retrieve Alpha's secret.
    leak_probe = client.post(
        f"/api/chat/sessions/{session_b['id']}/messages",
        json={
            "company_id": company_id,
            "content": "What is the crimson violin serial?",
            "include_company_docs": False,
        },
    )
    assert leak_probe.status_code == 201, leak_probe.text
    assert "vx-4411" not in leak_probe.json()["assistant"]["content"].lower()

    # Soft-delete session A removes vectors from retrieval and attachment list.
    deleted = client.delete(f"/api/chat/sessions/{session_a['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/chat/sessions/{session_a['id']}/attachments").status_code == 404

    # Re-create a session and confirm the old vectors are gone.
    session_c = _create_session(client)
    after_delete = client.post(
        f"/api/chat/sessions/{session_c['id']}/messages",
        json={
            "company_id": company_id,
            "content": "What is the crimson violin serial?",
            "include_company_docs": False,
        },
    )
    assert after_delete.status_code == 201
    assert "vx-4411" not in after_delete.json()["assistant"]["content"].lower()

    # Explicit delete of an attachment also works on a live session.
    session_d = _create_session(client)
    upload_d = _upload_chat_attachment(
        client,
        session_d["id"],
        company_id,
        "temp.pdf",
        "Temporary orchid token ORCH-9.",
    )
    remove = client.delete(
        f"/api/chat/sessions/{session_d['id']}/attachments/"
        f"{upload_d['document']['document_id']}"
    )
    assert remove.status_code == 204
    assert client.get(f"/api/chat/sessions/{session_d['id']}/attachments").json() == []
    assert doc_id  # keep reference used above for clarity


def test_chat_and_company_dedup_are_independent(client: TestClient):
    company_id = "attach-dedup"
    text = "Shared body text about silver orchard irrigation timers."
    company = _upload_company(client, company_id, "shared.pdf", text)
    session = _create_session(client)
    chat = _upload_chat_attachment(
        client,
        session["id"],
        company_id,
        "shared.pdf",
        text,
    )
    assert company["document"]["embedding_count"] > 0
    assert chat["document"]["embedding_count"] > 0
    assert company["document"]["document_id"] != chat["document"]["document_id"]
