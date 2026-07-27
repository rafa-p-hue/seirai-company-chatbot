"""End-to-end smoke tests using the in-memory vector store."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.chdir(tmp_path)

    from app.config import get_settings
    from app import dependencies
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
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_upload_retrieve_chat_and_refund_fallback(client: TestClient):
    sample = (FIXTURES / "sample_company.txt").read_text()
    files = {"file": ("company-overview.pdf", _pdf_bytes(sample), "application/pdf")}
    upload = client.post(
        "/api/documents/upload",
        data={"company_id": "acme"},
        files=files,
    )
    assert upload.status_code == 200, upload.text
    body = upload.json()
    assert body["document"]["embedding_count"] >= 1
    assert body["document"]["embedding_model"]
    assert body["chunks"]

    # Duplicate upload should skip already-known content hashes (embedding_count may be 0).
    duplicate = client.post(
        "/api/documents/upload",
        data={"company_id": "acme"},
        files=files,
    )
    assert duplicate.status_code == 200

    services = client.post(
        "/api/retrieve",
        json={
            "company_id": "acme",
            "question": "What services does the company provide?",
            "top_k": 5,
        },
    )
    assert services.status_code == 200
    results = services.json()["results"]
    assert results
    blob = " ".join(item["content"].lower() for item in results)
    assert "inventory" in blob or "automation" in blob or "analytics" in blob

    contact = client.post(
        "/api/chat",
        json={
            "company_id": "acme",
            "question": "How can customers contact the company?",
            "history": [],
        },
    )
    assert contact.status_code == 200
    contact_body = contact.json()
    assert "support@acme-orchard.example" in contact_body["answer"].lower() or contact_body["sources"]

    refund = client.post(
        "/api/chat",
        json={
            "company_id": "acme",
            "question": "What is the refund policy?",
            "history": [],
        },
    )
    assert refund.status_code == 200
    assert "could not find that information" in refund.json()["answer"].lower()

    # Company separation: beta should not see acme chunks.
    other = client.post(
        "/api/retrieve",
        json={
            "company_id": "beta",
            "question": "What services does the company provide?",
            "top_k": 5,
        },
    )
    assert other.status_code == 200
    assert other.json()["results"] == []
