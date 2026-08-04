"""Architecture-aligned regression tests (document-agnostic fixtures only)."""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.generation.citations import remap_answer_citations, sources_from_answer
from app.ingestion.cleaner import estimate_tokens
from app.ingestion.universal_chunker import create_universal_chunks
from app.models.api import ExtractedPage, RetrievedChunk
from app.generation.text_scrub import scrub_internal_metadata


POLICY = """Employee Handbook Overview
Welcome to the company handbook. This overview is high level only.

Leave Policy
Employees receive 15 days of paid leave each year.
Remote Work Policy: Employees may request remote work after 90 days.

Product Specification
Model: OrbitDock X2
Calibration temperature: 22C
Feature list:
1. Auto-lock dock
2. Humidity alarm
3. Manual override switch
"""

APPLICANT = """Nominee Information
Nominee's Full Name: Avery Quinn
Nominee's Email: avery.quinn@example.edu
Nominee's School: Riverbend Institute of Technology
Nominee's Major: Systems Design
"""


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "development")
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
    y = 56
    for line in text.splitlines():
        if not line.strip():
            y += 10
            continue
        page.insert_text((48, y), line[:110], fontsize=10)
        y += 14
        if y > 780:
            page = doc.new_page()
            y = 56
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client: TestClient, company_id: str, filename: str, text: str):
    response = client.post(
        "/api/documents/upload",
        data={"company_id": company_id},
        files={"file": (filename, _pdf_bytes(text), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_chunker_keeps_clean_text_and_metadata():
    pages = [ExtractedPage(page_number=1, text=POLICY)]
    chunks = create_universal_chunks(
        pages=pages,
        company_id="arch",
        document_id="doc1",
        document_name="handbook.pdf",
    )
    assert chunks
    assert all("Record type:" not in (c.content or "") for c in chunks)
    assert all(not (c.content or "").startswith("[") for c in chunks if c.content_type != "heading")
    # Soft token guidance: most content chunks stay under ~300 tokens.
    oversized = [
        c
        for c in chunks
        if c.content_type not in {"heading", "prompt"}
        and estimate_tokens(c.content) > 320
    ]
    assert not oversized
    section_titles = {c.section_title for c in chunks if c.section_title}
    assert "Leave Policy" in section_titles
    assert "Product Specification" in section_titles
    assert any(
        c.section_title == "Remote Work Policy"
        and "remote work after 90 days" in (c.content or "").lower()
        for c in chunks
    )
    # Short headings kept as anchors; leave body is not a key-value under overview.
    assert any(
        c.content_type == "heading" and c.content == "Leave Policy" for c in chunks
    )


def test_quantity_query_classifies_without_policy_word():
    from app.retrieval.query_understanding import classify_query

    assert (
        classify_query("How many paid leave days do employees receive?") == "quantity"
    )
    assert classify_query("How much does membership cost?") == "price"


def test_scrub_rejects_internal_scaffolding():
    raw = "Record type: profile\nPerson: Sample\nInternal score: 0.9\nUseful fact here."
    cleaned = scrub_internal_metadata(raw)
    assert "record type" not in cleaned.lower()
    assert "internal score" not in cleaned.lower()
    assert "useful fact" in cleaned.lower()


def test_citation_remap_keeps_display_aligned():
    answer = "Leave is 15 days [1]. Remote work after 90 days [2]."
    remapped = remap_answer_citations(answer, {1: 3, 2: 4})
    assert "[3]" in remapped and "[4]" in remapped
    assert "[1]" not in remapped


def test_sources_omit_unused_without_fallback():
    evidence = [
        RetrievedChunk(
            content="Employees receive 15 days of paid leave each year.",
            document_name="h.pdf",
            page_number=1,
            score=0.9,
            chunk_id="a",
        ),
        RetrievedChunk(
            content="Unrelated overview welcome text.",
            document_name="h.pdf",
            page_number=1,
            score=0.8,
            chunk_id="b",
        ),
    ]
    sources = sources_from_answer(
        "Employees receive 15 days of paid leave each year.",
        evidence,
        query_type="policy",
        allow_uncited_fallback=False,
    )
    assert sources == []


def test_policy_and_unsupported_with_diagnostics(client: TestClient):
    upload = _upload(client, "arch-policy", "handbook.pdf", POLICY)
    doc_id = upload["document"]["document_id"]

    chunks = client.get(
        f"/api/documents/{doc_id}/chunks",
        params={"company_id": "arch-policy"},
    )
    assert chunks.status_code == 200, chunks.text
    body = chunks.json()
    assert body["chunks"]
    assert all("record type" not in c["content"].lower() for c in body["chunks"])

    ok = client.post(
        "/api/chat",
        json={
            "company_id": "arch-policy",
            "question": "How many paid leave days do employees receive?",
            "history": [],
        },
    )
    assert ok.status_code == 200
    ok_body = ok.json()
    assert "15" in ok_body["answer"]
    assert "could not find" not in ok_body["answer"].lower()
    assert ok_body.get("diagnostics")
    assert "record type" not in ok_body["answer"].lower()

    miss = client.post(
        "/api/chat",
        json={
            "company_id": "arch-policy",
            "question": "What is the CEO zodiac sign?",
            "history": [],
        },
    )
    assert miss.status_code == 200
    miss_body = miss.json()
    assert "could not find" in miss_body["answer"].lower()

    retrieve = client.post(
        "/api/retrieve",
        json={
            "company_id": "arch-policy",
            "question": "How many paid leave days do employees receive?",
            "top_k": 5,
        },
    )
    assert retrieve.status_code == 200
    retrieved = retrieve.json()
    assert retrieved.get("inspection")
    assert retrieved["inspection"].get("query_intent") or retrieved.get("query_type")


def test_labeled_fields_still_work(client: TestClient):
    _upload(client, "arch-app", "nominee.pdf", APPLICANT)
    body = client.post(
        "/api/chat",
        json={
            "company_id": "arch-app",
            "question": "What is the person's major?",
            "history": [],
        },
    ).json()
    assert "systems design" in body["answer"].lower()
    assert "email" not in body["answer"].lower()
