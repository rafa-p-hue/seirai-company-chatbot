"""Regression: unsupported claims, research-only lists, location, multi-fact.

Uses invented fixtures only — no production PDF facts.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.generation.answer_composer import _looks_like_raw_chunk_dump
from app.models.api import RetrievedChunk
from app.retrieval.multi_question import split_questions
from app.retrieval.query_understanding import classify_query


PROFILE = """Nominee Information
Nominee's Full Name: Riley Morgan Chen
Nominee's Email: riley.chen@example.edu
Nominee's School: Lakeside Polytechnic Institute
Nominee's Major: Applied Data Systems
Nominee's Minor: Visual Rhetoric
Program Status: Nominee

Research Interests
Research Assistant, Harbor Interface Lab - Assist in studies on coastal sensor networks.
UROP Fellow - Conducted undergraduate research on sensor calibration.

Provide a list of accomplishments, involvements and activities.
Resident Advisor (2 Years) - Support a residential community of students.
Academics Chair, Society of Coastal Engineers - Organized study sessions.
Founder, Civic Coding Club - Established weekly peer tutoring for introductory computing.
President, Campus Service Society - Leads community outreach weekends.
Member, Nu Delta Fraternity - Served as Media Chair and Academic Chair.
Campus Concert Ensemble - Performs with the student music ensemble for campus events.
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
    safe = text.replace("–", "-").replace("—", "-")
    doc = fitz.open()
    page = doc.new_page()
    y = 56
    for line in safe.splitlines():
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


def _chat(client: TestClient, company_id: str, question: str, history=None):
    response = client.post(
        "/api/chat",
        json={
            "company_id": company_id,
            "question": question,
            "history": history or [],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_classify_unsupported_claim_intents():
    assert classify_query("Did he receive any awards?") == "awards"
    assert classify_query("Did he win a Grammy?") == "awards"
    assert classify_query("Does he play the violin?") == "instrument"
    assert classify_query("What company is he the CEO of?") == "executive"
    assert classify_query("Is he the president of Civic Coding Club?") == "leadership"
    assert classify_query("what state") == "location"
    assert classify_query("where is his school located") == "location"


def test_raw_email_dump_detected():
    chunk = RetrievedChunk(
        content="Nominee's Email: riley.chen@example.edu",
        document_name="x.pdf",
        page_number=1,
        chunk_id="1",
        content_type="key_value",
        label="Nominee's Email",
        value="riley.chen@example.edu",
        score=0.9,
    )
    assert _looks_like_raw_chunk_dump("Nominee's Email: riley.chen@example.edu.", [chunk])
    assert _looks_like_raw_chunk_dump("Nominee's Email: riley.chen@example.edu", [chunk])


def test_unsupported_claims_fallback_not_email(client: TestClient):
    _upload(client, "claim-miss", "nominee.pdf", PROFILE)
    for question in (
        "Did he receive any awards?",
        "Does he play the violin?",
        "What company is he the CEO of?",
        "Did he win a Grammy?",
        "Is he the president of Civic Coding Club?",
    ):
        body = _chat(client, "claim-miss", question)
        answer = body["answer"].lower()
        assert "could not find" in answer, question
        assert "email" not in answer, question
        assert "riley.chen@" not in answer, question


def test_research_positions_exclude_leadership(client: TestClient):
    _upload(client, "research-only", "nominee.pdf", PROFILE)
    body = _chat(client, "research-only", "List all of his research positions.")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "harbor" in answer or "research" in answer or "urop" in answer
    assert "civic coding" not in answer
    assert "fraternity" not in answer
    assert "academics chair" not in answer


def test_location_without_geo_evidence_fallback(client: TestClient):
    _upload(client, "loc-miss", "nominee.pdf", PROFILE)
    for question in ("what state", "where is his school located"):
        body = _chat(client, "loc-miss", question)
        answer = body["answer"].lower()
        assert "could not find" in answer, question
        assert "inferred" not in answer
        assert "@" not in answer


def test_compound_multi_fact_split():
    parts = split_questions(
        "What is Riley's major and minor, what fraternity is he in, "
        "what research has he conducted, and which details are not specified?"
    )
    assert len(parts) >= 3
    joined = " ".join(parts).lower()
    assert "major" in joined
    assert "fraternity" in joined
    assert "research" in joined


def test_compound_multi_fact_answers(client: TestClient):
    _upload(client, "multi-fact", "nominee.pdf", PROFILE)
    body = _chat(
        client,
        "multi-fact",
        "What is Riley's major and minor, what fraternity is he in, "
        "what research has he conducted, and which details are not specified in the document?",
    )
    answer = body["answer"].lower()
    assert "applied data" in answer
    assert "visual rhetoric" in answer
    assert "fraternity" in answer or "nu delta" in answer
    assert "harbor" in answer or "research" in answer or "urop" in answer
    # Must not claim major/fraternity/research are unspecified when present.
    assert not (
        "major" in answer
        and "not specified" in answer
        and "applied data" not in answer
    )
