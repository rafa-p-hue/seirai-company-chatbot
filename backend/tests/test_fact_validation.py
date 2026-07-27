"""Regression tests for fact validation, school/major split, and multi-questions.

Uses invented fixtures only — no production PDF facts.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.retrieval.multi_question import split_questions
from app.retrieval.query_understanding import classify_query


APPLICANT = """Nominee Information
Nominee's Full Name: Riley Morgan Chen
Nominee's Email: riley.chen@example.edu
Nominee's School: Lakeside Polytechnic Institute
Nominee's Major: Applied Data Systems
Nominee's Minor: Visual Rhetoric
Program Status: Nominee

Research Interests
Research Assistant, Harbor Interface Lab - Assist in studies on coastal sensor networks.

Provide a list of accomplishments, involvements and activities.
Resident Advisor (2 Years) - Support a residential community of students.
Orientation Leader - Helped welcome incoming students through campus orientation.
Campus Concert Ensemble - Performs with the student music ensemble for campus events.
Founder, Civic Coding Club - Established weekly peer tutoring for introductory computing.
President, Campus Service Society - Leads community outreach weekends.
"""

POLICY = """Employee Handbook
Leave Policy: Employees receive 15 days of paid leave each year.
Remote Work Policy: Employees may request remote work after 90 days.
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
    return response.json()


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


def test_school_vs_major_classification():
    assert classify_query("What school does the person attend?") == "school"
    assert classify_query("What is the person's major?") == "major"
    assert classify_query("What is the person's GPA?") == "gpa"
    assert classify_query("What is the person's birthday?") == "birthday"
    assert classify_query("What is the person's favorite food?") == "favorite_food"
    assert classify_query("What leadership roles are listed?") == "leadership"
    assert classify_query("What kinds of questions can I ask?") == "help"


def test_multi_question_split_preserves_order():
    parts = split_questions(
        "Is the person involved in music?\n"
        "What group did the person found?\n"
        "What instrument does the person play?"
    )
    assert len(parts) == 3
    assert "music" in parts[0].lower()
    assert "found" in parts[1].lower()
    assert "instrument" in parts[2].lower()


def test_school_and_major_not_confused(client: TestClient):
    _upload(client, "fact-app", "nominee.pdf", APPLICANT)

    school = _chat(client, "fact-app", "What school does the person attend?")
    assert "lakeside" in school["answer"].lower() or "polytechnic" in school["answer"].lower()
    assert "applied data" not in school["answer"].lower()
    assert "could not find" not in school["answer"].lower()
    assert "email" not in school["answer"].lower()

    major = _chat(client, "fact-app", "What is the person's major?")
    assert "applied data" in major["answer"].lower()
    assert "lakeside" not in major["answer"].lower()
    assert "could not find" not in major["answer"].lower()


def test_unsupported_fact_types_fallback(client: TestClient):
    _upload(client, "fact-miss", "nominee.pdf", APPLICANT)
    for question in (
        "What is the person's GPA?",
        "What is the person's birthday?",
        "What is the person's favorite food?",
    ):
        body = _chat(client, "fact-miss", question)
        assert "could not find" in body["answer"].lower()
        assert "email" not in body["answer"].lower()
        assert "riley" not in body["answer"].lower() or "could not find" in body["answer"].lower()


def test_multi_question_partial_fallback(client: TestClient):
    _upload(client, "fact-multi", "nominee.pdf", APPLICANT)
    body = _chat(
        client,
        "fact-multi",
        "Is the person involved in music?\n"
        "What group did the person found?\n"
        "What instrument does the person play?",
    )
    answer = body["answer"].lower()
    diag = body.get("diagnostics") or {}
    assert diag.get("sub_questions")
    assert len(diag["sub_questions"]) == 3
    assert "music" in answer or "ensemble" in answer
    assert "coding" in answer or "founder" in answer or "civic" in answer
    assert "could not find" in answer
    # One sub-question failing must not wipe the others.
    assert answer.count("could not find") == 1 or "instrument" in answer


def test_leadership_prefers_titles(client: TestClient):
    _upload(client, "fact-lead", "nominee.pdf", APPLICANT)
    body = _chat(client, "fact-lead", "What leadership roles are listed?")
    answer = body["answer"].lower()
    sent = " ".join(
        chunk["content"].lower()
        for chunk in (body.get("diagnostics") or {}).get("evidence_sent_to_llm") or []
    )
    assert "president" in answer or "president" in sent or "founder" in answer
    # Research assistant alone should not dominate leadership answers.
    if "research assistant" in answer and "president" not in answer and "founder" not in answer:
        pytest.fail("leadership answer should prioritize leadership titles")


def test_summary_is_synthesized(client: TestClient):
    _upload(client, "fact-sum", "nominee.pdf", APPLICANT)
    body = _chat(client, "fact-sum", "What can you tell me about this document?")
    answer = body["answer"]
    assert "could not find" not in answer.lower()
    assert "applying" not in answer.lower()
    # Should be complete sentences, not a raw chunk paste of email/minor only.
    assert not answer.lower().startswith("nominee's email")
    assert "." in answer


def test_capability_help_is_generic(client: TestClient):
    _upload(client, "fact-help", "nominee.pdf", APPLICANT)
    body = _chat(client, "fact-help", "What kinds of questions can I ask?")
    assert body.get("diagnostics", {}).get("retrieval_skipped") is True
    answer = body["answer"].lower()
    assert "people" in answer or "education" in answer or "documents" in answer
    assert "lakeside" not in answer
    assert "riley" not in answer


def test_policy_document_still_works(client: TestClient):
    _upload(client, "fact-policy", "handbook.pdf", POLICY)
    body = _chat(client, "fact-policy", "What can you tell me about this document?")
    assert "could not find" not in body["answer"].lower()
