"""Regression tests for institution acronyms and clean multi-question answers."""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.models.api import RetrievedChunk
from app.retrieval.institution import (
    content_has_institution_evidence,
    extract_best_institution,
)
from app.retrieval.query_understanding import classify_query


ACRONYM_ONLY = """Profile
Full Name: Casey Quinn
Education
While at NST, Casey completed coursework in systems design.
Campus Concert Ensemble - Performs with the student music ensemble.
Founder, Harbor Coding Collective - Established weekly peer tutoring.
"""

FULL_NAME_ONLY = """Profile
Full Name: Casey Quinn
School: Northshore Technical University
Founder, Harbor Coding Collective - Established weekly peer tutoring.
Campus Concert Ensemble - Performs with the student music ensemble.
"""

BOTH = """Profile
Full Name: Casey Quinn
School: Northshore Technical University (NST)
Student email: casey@nst.edu
Founder, Harbor Coding Collective - Established weekly peer tutoring.
[Recruitment] Campus Concert Ensemble - Performs with the student music ensemble for campus events.
"""

NO_SCHOOL = """Profile
Full Name: Casey Quinn
Major: Systems Design
Founder, Harbor Coding Collective - Established weekly peer tutoring.
Campus Concert Ensemble - Performs with the student music ensemble.
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


def _chat(client: TestClient, company_id: str, question: str):
    response = client.post(
        "/api/chat",
        json={"company_id": company_id, "question": question, "history": []},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_institution_helpers_acronym_and_full_name():
    assert content_has_institution_evidence("While at NST, completed coursework.")
    assert content_has_institution_evidence("School: NST")
    assert content_has_institution_evidence("School: Northshore Technical University")
    assert not content_has_institution_evidence("Major: Systems Design")
    # Email alone is weak / insufficient as sole evidence.
    assert extract_best_institution(
        [RetrievedChunk(content="Email: casey@nst.edu", document_name="x", score=1.0)]
    ) is None
    assert (
        extract_best_institution(
            [
                RetrievedChunk(
                    content="While at NST, completed coursework.",
                    document_name="x",
                    score=1.0,
                )
            ]
        )
        == "NST"
    )


def test_school_acronym_only(client: TestClient):
    _upload(client, "inst-acro", "profile.pdf", ACRONYM_ONLY)
    body = _chat(client, "inst-acro", "What school does the person attend?")
    assert "nst" in body["answer"].lower()
    assert "could not find" not in body["answer"].lower()
    # Do not invent a full expansion.
    assert "northshore" not in body["answer"].lower()


def test_school_full_name_only(client: TestClient):
    _upload(client, "inst-full", "profile.pdf", FULL_NAME_ONLY)
    body = _chat(client, "inst-full", "What school does the person attend?")
    assert "northshore" in body["answer"].lower()
    assert "could not find" not in body["answer"].lower()


def test_school_both_acronym_and_full_name(client: TestClient):
    _upload(client, "inst-both", "profile.pdf", BOTH)
    body = _chat(client, "inst-both", "What school does the person attend?")
    answer = body["answer"].lower()
    assert "northshore" in answer or "nst" in answer
    assert "could not find" not in answer


def test_school_missing_returns_fallback(client: TestClient):
    _upload(client, "inst-none", "profile.pdf", NO_SCHOOL)
    body = _chat(client, "inst-none", "What school does the person attend?")
    assert "could not find" in body["answer"].lower()
    assert "systems design" not in body["answer"].lower()
    assert "email" not in body["answer"].lower()


def test_multi_question_clean_answers(client: TestClient):
    _upload(client, "inst-multi", "profile.pdf", BOTH)
    body = _chat(
        client,
        "inst-multi",
        "Is the person involved in music?\n"
        "What group did the person found?\n"
        "What instrument does the person play?",
    )
    answer = body["answer"]
    lower = answer.lower()
    diag = body.get("diagnostics") or {}
    assert diag.get("sub_questions") and len(diag["sub_questions"]) == 3
    assert "music" in lower or "ensemble" in lower
    assert "harbor coding" in lower or "founded" in lower
    assert "could not find" in lower
    assert "[recruitment]" not in lower
    assert "…" not in answer and "..." not in answer
    # Founded answer should be a clean sentence, not a raw chunk dump.
    assert "established weekly peer tutoring" not in lower or "founded" in lower
    for part in (diag.get("sub_question_results") or []):
        sub = (part.get("answer") or "").lower()
        assert not sub.startswith("founder,")
        assert "[recruitment]" not in sub


def test_organization_established_question(client: TestClient):
    assert classify_query("What organization did the person establish?") == "leadership"
    _upload(client, "inst-est", "profile.pdf", FULL_NAME_ONLY)
    body = _chat(client, "inst-est", "What organization did the person establish?")
    lower = body["answer"].lower()
    assert "could not find" not in lower
    assert "harbor coding" in lower or "collective" in lower
    assert "…" not in body["answer"]
    assert not body["answer"].lower().startswith("founder,")
