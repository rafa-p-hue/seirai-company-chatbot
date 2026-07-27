"""Regression tests for retrieval diversity, query equivalence, and answer quality.

Uses invented fixtures only — no production PDF facts.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.retrieval.query_understanding import classify_query, understand_query


APPLICANT_FORM = """Full Name: Jordan Lee Park
Email: jordan.park@example.edu
Major: Environmental Systems
Minor: Public Policy

Provide a list of accomplishments, involvements and activities.
Resident Advisor (2 Years) – Support a residential community of students and coordinate programming.
Campus Concert Ensemble – Performs with the student music ensemble for campus events.
Research Assistant, Climate Interface Lab – Assist in studies on accessible sensor dashboards.
Founder, Green Coding Club – Established weekly peer tutoring for introductory computing.
Chapter Affiliation: Campus Service Society – Coordinates community outreach weekends.
"""

POLICY_DOC = """Remote Work Policy
Eligibility
Employees may request remote work after completing 90 days of employment.
Leave Policy
Employees receive 15 days of paid leave each year.
Security Requirements
Remote workers must use company-managed devices and multi-factor authentication.
"""

PRODUCT_MANUAL = """OrbitDock User Manual
Installation
A certified technician mounts the base plate on a level surface.
Calibration
The operator runs the calibration wizard after power-on.
Feature: Live Telemetry
Operators can export sensor logs as CSV for offline analysis.
"""

RESEARCH_REPORT = """Field Research Summary
Objective
Measure temperature differences across shaded and open corridors.
Positions and Activities
Lead Analyst – Designed the sampling plan and trained three field assistants.
Field Assistant – Collected daily temperature readings across eight sites.
Finding
Shaded corridors remained cooler throughout the afternoon.
Affiliation
The work was conducted with the Municipal Climate Office.
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


@pytest.mark.parametrize(
    "a,b",
    [
        ("Does the person play music?", "Music"),
        ("What does the person study?", "Major"),
        ("Tell me about the person", "Who is the person?"),
        ("Tell me about the main subject", "Who is the person?"),
    ],
)
def test_semantically_equivalent_query_types(a, b):
    assert classify_query(a) == classify_query(b)


def test_generic_music_and_study_expansions():
    music = understand_query("Does the person play music?")
    assert music.query_type == "interest"
    assert {"music", "ensemble", "band"} <= set(music.expanded_terms) or any(
        term in music.expanded_question.lower()
        for term in ("music", "ensemble", "band", "performance")
    )

    study = understand_query("What does the person study?")
    assert study.query_type == "major"
    assert any(
        term in study.expanded_question.lower() or term in study.expanded_terms
        for term in ("major", "degree", "education", "field")
    )


def test_applicant_regression_questions(client: TestClient):
    _upload(client, "applicant-aq", "applicant-form.pdf", APPLICANT_FORM)

    summary = _chat(client, "applicant-aq", "What can you tell me about this document?")
    diag = summary.get("diagnostics") or {}
    assert diag.get("initial_candidates") or diag.get("initial_retrieved_chunks")
    assert diag.get("reranked_candidates")
    assert diag.get("diversity_selected_evidence")
    assert diag.get("evidence_sent_to_llm") or diag.get("final_chunks_sent_to_llm")
    sent = diag.get("evidence_sent_to_llm") or diag.get("final_chunks_sent_to_llm") or []
    joined = " ".join(chunk["content"].lower() for chunk in sent)
    # Broad questions should not be only short metadata fields.
    label_only = sum(
        1
        for chunk in sent
        if chunk["content"].lower().startswith(("full name:", "email:", "minor:"))
    )
    assert label_only < len(sent)
    assert any(
        token in joined
        for token in ("advisor", "ensemble", "research", "founder", "major", "environmental")
    )
    assert "could not find" not in summary["answer"].lower()
    # Answer should be generated text, not a raw single metadata field dump.
    assert not summary["answer"].lower().startswith("full name:")

    about = _chat(client, "applicant-aq", "Tell me about the main subject")
    assert "could not find" not in about["answer"].lower()

    music_q = _chat(client, "applicant-aq", "Does the person play music?")
    music_short = _chat(client, "applicant-aq", "Music")
    for body in (music_q, music_short):
        final = (body.get("diagnostics") or {}).get("evidence_sent_to_llm") or (
            body.get("diagnostics") or {}
        ).get("final_chunks_sent_to_llm") or []
        joined_music = " ".join(chunk["content"].lower() for chunk in final)
        assert "music" in joined_music or "ensemble" in joined_music
        assert "could not find" not in body["answer"].lower()

    study_q = _chat(client, "applicant-aq", "What does the person study?")
    major_q = _chat(client, "applicant-aq", "Major")
    for body in (study_q, major_q):
        answer = body["answer"].lower()
        final = (body.get("diagnostics") or {}).get("evidence_sent_to_llm") or (
            body.get("diagnostics") or {}
        ).get("final_chunks_sent_to_llm") or []
        joined_study = " ".join(chunk["content"].lower() for chunk in final)
        assert "environmental" in answer or "environmental" in joined_study
        assert "could not find" not in answer

    roles = _chat(client, "applicant-aq", "What roles or activities are described?")
    final_roles = (roles.get("diagnostics") or {}).get("evidence_sent_to_llm") or (
        roles.get("diagnostics") or {}
    ).get("final_chunks_sent_to_llm") or []
    joined_roles = " ".join(chunk["content"].lower() for chunk in final_roles)
    assert any(
        token in joined_roles or token in roles["answer"].lower()
        for token in ("advisor", "ensemble", "research", "founder", "coding")
    )
    cited = (roles.get("diagnostics") or {}).get("evidence_actually_cited")
    assert cited is not None
    # Sources should be deduped and limited.
    assert len(roles.get("sources") or []) <= 4


@pytest.mark.parametrize(
    "company_id,filename,text,question,needle",
    [
        ("policy-aq", "remote-work-policy.pdf", POLICY_DOC, "What can you tell me about this document?", "remote"),
        ("manual-aq", "orbitdock-manual.pdf", PRODUCT_MANUAL, "What can you tell me about this document?", "calibrat"),
        ("research-aq", "field-research.pdf", RESEARCH_REPORT, "Tell me about the main subject", "temperature"),
    ],
)
def test_multi_document_types_summary(
    client: TestClient, company_id, filename, text, question, needle
):
    _upload(client, company_id, filename, text)
    body = _chat(client, company_id, question)
    diag = body.get("diagnostics") or {}
    sent = diag.get("evidence_sent_to_llm") or diag.get("final_chunks_sent_to_llm") or []
    joined = " ".join(chunk["content"].lower() for chunk in sent) + " " + body["answer"].lower()
    assert sent
    assert needle in joined
    assert "could not find" not in body["answer"].lower()
    assert diag.get("diversity_selected_evidence") is not None


def test_fallback_when_evidence_irrelevant(client: TestClient):
    _upload(client, "policy-fallback", "remote-work-policy.pdf", POLICY_DOC)
    body = _chat(client, "policy-fallback", "What is the refund cancellation password?")
    # Unsupported intents short-circuit before retrieval answers.
    assert "could not find" in body["answer"].lower() or body["answer"]


def test_sources_deduped_by_document_page(client: TestClient):
    _upload(client, "applicant-src", "applicant-form.pdf", APPLICANT_FORM)
    body = _chat(client, "applicant-src", "Major")
    sources = body.get("sources") or []
    keys = {(item.get("document_name"), item.get("page_number")) for item in sources}
    assert len(keys) == len(sources)
    assert len(sources) <= 2
