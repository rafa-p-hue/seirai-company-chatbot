"""Regression tests for labeled fields, help intents, and study vs research.

Uses invented fixtures only — no production PDF facts.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.ingestion.universal_chunker import create_universal_chunks
from app.models.api import ExtractedPage, SourceType
from app.retrieval.label_match import exact_label_match_score, label_family
from app.retrieval.query_understanding import classify_query


APPLICANT_FORM = """Nominee Information
Nominee's Full Name: Avery Quinn Hale
Nominee's Email: avery.hale@example.edu
Nominee's Major: Planetary Informatics
Nominee's Minor: Civic Media
Program Status: Nominee

Research Interests
Research Assistant, Horizon Sensing Lab - Assist in studies on coastal sensor networks.

Provide a list of accomplishments, involvements and activities.
Resident Advisor (2 Years) - Support a residential community of students and coordinate programming.
Orientation Leader - Helped welcome and guide incoming students through campus orientation.
Campus Concert Ensemble - Performs with the student music ensemble for campus events.
Founder, Civic Coding Club - Established weekly peer tutoring for introductory computing.
"""

POLICY_DOC = """Employee Handbook
Leave Policy: Employees receive 15 days of paid leave each year.
Remote Work Policy: Employees may request remote work after 90 days.
Effective Date: 2024-01-01
Managers are responsible for weekly check-ins during the first 90 days.
"""

PRODUCT_SPEC = """OrbitDock Specification Sheet
Product: OrbitDock Pro
Model: OD-200
Feature: Live Telemetry
Operators can export sensor logs as CSV.
Calibration Mode: Hold the mode button for three seconds.
"""

RESEARCH_REPORT = """Field Research Summary
Lead Author: Morgan Ellis
Research Topic: Afternoon temperature gradients in shaded corridors
Affiliation: Municipal Climate Office
Lead Analyst - Designed the sampling plan and trained three field assistants.
Field Assistant - Collected daily temperature readings across eight sites.
Finding: Shaded corridors remained cooler throughout the afternoon.
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


def test_help_intent_skips_retrieval(client: TestClient):
    _upload(client, "help-co", "applicant.pdf", APPLICANT_FORM)
    for question in (
        "Can you help?",
        "help",
        "what can you do",
        "how can you help",
    ):
        assert classify_query(question) == "help"
        body = _chat(client, "help-co", question)
        assert body.get("diagnostics", {}).get("retrieval_skipped") is True
        assert not body.get("diagnostics", {}).get("evidence_sent_to_llm")
        assert "document" in body["answer"].lower()
        assert "could not find" not in body["answer"].lower()
        assert "planetary" not in body["answer"].lower()


def test_key_value_ingestion_metadata():
    pages = [ExtractedPage(page_number=1, text=APPLICANT_FORM)]
    chunks = create_universal_chunks(
        pages=pages,
        company_id="c1",
        document_id="d1",
        document_name="form.pdf",
        source_type=SourceType.pdf,
    )
    kv = [chunk for chunk in chunks if chunk.content_type == "key_value"]
    assert kv
    labels = {chunk.label for chunk in kv}
    assert any(label and "Major" in label for label in labels)
    major = next(chunk for chunk in kv if chunk.label and "Major" in chunk.label)
    assert major.value and "Planetary" in major.value
    assert major.content == f"{major.label}: {major.value}"
    assert label_family(major.label or "") == "major"
    boost, family = exact_label_match_score(
        question="What does the person study?",
        query_type="major",
        payload={
            "content_type": "key_value",
            "label": major.label,
            "value": major.value,
            "content": major.content,
        },
    )
    assert family == "major"
    assert boost >= 0.8


def test_applicant_labeled_fact_questions(client: TestClient):
    _upload(client, "applicant-kv", "nominee-form.pdf", APPLICANT_FORM)

    who = _chat(client, "applicant-kv", "Who is the main person?")
    assert "avery" in who["answer"].lower()
    assert "applying" not in who["answer"].lower()
    assert "applicant" not in who["answer"].lower() or "nominee" in " ".join(
        c["content"].lower()
        for c in (who.get("diagnostics") or {}).get("evidence_sent_to_llm") or []
    )

    study = _chat(client, "applicant-kv", "What does the person study?")
    diag = study.get("diagnostics") or {}
    assert diag.get("detected_key_value_fields")
    assert diag.get("exact_label_matches")
    sent = diag.get("evidence_sent_to_llm") or []
    joined = " ".join(chunk["content"].lower() for chunk in sent)
    assert "major" in joined and "planetary" in joined
    assert "planetary" in study["answer"].lower()
    assert "coastal sensor" not in study["answer"].lower()

    major = _chat(client, "applicant-kv", "What is the major?")
    assert "planetary" in major["answer"].lower()
    assert "could not find" not in major["answer"].lower()

    his_major = _chat(
        client,
        "applicant-kv",
        "His major",
        history=[{"role": "user", "content": "Who is Avery Quinn Hale?"}],
    )
    assert "planetary" in his_major["answer"].lower()

    minor = _chat(client, "applicant-kv", "What is the minor?")
    assert "civic" in minor["answer"].lower()

    work = _chat(client, "applicant-kv", "Where does the person work?")
    work_sent = (work.get("diagnostics") or {}).get("evidence_sent_to_llm") or []
    work_joined = " ".join(chunk["content"].lower() for chunk in work_sent)
    role_hits = sum(
        1
        for token in ("advisor", "orientation", "research", "founder", "coding")
        if token in work_joined or token in work["answer"].lower()
    )
    assert role_hits >= 2
    assert "could not find" not in work["answer"].lower()

    instrument = _chat(client, "applicant-kv", "What instrument does the person play?")
    assert "could not find" in instrument["answer"].lower()


@pytest.mark.parametrize(
    "company_id,filename,text,question,needle",
    [
        ("policy-kv", "handbook.pdf", POLICY_DOC, "What is the leave policy?", "15 days"),
        ("product-kv", "spec.pdf", PRODUCT_SPEC, "What is the product?", "orbitdock"),
        (
            "research-kv",
            "report.pdf",
            RESEARCH_REPORT,
            "Who is the main person?",
            "morgan",
        ),
    ],
)
def test_multi_document_labeled_fields(
    client: TestClient, company_id, filename, text, question, needle
):
    _upload(client, company_id, filename, text)
    body = _chat(client, company_id, question)
    diag = body.get("diagnostics") or {}
    assert diag.get("detected_key_value_fields") is not None
    joined = body["answer"].lower() + " " + " ".join(
        chunk["content"].lower()
        for chunk in (diag.get("evidence_sent_to_llm") or [])
    )
    assert needle in joined
    assert "could not find" not in body["answer"].lower()


def test_research_topic_not_used_as_major(client: TestClient):
    _upload(client, "research-edu", "report.pdf", RESEARCH_REPORT)
    # Research report has a topic but no academic major label.
    body = _chat(client, "research-edu", "What is the major?")
    assert "could not find" in body["answer"].lower() or "temperature" not in body["answer"].lower()
