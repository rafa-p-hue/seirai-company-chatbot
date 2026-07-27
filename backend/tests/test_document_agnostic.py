"""Document-agnostic RAG tests across multiple synthetic document types.

Fixtures use invented content only. Production code must not hardcode these facts.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.ingestion.universal_chunker import create_universal_chunks
from app.models.api import ExtractedPage
from app.retrieval.query_understanding import classify_query, understand_query


RESUME_TEXT = """Full Name: Jordan Lee Quill
Email: jordan.quill@example.edu
Education
Northbridge State University
B.S. Applied Computing
Experience
Software Intern
Brightleaf Systems
Built internal dashboards for operations teams.
Research Assistant
Coastal Interaction Lab
Studied accessibility in voice interfaces.
"""

POLICY_TEXT = """Company Leave Policy
Eligible employees may take up to 15 days of paid leave each year.
Leave requests must be submitted at least 5 business days in advance.
Unused leave does not carry over after December 31.
Emergency leave may be approved by a department manager.
"""

MANUAL_TEXT = """PulseMeter Pro User Manual
Feature: Instant Calibration
To calibrate, press and hold the Mode button for three seconds.
Feature: Data Export
Exported readings are saved as CSV files on the device storage.
Do not immerse the housing in liquids.
"""

RESEARCH_TEXT = """Annual Research Report
Study Title: Urban Heat Mapping with Low-Cost Sensors
Method: Deployed 40 sensors across three neighborhoods.
Finding: Afternoon temperatures were 3.2C higher near asphalt corridors.
Recommendation: Increase tree canopy along major transit routes.
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


def test_universal_chunks_preserve_labels_and_pages():
    pages = [ExtractedPage(page_number=1, text=RESUME_TEXT)]
    chunks = create_universal_chunks(
        pages=pages,
        company_id="demo",
        document_id="d1",
        document_name="jordan-quill-resume.pdf",
    )
    assert len(chunks) >= 3
    assert all(chunk.record_type == "universal" for chunk in chunks)
    assert any(chunk.page_number == 1 for chunk in chunks)
    assert any("Full Name:" in chunk.content for chunk in chunks)
    assert any(chunk.document_name.endswith(".pdf") for chunk in chunks)


def test_generic_intent_expansion_has_no_hardcoded_person():
    understanding = understand_query(
        "school",
        document_entities=["Jordan Lee Quill"],
        document_name="jordan-quill-resume.pdf",
    )
    assert understanding.query_type == "school"
    assert "institution" in understanding.expanded_question.lower() or "school" in understanding.expanded_question.lower()
    assert "jordan" in understanding.expanded_question.lower()
    # Ensure no unrelated hardcoded names appear in production expansions.
    from pathlib import Path
    import re

    query_file = Path(__file__).resolve().parents[1] / "app" / "retrieval" / "query_understanding.py"
    source = query_file.read_text(encoding="utf-8")
    assert not re.search(r"rafael", source, re.I)


def test_query_types_are_generic():
    assert classify_query("Who is Jordan?") == "identity"
    assert classify_query("What school?") == "school"
    assert classify_query("role") == "experience"
    assert classify_query("What are his experiences?") == "experience"
    assert classify_query("How do I calibrate the device?") == "product"
    assert classify_query("What is the leave policy?") == "policy"


def _upload(client: TestClient, company_id: str, filename: str, text: str):
    response = client.post(
        "/api/documents/upload",
        data={"company_id": company_id},
        files={"file": (filename, _pdf_bytes(text), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document"]["universal_chunk_count"] >= 1
    return body


def test_resume_profile_document(client: TestClient):
    body = _upload(client, "docs", "Jordan Lee Quill Resume.pdf", RESUME_TEXT)
    assert any(chunk.get("record_type") == "universal" for chunk in body["chunks"])

    who = client.post(
        "/api/chat",
        json={"company_id": "docs", "question": "Who is Jordan?", "history": []},
    )
    assert who.status_code == 200
    assert "Jordan" in who.json()["answer"]

    school = client.post(
        "/api/chat",
        json={
            "company_id": "docs",
            "question": "What school?",
            "history": [{"role": "user", "content": "Who is Jordan Lee Quill?"}],
        },
    )
    assert "Northbridge" in school.json()["answer"] or "University" in school.json()["answer"]


def test_company_policy_document(client: TestClient):
    _upload(client, "policyco", "leave-policy.pdf", POLICY_TEXT)
    answer = client.post(
        "/api/chat",
        json={
            "company_id": "policyco",
            "question": "How many paid leave days are allowed each year?",
            "history": [],
        },
    )
    body = answer.json()["answer"].lower()
    assert "15" in body or "leave" in body
    assert "could not find" not in body


def test_product_manual_document(client: TestClient):
    _upload(client, "productco", "pulsemeter-manual.pdf", MANUAL_TEXT)
    answer = client.post(
        "/api/chat",
        json={
            "company_id": "productco",
            "question": "How do I calibrate the device?",
            "history": [],
        },
    )
    text = answer.json()["answer"].lower()
    assert "calibrat" in text or "mode" in text
    assert "could not find" not in text


def test_research_report_document(client: TestClient):
    _upload(client, "researchco", "heat-mapping-report.pdf", RESEARCH_TEXT)
    answer = client.post(
        "/api/chat",
        json={
            "company_id": "researchco",
            "question": "What was the main finding?",
            "history": [],
        },
    )
    text = answer.json()["answer"].lower()
    assert "temperature" in text or "asphalt" in text or "3.2" in text
    assert "could not find" not in text


def test_insufficient_evidence_returns_fallback(client: TestClient):
    _upload(client, "policyco2", "leave-policy.pdf", POLICY_TEXT)
    answer = client.post(
        "/api/chat",
        json={
            "company_id": "policyco2",
            "question": "What is the zodiac sign of the CEO?",
            "history": [],
        },
    )
    assert "could not find" in answer.json()["answer"].lower()
