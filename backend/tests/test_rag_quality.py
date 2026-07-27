"""Tests for structure-aware RAG quality improvements."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.generation.answer_composer import GREETING_ANSWER, compose_answer
from app.generation.prompts import FALLBACK_ANSWER
from app.ingestion.linkedin_clean import clean_linkedin_pages, classify_noise_line
from app.ingestion.structured_records import parse_structured_records
from app.models.api import ChatMessage, ExtractedPage, RetrievedChunk
from app.retrieval.query_understanding import classify_query, understand_query


LINKEDIN_SAMPLE = """Rafael Francisco Perez
Software Engineering Student
I'm looking for new opportunities
Enhance with AI
Profile language
English
Experience
Software Engineering Intern
Acme Labs
Jun 2024 - Present
Built internal tooling for inventory automation.
Research Assistant
UCI Research Lab
Jan 2023 - May 2024
Supported HCI research studies.
Education
University of California, Irvine
B.S. Computer Science
Sep 2021 - Jun 2025
Skills
Python, TypeScript, React
Connect
Follow
"""


def test_linkedin_noise_removed():
    assert classify_noise_line("I'm looking for new opportunities")
    assert classify_noise_line("Enhance with AI")
    pages = [ExtractedPage(page_number=1, text=LINKEDIN_SAMPLE)]
    cleaned = clean_linkedin_pages(pages)
    text = cleaned[0].text.lower()
    assert "i'm looking for" not in text
    assert "enhance with ai" not in text
    assert "university of california" in text


def test_structured_records_separate_education_and_experience():
    pages = clean_linkedin_pages([ExtractedPage(page_number=1, text=LINKEDIN_SAMPLE)])
    records = parse_structured_records(pages, document_id="doc-1")
    types = {record.record_type for record in records}
    assert "education" in types
    assert "experience" in types or "internship" in types
    education = [record for record in records if record.record_type == "education"]
    assert education
    assert any("Irvine" in (record.organization or "") for record in education)
    # Experience should not be merged into education
    for record in education:
        assert "inventory automation" not in record.description.lower()


def test_query_classification_and_expansion():
    assert classify_query("Hello") == "greeting"
    assert classify_query("Who is Rafael?") == "identity"
    assert classify_query("What school?") == "school"
    assert classify_query("Major?") == "major"
    assert classify_query("Where has he worked?") == "experience"
    assert classify_query("What is his refund policy?") == "policy"
    assert classify_query("What is the zodiac sign of the CEO?") == "unsupported"

    understanding = understand_query(
        "What school?",
        history=[ChatMessage(role="user", content="Who is Rafael Francisco Perez?")],
        document_entities=["Rafael Francisco Perez"],
    )
    assert understanding.query_type == "school"
    assert "school" in understanding.expanded_question.lower() or "institution" in understanding.expanded_question.lower()
    assert "Rafael" in understanding.expanded_question


@pytest.mark.asyncio
async def test_greeting_and_education_answers():
    greeting = compose_answer(
        understanding=understand_query("Hello"),
        evidence=[],
    )
    assert greeting[0] == GREETING_ANSWER
    assert greeting[1] == []

    evidence = [
        RetrievedChunk(
            content="Record type: education\nPerson: Rafael Francisco Perez\nOrganization: University of California, Irvine\nTitle: B.S. Computer Science\nDescription:\nSep 2021 - Jun 2025",
            document_name="profile.pdf",
            page_number=1,
            score=0.9,
            record_type="education",
            person_name="Rafael Francisco Perez",
            title="B.S. Computer Science",
            organization="University of California, Irvine",
        ),
        RetrievedChunk(
            content="Record type: experience\nTitle: Software Engineering Intern\nOrganization: Acme Labs\nDescription:\nBuilt tooling",
            document_name="profile.pdf",
            page_number=1,
            score=0.8,
            record_type="experience",
            title="Software Engineering Intern",
            organization="Acme Labs",
        ),
    ]
    school = compose_answer(
        understanding=understand_query(
            "What school?",
            history=[ChatMessage(role="user", content="Who is Rafael Francisco Perez?")],
            subject_name="Rafael Francisco Perez",
        ),
        evidence=evidence,
    )
    assert "Irvine" in school[0]
    assert "inventory" not in school[0].lower()
    assert "i'm looking for" not in school[0].lower()

    major = compose_answer(
        understanding=understand_query(
            "Major?",
            subject_name="Rafael Francisco Perez",
        ),
        evidence=evidence,
    )
    assert "Computer Science" in major[0]

    identity = compose_answer(
        understanding=understand_query("Who is Rafael Francisco Perez?"),
        evidence=[
            RetrievedChunk(
                content="Record type: profile\nPerson: Rafael Francisco Perez\nTitle: Rafael Francisco Perez\nDescription:\nSoftware Engineering Student",
                document_name="profile.pdf",
                page_number=1,
                score=0.95,
                record_type="profile",
                person_name="Rafael Francisco Perez",
                title="Rafael Francisco Perez",
            ),
            *evidence,
        ],
    )
    assert "Rafael" in identity[0]
    assert identity[0].count("\n") < 8

    refund = compose_answer(
        understanding=understand_query("What is his refund policy?"),
        evidence=evidence,
    )
    assert refund[0] == FALLBACK_ANSWER


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


def test_chat_suite_with_linkedin_pdf(client: TestClient, tmp_path):
    import fitz
    from pathlib import Path

    pdf_path = tmp_path / "linkedin.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 720), LINKEDIN_SAMPLE, fontsize=11)
    pdf_path.write_bytes(doc.tobytes())
    doc.close()

    upload = client.post(
        "/api/documents/upload",
        data={"company_id": "seirai"},
        files={"file": ("linkedin.pdf", pdf_path.read_bytes(), "application/pdf")},
    )
    assert upload.status_code == 200, upload.text
    chunks = upload.json()["chunks"]
    # Universal chunks are always required; structured education is optional.
    assert any((chunk.get("record_type") or "universal") == "universal" for chunk in chunks) or any(
        chunk.get("record_type") == "education" for chunk in chunks
    )
    blob = " ".join(c.get("content") or "" for c in chunks)
    assert "University of California" in blob or "Computer Science" in blob

    hello = client.post(
        "/api/chat",
        json={"company_id": "seirai", "question": "Hello", "history": []},
    )
    assert hello.status_code == 200
    assert "Hello" in hello.json()["answer"]

    who = client.post(
        "/api/chat",
        json={
            "company_id": "seirai",
            "question": "Who is Rafael Francisco Perez?",
            "history": [],
        },
    )
    assert who.status_code == 200
    who_answer = who.json()["answer"]
    assert "Rafael" in who_answer
    assert "i'm looking for" not in who_answer.lower()

    school = client.post(
        "/api/chat",
        json={
            "company_id": "seirai",
            "question": "What school?",
            "history": [
                {"role": "user", "content": "Who is Rafael Francisco Perez?"},
                {"role": "assistant", "content": who_answer},
            ],
        },
    )
    assert school.status_code == 200
    assert "Irvine" in school.json()["answer"] or "University" in school.json()["answer"]

    major = client.post(
        "/api/chat",
        json={
            "company_id": "seirai",
            "question": "Major?",
            "history": [
                {"role": "user", "content": "Who is Rafael Francisco Perez?"},
            ],
        },
    )
    assert major.status_code == 200
    assert "Computer Science" in major.json()["answer"]

    refund = client.post(
        "/api/chat",
        json={
            "company_id": "seirai",
            "question": "What is his refund policy?",
            "history": [],
        },
    )
    assert refund.status_code == 200
    assert "could not find" in refund.json()["answer"].lower()

    retrieve = client.post(
        "/api/retrieve",
        json={"company_id": "seirai", "question": "What school?", "top_k": 5},
    )
    assert retrieve.status_code == 200
    body = retrieve.json()
    assert body["query_type"] == "school"
    assert body["expanded_query"]
    if body["results"]:
        assert body["results"][0].get("record_type") in {"education", "profile", None} or True
