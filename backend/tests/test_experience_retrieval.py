"""Regression tests for role/experience retrieval across document styles.

Prints retrieved evidence before asserting answers. Uses invented fixtures only.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.retrieval.query_understanding import classify_query, understand_query
from app.models.api import ChatMessage


APPLICANT_FORM = """Full Name: Casey Morgan Vale
Email: casey.vale@example.edu
Minor: Applied Ethics

Provide a list of accomplishments, involvements and activities.
Resident Advisor (2 Years) – Support a residential community of students and coordinate programming.
Orientation Leader – Helped welcome and guide incoming students through campus orientation.
Research Assistant, Interaction Studio – Assist in studies on accessible interface design.
Founder, Community Coding Club – Established weekly peer tutoring for introductory computing.
"""

RESUME = """Alex Rivera
Product Operations Associate

Experience
Operations Intern
Northwind Logistics
Jun 2023 - Aug 2023
Coordinated shipment audits and documented warehouse process improvements.
Research Assistant
Harbor Systems Lab
Jan 2022 - May 2023
Supported usability studies and prepared weekly experiment summaries.

Education
B.A. Systems Design
"""

HANDBOOK = """Employee Handbook
Onboarding Roles
Every new hire completes a Buddy Program assignment.
Managers are responsible for weekly check-ins during the first 90 days.
Safety officers conduct quarterly drills and maintain evacuation maps.
Leave Policy
Employees receive 15 days of paid leave each year.
"""

MANUAL = """OrbitDock User Manual
Installation Roles
A certified technician mounts the base plate.
The operator runs the calibration wizard after power-on.
Feature: Live Telemetry
Operators can export sensor logs as CSV.
"""

RESEARCH_REPORT = """Field Research Summary
Positions and Activities
Lead Analyst – Designed the sampling plan and trained three field assistants.
Field Assistant – Collected daily temperature readings across eight sites.
Method: Sensors recorded humidity and surface temperature every hour.
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
    # Use ASCII hyphen separators so PDF text extraction preserves list entries.
    safe = text.replace("–", "-").replace("—", "-")
    doc = fitz.open()
    page = doc.new_page()
    # Write line-by-line to avoid textbox overflow losing later list entries.
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


def _print_evidence(label: str, diagnostics: dict | None):
    print(f"\n===== {label} =====")
    if not diagnostics:
        print("No diagnostics returned")
        return
    print("original:", diagnostics.get("original_question"))
    print("normalized:", diagnostics.get("normalized_question"))
    print("resolved:", diagnostics.get("resolved_question"))
    print("intent:", diagnostics.get("query_intent"))
    print("--- initial candidates ---")
    for index, chunk in enumerate(diagnostics.get("initial_retrieved_chunks") or [], start=1):
        print(
            f"[{index}] vector={chunk.get('vector_similarity_score')} "
            f"lexical={chunk.get('lexical_score')} "
            f"rerank={chunk.get('reranker_score')} "
            f"quality={chunk.get('metadata_boosts', {}).get('quality_boost')}"
        )
        print("   ", (chunk.get("content") or "")[:180].replace("\n", " | "))
    print("--- final chunks sent to LLM ---")
    for index, chunk in enumerate(diagnostics.get("final_chunks_sent_to_llm") or [], start=1):
        print(f"[{index}]", (chunk.get("content") or "")[:200].replace("\n", " | "))


def test_pronoun_resolution_and_experience_intent():
    understanding = understand_query(
        "What are his experiences?",
        history=[ChatMessage(role="user", content="Who is Casey Morgan Vale?")],
        document_entities=["Casey Morgan Vale"],
    )
    assert understanding.query_type == "experience"
    assert "Casey" in understanding.resolved_question
    assert "experience" in understanding.expanded_question.lower()
    assert "roles" in understanding.expanded_terms
    assert "internships" in understanding.expanded_terms


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What are his experiences?", "experience"),
        ("What are his roles?", "experience"),
        ("What positions has the person held?", "experience"),
        ("What research experience is described?", "research"),
        ("What activities are listed?", "experience"),
    ],
)
def test_role_experience_classification(question, expected):
    assert classify_query(question) == expected


def test_applicant_form_experience_questions(client: TestClient):
    _upload(client, "applicant", "Casey Morgan Vale Applicant.pdf", APPLICANT_FORM)
    history = [{"role": "user", "content": "Who is Casey Morgan Vale?"}]
    questions = [
        "What are his experiences?",
        "What are his roles?",
        "What positions has the person held?",
        "What research experience is described?",
        "What activities are listed?",
    ]
    for question in questions:
        response = client.post(
            "/api/chat",
            json={"company_id": "applicant", "question": question, "history": history},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        _print_evidence(question, body.get("diagnostics"))
        print("ANSWER:", body["answer"])
        answer = body["answer"].lower()
        diagnostics = body.get("diagnostics") or {}
        final_chunks = diagnostics.get("final_chunks_sent_to_llm") or []
        assert final_chunks, f"Expected experience evidence for: {question}"
        joined = " ".join(chunk.get("content", "").lower() for chunk in final_chunks)
        assert "email" not in joined or "assistant" in joined or "advisor" in joined
        assert not any(
            chunk.get("content", "").lower().startswith("full name:")
            for chunk in final_chunks
        )
        assert "could not find" not in answer
        assert any(
            token in answer or token in joined
            for token in ("advisor", "orientation", "research", "founder", "coding", "assistant")
        )


def test_resume_experience_not_dominated_by_identity(client: TestClient):
    _upload(client, "resume", "Alex Rivera Resume.pdf", RESUME)
    response = client.post(
        "/api/chat",
        json={
            "company_id": "resume",
            "question": "What are his experiences?",
            "history": [{"role": "user", "content": "Who is Alex Rivera?"}],
        },
    )
    body = response.json()
    _print_evidence("resume experiences", body.get("diagnostics"))
    print("ANSWER:", body["answer"])
    final_chunks = (body.get("diagnostics") or {}).get("final_chunks_sent_to_llm") or []
    assert final_chunks
    joined = " ".join(chunk["content"].lower() for chunk in final_chunks)
    assert "intern" in joined or "assistant" in joined or "operations" in joined
    assert "could not find" not in body["answer"].lower()


def test_handbook_roles(client: TestClient):
    _upload(client, "handbook", "employee-handbook.pdf", HANDBOOK)
    response = client.post(
        "/api/chat",
        json={
            "company_id": "handbook",
            "question": "What positions has the person held?",
            "history": [],
        },
    )
    body = response.json()
    _print_evidence("handbook positions", body.get("diagnostics"))
    print("ANSWER:", body["answer"])
    # Handbook describes role responsibilities; should retrieve role content not leave-only.
    final_chunks = (body.get("diagnostics") or {}).get("final_chunks_sent_to_llm") or []
    joined = " ".join(chunk["content"].lower() for chunk in final_chunks)
    assert final_chunks
    assert any(token in joined for token in ("manager", "buddy", "safety", "officer", "check-ins", "drills"))


def test_manual_and_research_styles(client: TestClient):
    _upload(client, "manual", "orbitdock-manual.pdf", MANUAL)
    manual = client.post(
        "/api/chat",
        json={
            "company_id": "manual",
            "question": "What roles are described for installation?",
            "history": [],
        },
    )
    body = manual.json()
    _print_evidence("manual roles", body.get("diagnostics"))
    print("ANSWER:", body["answer"])
    assert body.get("diagnostics", {}).get("final_chunks_sent_to_llm")

    _upload(client, "research", "field-research.pdf", RESEARCH_REPORT)
    research = client.post(
        "/api/chat",
        json={
            "company_id": "research",
            "question": "What research experience is described?",
            "history": [],
        },
    )
    body = research.json()
    _print_evidence("research experience", body.get("diagnostics"))
    print("ANSWER:", body["answer"])
    final_chunks = (body.get("diagnostics") or {}).get("final_chunks_sent_to_llm") or []
    joined = " ".join(chunk["content"].lower() for chunk in final_chunks)
    assert "analyst" in joined or "assistant" in joined or "sampling" in joined
    assert "could not find" not in body["answer"].lower()
