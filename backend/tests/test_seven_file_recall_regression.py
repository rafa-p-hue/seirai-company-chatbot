"""Seven-file recall + domain-accuracy regressions (cases A–H).

WARNING: Markdown CORPUS below is synthetic and outdated vs live uploads
(live current fees ¥350/¥250; pets in PPTX; sofa in HTML). See
tests/fixtures/OUTDATED_FIXTURE_WARNING.md. Do not cite ¥300/¥200 as a
live current-fee pass.

All seven municipal fixtures are indexed in one session. Expected facts live
only in this test module — production code stays document-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from app.retrieval.entity_validation import format_not_found_answer
from app.retrieval.query_understanding import classify_query
from app.ingestion.service_domain import extract_query_service_domain
from app.retrieval.conversation_context import is_elliptical_followup_question
from app.retrieval.entity_validation import extract_requested_service_phrases


logger = logging.getLogger(__name__)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    database_path = tmp_path / "seven_recall.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("NOT_FOUND_CONTACT_NAME", "Hikari City")
    monkeypatch.chdir(tmp_path)

    from app import dependencies
    from app.config import get_settings
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


def _md(lines: List[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


CORPUS: List[Dict[str, Any]] = [
    {
        "filename": "Resident_Registration_Moving_In.md",
        "lines": [
            "# Resident Registration and Moving-In Guide",
            "Status: current",
            "## Moving In Notification",
            "You must register within 14 days of moving in.",
            "Bring your residence card, passport or national ID.",
            "Submit the move-in notification at the Citizen Services Window.",
        ],
    },
    {
        "filename": "National_Health_Insurance_Enrollment.md",
        "lines": [
            "# National Health Insurance Enrollment Guide",
            "Status: current",
            "If you leave your employer's insurance, enroll within 14 days.",
            "Bring your residence card and certificate of loss of eligibility.",
            "## Patient Cost Share",
            "With National Health Insurance, you normally pay 30% of medical costs at the counter.",
            "The patient share for insured treatment is 30 percent.",
            "Children under elementary school age may pay a lower co-payment under a separate child medical subsidy.",
        ],
    },
    {
        "filename": "Child_Allowance_Application.md",
        "lines": [
            "# Child Allowance and Childcare Support Guide",
            "Status: current",
            "## Child Allowance Amounts",
            "For a 2-year-old child, child allowance is 15,000 yen per month.",
            "For children aged 3 to 12, child allowance is 10,000 yen per month.",
            "## Child Allowance Payment Schedule",
            "The allowance is paid in February, June, and October.",
            "Apply at the Child Welfare Desk within 15 days of moving.",
        ],
    },
    {
        "filename": "Waste_and_Recycling_Collection.md",
        "lines": [
            "# Waste and Recycling Collection Guide",
            "Status: current",
            "## Household Waste Collection Days",
            "Burnable garbage is collected every Tuesday and Friday.",
            "Put bags at the designated collection point by 8:00 a.m.",
            "Use official yellow city bags for burnable garbage.",
            "Burnable garbage summary: Tuesday and Friday by 8:00 a.m. at the designated collection point using official yellow city bags.",
            "Recyclables are collected every Wednesday.",
            "## Oversized Garbage Disposal",
            "Sofas are classified as oversized garbage.",
            "Reserve collection at least one week ahead by phone or online.",
            "Buy the required oversized-garbage sticker.",
            "Sticker fees range from 200 to 1,500 yen depending on size.",
            "Attach the sticker to the item.",
            "Place the item out by 8:00 a.m. on the reserved collection day.",
        ],
    },
    {
        "filename": "Certificates_and_Fees_Current.md",
        "lines": [
            "# Certificates and Fees — Current Fee Schedule",
            "Status: current",
            "Effective date: April 1, 2025",
            "## Certificate Issuance Fees",
            "Residence certificate counter fee: 300 yen.",
            "Residence certificate kiosk fee: 200 yen.",
            "Family register abstract: 450 yen.",
        ],
    },
    {
        "filename": "Certificates_and_Fees_Archived_2024.md",
        "lines": [
            "# Certificates and Fees — Archived Fee Schedule 2024",
            "Status: archived",
            "Residence certificate counter fee: 400 yen.",
            "This archived schedule is no longer valid.",
        ],
    },
    {
        "filename": "Disaster_Preparedness_Guide.md",
        "lines": [
            "# Disaster Preparedness Guide",
            "Status: current",
            "## Emergency Stockpile Recommendations",
            "Keep at least 3 liters of water per person per day.",
            "## Evacuation Shelters",
            "Greenfield Community Center accepts pets. Capacity: 300 people.",
            "Riverside Gym does not accept pets. Capacity: 500 people.",
            "Harborlight School accepts pets only in carriers. Capacity: 200 people.",
            "Pet-friendly shelter summary: Greenfield Community Center accepts pets and has capacity for 300 people.",
        ],
    },
]


def _create_session(client: TestClient) -> dict:
    response = client.post("/api/chat/sessions")
    assert response.status_code == 201, response.text
    return response.json()


def _upload_all(client: TestClient, session_id: str, company_id: str) -> None:
    for item in CORPUS:
        response = client.post(
            f"/api/chat/sessions/{session_id}/attachments",
            data={"company_id": company_id},
            files={
                "file": (
                    item["filename"],
                    _md(item["lines"]),
                    "text/markdown",
                )
            },
        )
        assert response.status_code == 201, response.text


def _ask(
    client: TestClient,
    session_id: str,
    content: str,
    *,
    company_id: str,
) -> dict:
    response = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "company_id": company_id,
            "content": content,
            "top_k": 8,
            "include_company_docs": False,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _answer(payload: dict) -> str:
    return str(payload.get("content") or payload.get("answer") or "")


def _sources(payload: dict) -> List[str]:
    names: List[str] = []
    for item in payload.get("citations") or payload.get("sources") or []:
        name = str(item.get("document_name") or item.get("filename") or "")
        if name:
            names.append(name)
    return names


def _rate(
    *,
    required: List[bool],
    forbidden: List[bool],
) -> str:
    if any(forbidden):
        return "FAIL"
    if all(required):
        return "PASS"
    if any(required):
        return "PARTIAL"
    return "FAIL"


def _report(case_id: str, rating: str, **details: Any) -> None:
    logger.info("CASE %s => %s | %s", case_id, rating, details)
    print(f"CASE {case_id}: {rating} | {details}")


def test_unit_standalone_questions_are_not_elliptical():
    assert not is_elliptical_followup_question(
        "What share of medical costs do I pay with National Health Insurance?"
    )
    assert not is_elliptical_followup_question(
        "How much child allowance does a 2-year-old receive?"
    )
    assert extract_query_service_domain(
        "What share of medical costs do I pay with National Health Insurance?"
    ) == "health_insurance"
    assert extract_query_service_domain(
        "How much child allowance does a 2-year-old receive?"
    ) == "childcare_support"
    assert classify_query(
        "What share of medical costs do I pay with National Health Insurance?"
    ) == "policy"
    phrases = extract_requested_service_phrases(
        "How much child allowance does a 2-year-old receive?"
    )
    assert any("child allowance" in p.lower() for p in phrases)


def test_a_burnable_garbage_schedule(client: TestClient):
    session = _create_session(client)
    company_id = "seven-a"
    _upload_all(client, session["id"], company_id)
    payload = _ask(
        client,
        session["id"],
        "When is burnable garbage collected?",
        company_id=company_id,
    )
    answer = _answer(payload).lower()
    sources = _sources(payload)
    required = [
        "tuesday" in answer and "friday" in answer,
        "8:00" in answer or "8 a.m" in answer or "8am" in answer,
        "collection point" in answer or "designated" in answer,
        "yellow" in answer and "bag" in answer,
    ]
    forbidden = ["recyclable" in answer and "burnable" not in answer]
    rating = _rate(required=required, forbidden=forbidden)
    _report(
        "A",
        rating,
        sources=sources,
        answer=_answer(payload)[:240],
        required=required,
    )
    assert rating == "PASS", (answer, sources)


def test_b_sofa_disposal(client: TestClient):
    session = _create_session(client)
    company_id = "seven-b"
    _upload_all(client, session["id"], company_id)
    payload = _ask(
        client,
        session["id"],
        "How do I throw away a sofa?",
        company_id=company_id,
    )
    answer = _answer(payload).lower()
    sources = _sources(payload)
    required = [
        "oversized" in answer,
        "week" in answer,
        "phone" in answer or "online" in answer,
        "sticker" in answer,
        "200" in answer or "1,500" in answer or "1500" in answer,
        "attach" in answer,
        "8:00" in answer or "8 a.m" in answer,
    ]
    rating = _rate(required=required, forbidden=[False])
    _report("B", rating, sources=sources, answer=_answer(payload)[:300])
    assert rating in {"PASS", "PARTIAL"}
    assert "oversized" in answer
    assert "sticker" in answer


def test_c_residence_certificate_fee_after_unrelated(client: TestClient):
    session = _create_session(client)
    company_id = "seven-c"
    _upload_all(client, session["id"], company_id)
    # Prior turn should not poison fee retrieval.
    _ask(
        client,
        session["id"],
        "How do I throw away a sofa?",
        company_id=company_id,
    )
    payload = _ask(
        client,
        session["id"],
        "How much does a residence certificate cost?",
        company_id=company_id,
    )
    answer = _answer(payload).lower()
    sources = _sources(payload)
    required = [
        "300" in answer,
        "200" in answer,
        any("current" in s.lower() for s in sources)
        or "certificates_and_fees_current" in " ".join(sources).lower(),
    ]
    forbidden = ["400" in answer]
    rating = _rate(required=required, forbidden=forbidden)
    _report("C", rating, sources=sources, answer=_answer(payload)[:300])
    assert rating == "PASS", (answer, sources)
    assert "could not find" not in answer


def test_d_move_in_registration(client: TestClient):
    session = _create_session(client)
    company_id = "seven-d"
    _upload_all(client, session["id"], company_id)
    deadline = _ask(
        client,
        session["id"],
        "What is the move-in registration deadline?",
        company_id=company_id,
    )
    docs = _ask(
        client,
        session["id"],
        "What do I bring for move-in registration?",
        company_id=company_id,
    )
    deadline_a = _answer(deadline).lower()
    docs_a = _answer(docs).lower()
    sources = _sources(docs)
    required = [
        "14" in deadline_a and "day" in deadline_a,
        "citizen services" in docs_a or "window" in docs_a,
        "residence card" in docs_a,
        "passport" in docs_a or "national id" in docs_a,
    ]
    rating = _rate(required=required, forbidden=["could not find" in docs_a])
    _report(
        "D",
        rating,
        sources=sources,
        deadline=_answer(deadline)[:200],
        docs=_answer(docs)[:300],
    )
    assert "14" in deadline_a
    assert "residence card" in docs_a
    assert rating in {"PASS", "PARTIAL"}


def test_e_child_allowance_age_2_after_fee(client: TestClient):
    session = _create_session(client)
    company_id = "seven-e"
    _upload_all(client, session["id"], company_id)
    _ask(
        client,
        session["id"],
        "How much does a residence certificate cost?",
        company_id=company_id,
    )
    payload = _ask(
        client,
        session["id"],
        "How much child allowance does a 2-year-old receive?",
        company_id=company_id,
    )
    answer = _answer(payload).lower()
    sources = _sources(payload)
    required = [
        "15,000" in answer or "15000" in answer,
        "child allowance" in answer or "allowance" in answer,
        any("child" in s.lower() for s in sources),
    ]
    forbidden = [
        "could not find" in answer,
        "10,000" in answer and "15,000" not in answer and "15000" not in answer,
        any("certificate" in s.lower() and "child" not in s.lower() for s in sources)
        and "15,000" not in answer,
    ]
    rating = _rate(required=required, forbidden=forbidden)
    _report("E", rating, sources=sources, answer=_answer(payload)[:300])
    assert rating == "PASS", (answer, sources)


def test_f_nhi_share_after_child_allowance(client: TestClient):
    session = _create_session(client)
    company_id = "seven-f"
    _upload_all(client, session["id"], company_id)
    _ask(
        client,
        session["id"],
        "How much child allowance does a 2-year-old receive?",
        company_id=company_id,
    )
    payload = _ask(
        client,
        session["id"],
        "What share of medical costs do I pay with National Health Insurance?",
        company_id=company_id,
    )
    answer = _answer(payload).lower()
    sources = _sources(payload)
    required = [
        "30" in answer and ("%" in answer or "percent" in answer),
        any("health" in s.lower() or "insurance" in s.lower() for s in sources),
    ]
    forbidden = [
        "15,000" in answer,
        "child allowance" in answer and "30" not in answer,
        any("child_allowance" in s.lower() for s in sources)
        and not any("health" in s.lower() or "insurance" in s.lower() for s in sources),
    ]
    rating = _rate(required=required, forbidden=forbidden)
    _report("F", rating, sources=sources, answer=_answer(payload)[:300])
    assert rating == "PASS", (answer, sources)


def test_g_pet_friendly_shelter(client: TestClient):
    session = _create_session(client)
    company_id = "seven-g"
    _upload_all(client, session["id"], company_id)
    # Prior unrelated turn must not block disaster retrieval.
    _ask(
        client,
        session["id"],
        "How much child allowance does a 2-year-old receive?",
        company_id=company_id,
    )
    payload = _ask(
        client,
        session["id"],
        "Which evacuation shelter accepts pets?",
        company_id=company_id,
    )
    answer = _answer(payload).lower()
    sources = _sources(payload)
    required = [
        "greenfield" in answer,
        "accepts pets" in answer or "pet" in answer,
        any("disaster" in s.lower() for s in sources),
    ]
    forbidden = ["could not find" in answer]
    rating = _rate(required=required, forbidden=forbidden)
    _report("G", rating, sources=sources, answer=_answer(payload)[:300])
    assert rating == "PASS", (answer, sources)


def test_h_residential_parking_permit_not_found(client: TestClient):
    session = _create_session(client)
    company_id = "seven-h"
    _upload_all(client, session["id"], company_id)
    payload = _ask(
        client,
        session["id"],
        "How much is a residential parking permit?",
        company_id=company_id,
    )
    answer = _answer(payload)
    expected = format_not_found_answer("Hikari City")
    rating = "PASS" if answer.strip() == expected.strip() else "FAIL"
    _report("H", rating, answer=answer[:300], sources=_sources(payload))
    assert answer.strip() == expected.strip()
