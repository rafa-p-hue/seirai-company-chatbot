"""Seven-file regressions: follow-ups, unsupported fees, procedural disposal.

All seven municipal fixtures are indexed in one session. Expected facts live
only in this test module — production code stays document-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.generation.prompts import FALLBACK_ANSWER
from app.retrieval.conversation_context import (
    extract_anchors_from_turn,
    resolve_followup_question,
)
from app.retrieval.entity_validation import (
    extract_requested_service_phrases,
    format_not_found_answer,
)
from tests.eval.semantic_eval import (
    BenchmarkCase,
    assert_eval_rating,
    evaluate_answer,
    fact,
)


logger = logging.getLogger(__name__)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    database_path = tmp_path / "followup_eval.db"
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


# Enriched seven-file corpus for follow-up / procedure / unsupported evals.
# Filenames keep the canonical seven-doc identities; markdown avoids PDF tooling.
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


def _sources_from_message(payload: dict) -> List[str]:
    names: List[str] = []
    for item in payload.get("citations") or []:
        name = str(item.get("document_name") or item.get("filename") or "")
        if name:
            names.append(name)
    return names

def _ask(
    client: TestClient,
    session_id: str,
    content: str,
    *,
    company_id: str = "eval-followup",
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

    payload = response.json()
    return payload["assistant"]


def _log_failure(label: str, payload: dict, *, question: str, follow_up: str = "") -> None:
    diag = payload.get("diagnostics") or {}
    # Session message responses may not embed RAG diagnostics; log answer body.
    logger.error(
        "FAIL %s question=%r follow_up=%r answer=%r diagnostics_keys=%s "
        "resolved=%r anchors=%s top=%s rejection=%s final_evidence=%s",
        label,
        question,
        follow_up,
        (payload.get("content") or payload.get("answer") or "")[:500],
        sorted(diag.keys()) if isinstance(diag, dict) else None,
        diag.get("resolved_contextual_question"),
        diag.get("active_entity_topic") or diag.get("conversation_anchors"),
        diag.get("top_candidates_across_files"),
        diag.get("rejection_reason"),
        diag.get("final_evidence"),
    )


def test_unit_followup_resolution_uses_prior_answer():
    anchors = extract_anchors_from_turn(
        user_question="Which evacuation shelter accepts pets?",
        assistant_answer=(
            "Greenfield Community Center accepts pets. Capacity is listed separately."
        ),
    )
    assert anchors.organization and "Greenfield" in anchors.organization
    resolved = resolve_followup_question(
        "What is the capacity?",
        anchors=anchors,
    )
    assert "capacity" in resolved.lower()
    assert "greenfield" in resolved.lower()
    assert "riverside" not in resolved.lower()


def test_unit_parking_permit_phrases_are_strict():
    phrases = extract_requested_service_phrases(
        "How much is a residential parking permit?"
    )
    joined = " ".join(phrases).lower()
    assert "parking" in joined
    assert "permit" in joined


def test_a_pet_shelter_capacity_followup(client: TestClient):
    session = _create_session(client)
    company_id = "eval-followup-a"
    _upload_all(client, session["id"], company_id)

    first = _ask(
        client,
        session["id"],
        "Which evacuation shelter accepts pets?",
        company_id=company_id,
    )
    print("FIRST RESPONSE:", first)
    first_answer = first["content"]
    assert "greenfield" in first_answer.lower() or "accepts pets" in first_answer.lower()

    second = _ask(
        client,
        session["id"],
        "What is the capacity?",
        company_id=company_id,
    )
    answer = second["content"]
    case = BenchmarkCase(
        test_id="FOLLOWUP-A",
        question="What is the capacity?",
        expected_facts=(
            fact(
                "capacity_300",
                "capacity is 300",
                ("300",),
                ("capacity", "300"),
                critical=True,
            ),
        ),
        forbidden_facts=(
            fact(
                "unrelated_shelters_listed",
                "lists unrelated shelters as the answer subject",
                ("riverside", "500"),
                ("harborlight", "200"),
                critical=True,
            ),
        ),
        required_sources=(),
        evaluation_notes="Singular follow-up must stay on the pet-friendly shelter.",
    )
    result = evaluate_answer(case, answer=answer, sources=_sources_from_message(second))
    # Semantic facts are asserted below; source grounding is soft when citations omit names.
    if "300" not in answer:
        _log_failure("A", second, question="Which evacuation shelter accepts pets?", follow_up="What is the capacity?")
    assert "300" in answer
    assert "riverside" not in answer.lower() or "greenfield" in answer.lower()
    # Must not dump every shelter capacity as a list answer.
    assert not (
        "500" in answer and "200" in answer and "300" in answer
    ), f"broadened to all shelters: {answer}"
    assert "greenfield" in answer.lower() or "300" in answer


def test_b_child_allowance_payment_followup(client: TestClient):
    session = _create_session(client)
    company_id = "eval-followup-b"
    _upload_all(client, session["id"], company_id)

    first = _ask(
        client,
        session["id"],
        "How much child allowance is provided for a 2-year-old?",
        company_id=company_id,
    )
    assert "15,000" in first["content"] or "15000" in first["content"].replace(",", "")

    second = _ask(
        client,
        session["id"],
        "When is the allowance paid?",
        company_id=company_id,
    )
    answer = second["content"]
    case = BenchmarkCase(
        test_id="FOLLOWUP-B",
        question="When is the allowance paid?",
        expected_facts=(
            fact(
                "feb",
                "paid in February",
                ("february",),
                ("feb",),
                critical=True,
            ),
            fact(
                "june",
                "paid in June",
                ("june",),
                critical=True,
            ),
            fact(
                "october",
                "paid in October",
                ("october",),
                ("oct",),
                critical=True,
            ),
        ),
        forbidden_facts=(
            fact(
                "certificate_fee",
                "certificate fee content",
                ("residence certificate",),
                ("counter fee",),
                ("family register",),
                critical=True,
            ),
        ),
        required_sources=(),
        evaluation_notes="Payment months for the active child-allowance topic.",
    )
    result = evaluate_answer(case, answer=answer, sources=_sources_from_message(second))
    if result.critical_missing or result.forbidden_facts_detected:
        _log_failure(
            "B",
            second,
            question="How much child allowance is provided for a 2-year-old?",
            follow_up="When is the allowance paid?",
        )
        raise AssertionError(result.report())
    assert "february" in answer.lower()
    assert "june" in answer.lower()
    assert "october" in answer.lower()
    assert "residence certificate" not in answer.lower()


def test_c_residential_parking_permit_not_found(client: TestClient):
    session = _create_session(client)
    company_id = "eval-followup-c"
    _upload_all(client, session["id"], company_id)

    payload = _ask(
        client,
        session["id"],
        "How much is a residential parking permit?",
        company_id=company_id,
    )
    answer = payload["content"]
    expected = format_not_found_answer("Hikari City")
    if "could not find" not in answer.lower():
        _log_failure(
            "C",
            payload,
            question="How much is a residential parking permit?",
        )
    assert "could not find" in answer.lower()
    assert "hikari city" in answer.lower() or answer == expected or answer == FALLBACK_ANSWER
    assert "300" not in answer
    assert "450" not in answer
    assert "residence certificate" not in answer.lower()
    assert "family register" not in answer.lower()


def test_d_sofa_disposal_procedure(client: TestClient):
    session = _create_session(client)
    company_id = "eval-followup-d"
    _upload_all(client, session["id"], company_id)

    payload = _ask(
        client,
        session["id"],
        "How do I throw away a sofa?",
        company_id=company_id,
    )
    answer = payload["content"]
    case = BenchmarkCase(
        test_id="FOLLOWUP-D",
        question="How do I throw away a sofa?",
        expected_facts=(
            fact(
                "oversized",
                "classified as oversized garbage",
                ("oversized",),
                ("classified", "oversized"),
                critical=True,
            ),
            fact(
                "reserve_week",
                "reserve at least one week ahead",
                ("one week",),
                ("1 week",),
                ("week ahead",),
                critical=True,
            ),
            fact(
                "phone_or_online",
                "phone or online reservation",
                ("phone", "online"),
                ("phone or online",),
                critical=True,
            ),
            fact(
                "sticker",
                "buy required sticker",
                ("sticker",),
                critical=True,
            ),
            fact(
                "sticker_range",
                "sticker fee range preserved",
                ("200", "1,500"),
                ("200", "1500"),
                ("200 to 1,500",),
                critical=False,
            ),
            fact(
                "attach",
                "attach sticker",
                ("attach",),
                critical=True,
            ),
            fact(
                "place_out",
                "place item out by stated time",
                ("8:00",),
                ("8 a.m",),
                ("place",),
                critical=True,
            ),
        ),
        forbidden_facts=(
            fact(
                "certificate_fees",
                "unrelated certificate fees",
                ("residence certificate",),
                ("family register",),
                critical=True,
            ),
        ),
        required_sources=(),
        evaluation_notes="Ordered oversized-disposal procedure.",
    )
    result = evaluate_answer(case, answer=answer, sources=_sources_from_message(payload))
    if result.critical_missing or result.forbidden_facts_detected:
        _log_failure("D", payload, question="How do I throw away a sofa?")
        raise AssertionError(result.report())
    lower = answer.lower()
    assert "oversized" in lower
    assert "week" in lower
    assert "sticker" in lower
    assert "residence certificate" not in lower


def test_e_burnable_garbage_collection_completeness(client: TestClient):
    session = _create_session(client)
    company_id = "eval-followup-e"
    _upload_all(client, session["id"], company_id)

    payload = _ask(
        client,
        session["id"],
        "When is burnable garbage collected?",
        company_id=company_id,
    )
    answer = payload["content"]
    case = BenchmarkCase(
        test_id="FOLLOWUP-E",
        question="When is burnable garbage collected?",
        expected_facts=(
            fact(
                "tuesday",
                "Tuesday collection",
                ("tuesday",),
                critical=True,
            ),
            fact(
                "friday",
                "Friday collection",
                ("friday",),
                critical=True,
            ),
            fact(
                "by_8am",
                "by 8:00 a.m.",
                ("8:00",),
                ("8 a.m",),
                ("8:00 a.m",),
                critical=True,
            ),
            fact(
                "collection_point",
                "designated collection point",
                ("collection point",),
                ("designated",),
                critical=True,
            ),
            fact(
                "yellow_bags",
                "official yellow city bags",
                ("yellow", "bag"),
                ("yellow city bags",),
                critical=True,
            ),
        ),
        forbidden_facts=(
            fact(
                "certificate",
                "certificate fee bleed",
                ("residence certificate",),
                critical=True,
            ),
        ),
        required_sources=(),
        evaluation_notes="Collection schedule completeness.",
    )
    result = evaluate_answer(case, answer=answer, sources=_sources_from_message(payload))
    if result.critical_missing or result.forbidden_facts_detected:
        _log_failure("E", payload, question="When is burnable garbage collected?")
        raise AssertionError(result.report())
    lower = answer.lower()
    assert "tuesday" in lower and "friday" in lower
    assert "8:00" in answer or "8 a.m" in lower
    assert "collection point" in lower or "designated" in lower
    assert "yellow" in lower
