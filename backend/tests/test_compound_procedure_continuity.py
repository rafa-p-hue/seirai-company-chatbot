"""Regression: compound multi-file procedure continuity (Tests A/B/C).

All seven municipal fixtures are attached to one chat. Evaluation is semantic
(not exact wording). Benchmark facts stay in tests only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import fitz
import pytest
from fastapi.testclient import TestClient

from app.ingestion.service_domain import HEALTH_INSURANCE, RESIDENT_REGISTRATION
from app.retrieval.procedure_context import (
    assign_subquestion_domains,
    build_procedure_context,
    question_explicitly_spans_domains,
)
from app.retrieval.query_understanding import resolve_compound_subquestions
from app.retrieval.multi_question import split_questions
from tests.eval.semantic_eval import (
    BenchmarkCase,
    assert_eval_rating,
    evaluate_answer,
    fact,
)


logger = logging.getLogger(__name__)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    database_path = tmp_path / "compound_continuity.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "development")
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


def _pdf(lines: Sequence[str]) -> bytes:
    document = fitz.open()
    page = document.new_page()
    y = 48
    for line in lines:
        remaining = line
        while remaining:
            chunk = remaining[:92]
            if len(remaining) > 92:
                split = chunk.rfind(" ")
                if split > 40:
                    chunk = remaining[:split]
                    remaining = remaining[split:].lstrip()
                else:
                    remaining = remaining[92:]
            else:
                remaining = ""
            page.insert_text((40, y), chunk, fontsize=10)
            y += 13
            if y > 780:
                page = document.new_page()
                y = 48
    data = document.tobytes()
    document.close()
    return data


CORPUS = [
    (
        "Resident_Registration_Moving_In.pdf",
        [
            "Resident Registration and Moving-In Guide",
            "Status: current",
            "Moving In Notification",
            "Submit a move-in notification at Citizen Affairs Window 3.",
            "You must register within 14 days of beginning to live at the new address.",
            "Bring the following:",
            "• Residence card, or passport with landing permission",
            "• Moving-Out Certificate when moving from another municipality",
            "• My Number cards or notification cards for all household members",
        ],
    ),
    (
        "National_Health_Insurance_Enrollment.pdf",
        [
            "National Health Insurance Enrollment Guide",
            "Status: current",
            "Insurance Enrollment After Leaving Employer Insurance",
            "If you leave your employer's insurance, enroll in national health insurance.",
            "You must enroll within 14 days of losing employer coverage.",
            "Bring your residence card, certificate of loss of eligibility, and personal seal.",
            "Submit enrollment at Insurance & Pension Window 6.",
            "The NHI card is mailed separately.",
        ],
    ),
    (
        "Child_Allowance_Application.pdf",
        [
            "Child Allowance and Childcare Support Guide",
            "To receive child allowance after moving here, apply within 15 days.",
            "Bring residence card, bank details, and children's health insurance cards.",
        ],
    ),
    (
        "Waste_and_Recycling_Collection.pdf",
        [
            "Waste and Recycling Collection Guide",
            "Burnable garbage is collected every Tuesday and Friday.",
        ],
    ),
    (
        "Certificates_and_Fees_Current.pdf",
        [
            "Certificates and Fees — Current Fee Schedule",
            "Status: current",
            "Effective date: April 1, 2026",
            "Residence certificate counter fee: ¥350",
            "Convenience-store kiosk fee: ¥250",
        ],
    ),
    (
        "Certificates_and_Fees_Archived_2024.pdf",
        [
            "Certificates and Fees — Archived Fee Schedule 2024",
            "Status: archived",
            "Residence certificate counter fee: ¥300",
            "Kiosk fee: ¥200",
        ],
    ),
    (
        "Disaster_Preparedness_Guide.pdf",
        [
            "Disaster Preparedness Guide",
            "Keep at least 3 liters of water per person per day for an emergency.",
        ],
    ),
]


CASE_A = BenchmarkCase(
    test_id="COMPOUND-A",
    question="I just moved to the city. When must I register, and what do I bring?",
    expected_facts=(
        fact(
            "move_in_notification",
            "move-in notification",
            ("move in notification",),
            ("move-in notification",),
            ("moving in notification",),
        ),
        fact(
            "window_or_office",
            "resident-registration office/window",
            ("window 3",),
            ("citizen affairs",),
            ("citizen services",),
        ),
        fact("deadline_14", "14-day deadline", ("14 days",), ("within 14",)),
        fact(
            "id_docs",
            "residence card or passport",
            ("residence card",),
            ("passport", "landing"),
            ("passport",),
        ),
        fact(
            "moving_out_cert",
            "Moving-Out Certificate when applicable",
            ("moving out certificate",),
            ("moving-out certificate",),
        ),
        fact(
            "my_number_household",
            "My Number / notification cards for household members",
            ("my number", "household"),
            ("notification card", "household"),
            ("household members",),
            critical=False,
        ),
    ),
    required_sources=("Resident_Registration", "Moving_In", "resident"),
    forbidden_facts=(
        fact(
            "insurance_enroll",
            "insurance enrollment",
            ("national health insurance",),
            ("health insurance", "enroll"),
            ("employer", "insurance"),
        ),
        fact("window_6", "Insurance & Pension window", ("window 6",)),
        fact(
            "loss_cert",
            "certificate of loss of employer insurance",
            ("certificate of loss",),
            ("loss of eligibility",),
        ),
        fact(
            "nhi_mail",
            "NHI card mailing",
            ("nhi card",),
            ("mailed separately",),
        ),
    ),
    evaluation_notes="Single procedure: resident registration continuity across sub-questions.",
    compound_parts=2,
)

CASE_B = BenchmarkCase(
    test_id="COMPOUND-B",
    question="I left employer insurance. When must I enroll, and what do I bring?",
    expected_facts=(
        fact("enroll_deadline", "enrollment deadline", ("14 days",), ("within 14",)),
        fact(
            "loss_cert",
            "certificate of loss / loss of eligibility",
            ("certificate of loss",),
            ("loss of eligibility",),
        ),
        fact(
            "insurance_window",
            "insurance counter/window",
            ("insurance", "window"),
            ("window 6",),
            ("insurance counter",),
            critical=False,
        ),
    ),
    required_sources=("Health_Insurance", "Insurance", "Enrollment"),
    forbidden_facts=(
        fact(
            "move_in",
            "move-in notification / Window 3 registration",
            ("move in notification",),
            ("window 3",),
            ("moving-out certificate",),
        ),
    ),
    evaluation_notes="Both parts remain in health-insurance domain.",
    compound_parts=2,
)

CASE_C = BenchmarkCase(
    test_id="COMPOUND-C",
    question=(
        "I moved here and left employer insurance. When do I register my address, "
        "and when do I enroll in insurance?"
    ),
    expected_facts=(
        fact(
            "register_deadline",
            "address registration deadline",
            ("14 days", "register"),
            ("14 days", "address"),
            ("within 14",),
        ),
        fact(
            "enroll_deadline",
            "insurance enrollment deadline",
            ("14 days", "enroll"),
            ("losing employer",),
            ("employer coverage",),
            ("14 days", "insurance"),
        ),
    ),
    required_sources=("Resident_Registration", "Health_Insurance"),
    forbidden_facts=(),
    evaluation_notes="Legitimate split across two domains.",
    compound_parts=2,
)


def _upload_corpus(client: TestClient, session_id: str, company_id: str) -> None:
    for name, lines in CORPUS:
        response = client.post(
            f"/api/chat/sessions/{session_id}/attachments",
            data={"company_id": company_id},
            files={"file": (name, _pdf(lines), "application/pdf")},
        )
        assert response.status_code == 201, response.text


def _chat(client: TestClient, company_id: str, session_id: str, question: str) -> dict:
    response = client.post(
        "/api/chat",
        json={
            "company_id": company_id,
            "session_id": session_id,
            "include_company_docs": False,
            "question": question,
            "top_k": 6,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _log_case(case_id: str, body: dict, result) -> None:
    diag = body.get("diagnostics") or {}
    logger.info(
        "\n===== %s =====\n%s\nprocedure=%s\nresolved=%s\nbefore=%s\nafter=%s\nrejected=%s\n",
        case_id,
        result.report(),
        diag.get("procedure_context"),
        diag.get("resolved_sub_questions"),
        [
            {
                "doc": c.get("document_name"),
                "domain": c.get("service_domain"),
                "score": c.get("combined_score") or c.get("final_score"),
            }
            for block in diag.get("sub_question_results") or []
            for c in (block.get("candidates_before_rerank") or [])[:4]
        ],
        [
            {
                "doc": c.get("document_name"),
                "domain": c.get("service_domain"),
                "cont": c.get("continuity_score"),
                "penalty": c.get("cross_domain_penalty"),
                "score": c.get("combined_score") or c.get("final_score"),
            }
            for block in diag.get("sub_question_results") or []
            for c in (block.get("candidates_after_rerank") or block.get("top_candidates") or [])[:6]
        ],
        [
            r
            for block in diag.get("sub_question_results") or []
            for r in (block.get("rejected_cross_domain") or [])[:4]
        ],
    )


def test_single_procedure_does_not_allow_domain_switch():
    q = "I just moved to the city. When must I register, and what do I bring?"
    assert not question_explicitly_spans_domains(q)
    ctx = build_procedure_context(q)
    assert ctx.active_domain == RESIDENT_REGISTRATION
    assert ctx.locked is True
    subs = split_questions(q)
    resolved = resolve_compound_subquestions(
        subs,
        original_question=q,
        active_procedure=ctx.active_procedure,
        active_domain=ctx.active_domain,
    )
    assert all("moving-in" in item.lower() or "resident" in item.lower() or "registration" in item.lower() for item in resolved)
    assert assign_subquestion_domains(
        original_question=q, sub_questions=subs, context=ctx
    ) == [RESIDENT_REGISTRATION, RESIDENT_REGISTRATION]


def test_dual_topic_allows_domain_switch():
    q = CASE_C.question
    assert question_explicitly_spans_domains(q)
    ctx = build_procedure_context(q)
    assert ctx.allow_domain_switch is True
    subs = split_questions(q)
    domains = assign_subquestion_domains(
        original_question=q, sub_questions=subs, context=ctx
    )
    assert RESIDENT_REGISTRATION in domains
    assert HEALTH_INSURANCE in domains


def test_compound_a_move_in_continuity(client: TestClient):
    company_id = "compound-a"
    session = client.post("/api/chat/sessions").json()
    _upload_corpus(client, session["id"], company_id)
    body = _chat(client, company_id, session["id"], CASE_A.question)
    result = evaluate_answer(
        CASE_A,
        answer=body["answer"],
        sources=body.get("sources") or [],
        diagnostics=body.get("diagnostics"),
    )
    _log_case("A", body, result)

    diag = body.get("diagnostics") or {}
    assert diag.get("active_domain") == RESIDENT_REGISTRATION or diag.get(
        "service_domain"
    ) == RESIDENT_REGISTRATION
    sub2 = (diag.get("sub_question_results") or [None, None])[1]
    assert sub2 is not None
    # Health-insurance evidence must be rejected or outranked for sub-question 2.
    rejected = sub2.get("rejected_cross_domain") or []
    after = sub2.get("candidates_after_rerank") or sub2.get("top_candidates") or []
    top_docs = [c.get("document_name") or "" for c in after[:3]]
    assert any("Resident_Registration" in name for name in top_docs), after[:5]
    assert not any(
        "Health_Insurance" in (c.get("document_name") or "")
        and float(c.get("domain_match_score") or c.get("continuity_score") or 0) > 0
        for c in after[:2]
    )
    evidence_docs = {
        e.get("document_name") for e in (sub2.get("evidence_sent_to_llm") or [])
    }
    assert not any("Health_Insurance" in (name or "") for name in evidence_docs), evidence_docs
    if rejected:
        assert any(
            (r.get("service_domain") == HEALTH_INSURANCE)
            or ("Health_Insurance" in str(r.get("document_name") or ""))
            for r in rejected
        ), rejected

    assert_eval_rating(result, minimum="PARTIAL")
    assert not result.forbidden_facts_detected, result.report()
    assert not any(
        "deadline_14" in item or "move_in_notification" in item
        for item in result.critical_missing
    ), result.report()


def test_compound_b_insurance_continuity(client: TestClient):
    company_id = "compound-b"
    session = client.post("/api/chat/sessions").json()
    _upload_corpus(client, session["id"], company_id)
    body = _chat(client, company_id, session["id"], CASE_B.question)
    result = evaluate_answer(
        CASE_B,
        answer=body["answer"],
        sources=body.get("sources") or [],
        diagnostics=body.get("diagnostics"),
    )
    _log_case("B", body, result)
    diag = body.get("diagnostics") or {}
    assert (diag.get("active_domain") or diag.get("service_domain")) == HEALTH_INSURANCE
    for block in diag.get("sub_question_results") or []:
        assert block.get("service_domain") == HEALTH_INSURANCE or block.get(
            "active_domain"
        ) == HEALTH_INSURANCE
        evidence_docs = {
            e.get("document_name") for e in (block.get("evidence_sent_to_llm") or [])
        }
        assert not any(
            "Resident_Registration" in (name or "") for name in evidence_docs
        ), evidence_docs
    assert_eval_rating(result, minimum="PARTIAL")
    assert not result.forbidden_facts_detected, result.report()


def test_compound_c_legitimate_dual_domain(client: TestClient):
    company_id = "compound-c"
    session = client.post("/api/chat/sessions").json()
    _upload_corpus(client, session["id"], company_id)
    body = _chat(client, company_id, session["id"], CASE_C.question)
    result = evaluate_answer(
        CASE_C,
        answer=body["answer"],
        sources=body.get("sources") or [],
        diagnostics=body.get("diagnostics"),
    )
    _log_case("C", body, result)
    diag = body.get("diagnostics") or {}
    assert diag.get("procedure_context", {}).get("allow_domain_switch") is True
    domains = diag.get("sub_question_domains") or [
        block.get("service_domain")
        for block in diag.get("sub_question_results") or []
    ]
    assert RESIDENT_REGISTRATION in domains
    assert HEALTH_INSURANCE in domains
    # Evidence / sources should reflect both procedures across the compound answer.
    evidence_names = []
    for block in diag.get("sub_question_results") or []:
        for item in block.get("evidence_sent_to_llm") or []:
            evidence_names.append(item.get("document_name") or "")
    source_names = " ".join(
        list(evidence_names)
        + [s.get("document_name") or "" for s in (body.get("sources") or [])]
    )
    assert "Resident" in source_names or "Registration" in source_names, source_names
    assert "Insurance" in source_names or "Health" in source_names, source_names
    assert_eval_rating(result, minimum="PARTIAL")


def test_fee_behavior_still_prefers_current_schedule(client: TestClient):
    """Preserve residence-certificate fee current-vs-archived behavior."""
    company_id = "compound-fees"
    session = client.post("/api/chat/sessions").json()
    _upload_corpus(client, session["id"], company_id)
    body = _chat(
        client,
        company_id,
        session["id"],
        "How much does a residence certificate (juminhyo) cost?",
    )
    answer = body["answer"].lower()
    assert "350" in answer
    assert "could not find" not in answer
    # Must not present archived counter fee as the current answer.
    assert "300" not in answer or "350" in answer
    source_blob = " ".join(
        s.get("document_name") or "" for s in (body.get("sources") or [])
    ).lower()
    assert "current" in source_blob or "certificates_and_fees" in source_blob
    assert "archived" not in source_blob or "current" in source_blob
