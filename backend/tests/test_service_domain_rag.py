"""Corpus-level multi-file service-domain retrieval regressions.

WARNING: This synthetic PDF CORPUS is NOT the live Hikari upload set.
Live current residence-certificate fees are ¥350/¥250 (see
tests/fixtures/OUTDATED_FIXTURE_WARNING.md). Do not treat ¥300/¥200 here
as live current-fee ground truth.

All seven municipal documents are attached to the same chat session.
Facts below exist only in test fixtures — production code stays document-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import fitz
import pytest
from fastapi.testclient import TestClient

from app.ingestion.service_domain import (
    CERTIFICATE_FEES,
    CHILDCARE_SUPPORT,
    DISASTER_PREPAREDNESS,
    HEALTH_INSURANCE,
    RESIDENT_REGISTRATION,
    WASTE_RECYCLING,
    detect_service_domain,
    extract_query_service_domain,
)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    database_path = tmp_path / "service_domain.db"
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
            chunk = remaining
            if len(chunk) > 92:
                split_at = chunk.rfind(" ", 0, 92)
                if split_at <= 0:
                    split_at = 92
                chunk, remaining = remaining[:split_at], remaining[split_at:].lstrip()
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


CORPUS: List[Dict[str, Any]] = [
    {
        "filename": "Resident_Registration_Moving_In.pdf",
        "domain": RESIDENT_REGISTRATION,
        "lines": [
            "Resident Registration and Moving-In Guide",
            "Status: current",
            "Effective date: April 1, 2026",
            "Moving In Notification",
            "When you move into the city, you must complete resident registration.",
            "You must register within 14 days of moving in.",
            "Bring your residence card, passport or national ID, and previous address certificate.",
            "Submit the move-in notification at the Citizen Services Window.",
            "Office hours Monday through Friday.",
        ],
    },
    {
        "filename": "National_Health_Insurance_Enrollment.pdf",
        "domain": HEALTH_INSURANCE,
        "lines": [
            "National Health Insurance Enrollment Guide",
            "Status: current",
            "Effective date: April 1, 2026",
            "Insurance Enrollment After Leaving Employer Insurance",
            "If you leave your employer's insurance, you must enroll in national health insurance.",
            "You must enroll within 14 days of losing employer coverage.",
            "Bring your residence card, certificate of loss of eligibility, and personal seal if you have one.",
            "Submit enrollment at the Insurance Counter.",
            "Insurance premium notices are mailed separately.",
        ],
    },
    {
        "filename": "Child_Allowance_Application.pdf",
        "domain": CHILDCARE_SUPPORT,
        "lines": [
            "Child Allowance and Childcare Support Guide",
            "Status: current",
            "Effective date: April 1, 2026",
            "Child Allowance Application After Moving",
            "To receive child allowance after moving here, you must apply within 15 days of the move.",
            "Bring your residence card, bank account details, and children's health insurance cards.",
            "Submit the application at the Child Welfare Desk.",
            "Parenting support counseling is available by appointment.",
        ],
    },
    {
        "filename": "Waste_and_Recycling_Collection.pdf",
        "domain": WASTE_RECYCLING,
        "lines": [
            "Waste and Recycling Collection Guide",
            "Status: current",
            "Effective date: April 1, 2026",
            "Household Waste Collection Days",
            "Burnable garbage is collected every Tuesday and Friday.",
            "Recyclables are collected every Wednesday.",
            "Put bags at the designated collection point by 8:00 a.m.",
            "Contact the Waste Management Office for missed pickups.",
        ],
    },
    {
        "filename": "Certificates_and_Fees_Current.pdf",
        "domain": CERTIFICATE_FEES,
        "lines": [
            "Certificates and Fees — Current Fee Schedule",
            "Status: current",
            "Effective date: April 1, 2026",
            "Certificate Issuance Fees",
            "Residence certificate counter fee: 300 yen.",
            "Residence certificate kiosk fee: 200 yen.",
            "Family register abstract: 450 yen.",
            "This schedule replaces earlier fee tables.",
        ],
    },
    {
        "filename": "Certificates_and_Fees_Archived_2024.pdf",
        "domain": CERTIFICATE_FEES,
        "lines": [
            "Certificates and Fees — Archived Fee Schedule 2024",
            "Status: archived",
            "Effective date: April 1, 2024",
            "Document status: archived / superseded",
            "Certificate Issuance Fees (2024)",
            "Residence certificate counter fee: 400 yen.",
            "Residence certificate kiosk fee: 250 yen.",
            "Family register abstract: 450 yen.",
            "This archived schedule is no longer valid for current requests.",
        ],
    },
    {
        "filename": "Disaster_Preparedness_Guide.pdf",
        "domain": DISASTER_PREPAREDNESS,
        "lines": [
            "Disaster Preparedness Guide",
            "Status: current",
            "Effective date: April 1, 2026",
            "Emergency Stockpile Recommendations",
            "Keep at least 3 liters of water per person per day for an emergency.",
            "Store a three-day supply of food and a flashlight.",
            "Know your nearest evacuation site.",
            "Contact the Disaster Management Office for neighborhood drills.",
        ],
    },
]


CASES = [
    {
        "id": "A",
        "question": "I just moved to the city. When must I register, and what do I bring?",
        "domain": RESIDENT_REGISTRATION,
        "must_include": ["14 days", "residence card"],
        "must_exclude": ["employer", "child allowance", "burnable", "3 liters"],
        "prefer_file": "Resident_Registration_Moving_In.pdf",
    },
    {
        "id": "B",
        "question": "I left my employer's insurance. When must I enroll, and what do I bring?",
        "domain": HEALTH_INSURANCE,
        "must_include": ["14 days", "certificate of loss"],
        "must_exclude": ["move-in notification", "child allowance", "burnable"],
        "prefer_file": "National_Health_Insurance_Enrollment.pdf",
    },
    {
        "id": "C",
        "question": "When must I apply after moving here to receive child allowance?",
        "domain": CHILDCARE_SUPPORT,
        "must_include": ["15 days"],
        "must_exclude": ["employer", "burnable", "3 liters"],
        "prefer_file": "Child_Allowance_Application.pdf",
    },
    {
        "id": "D",
        "question": "When is burnable garbage collected?",
        "domain": WASTE_RECYCLING,
        "must_include": ["Tuesday", "Friday"],
        "must_exclude": ["residence certificate", "3 liters"],
        "prefer_file": "Waste_and_Recycling_Collection.pdf",
    },
    {
        "id": "E",
        "question": "How much does a residence certificate cost now?",
        "domain": CERTIFICATE_FEES,
        "must_include": ["300"],
        "must_exclude": ["400 yen"],
        "prefer_file": "Certificates_and_Fees_Current.pdf",
        "avoid_file": "Certificates_and_Fees_Archived_2024.pdf",
    },
    {
        "id": "F",
        "question": "What was the residence-certificate fee in 2024?",
        "domain": CERTIFICATE_FEES,
        "must_include": ["400"],
        "must_exclude": [],
        "prefer_file": "Certificates_and_Fees_Archived_2024.pdf",
    },
    {
        "id": "G",
        "question": "How much water should I keep for an emergency?",
        "domain": DISASTER_PREPAREDNESS,
        "must_include": ["3 liters"],
        "must_exclude": ["residence certificate", "burnable"],
        "prefer_file": "Disaster_Preparedness_Guide.pdf",
    },
]


def _create_session(client: TestClient) -> dict:
    response = client.post("/api/chat/sessions")
    assert response.status_code == 201, response.text
    return response.json()


def _upload_all(client: TestClient, session_id: str, company_id: str) -> List[dict]:
    uploaded = []
    for item in CORPUS:
        response = client.post(
            f"/api/chat/sessions/{session_id}/attachments",
            data={"company_id": company_id},
            files={
                "file": (
                    item["filename"],
                    _pdf(item["lines"]),
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        uploaded.append(body)
        assert body["document"]["document_scope"] == "chat"
        assert body["document"]["session_id"] == session_id
        assert body["document"]["service_domain"] == item["domain"], body["document"]
    return uploaded


def _top_candidates(payload: dict) -> List[dict]:
    diag = payload.get("diagnostics") or {}
    if diag.get("top_candidates_across_files"):
        return list(diag["top_candidates_across_files"])
    if diag.get("sub_question_results"):
        rows: List[dict] = []
        for block in diag["sub_question_results"]:
            rows.extend(block.get("top_candidates") or [])
        return rows
    return list(diag.get("initial_candidates") or diag.get("reranked_candidates") or [])


def _log_candidates(caplog, case_id: str, question: str, candidates: Sequence[dict]) -> None:
    with caplog.at_level(logging.INFO):
        logging.getLogger("test_service_domain_rag").info(
            "CASE %s Q=%r top_candidates=%s",
            case_id,
            question,
            [
                {
                    "filename": c.get("filename") or c.get("document_name"),
                    "service_domain": c.get("service_domain"),
                    "heading": c.get("heading") or c.get("section_title"),
                    "domain_match_score": c.get("domain_match_score"),
                    "lexical_score": c.get("lexical_score"),
                    "vector_score": c.get("vector_score")
                    or c.get("vector_similarity_score"),
                    "archive_penalty": c.get("archive_penalty"),
                    "final_score": c.get("final_score") or c.get("combined_score"),
                }
                for c in candidates[:12]
            ],
        )


def test_query_domain_detection_from_full_question():
    assert (
        extract_query_service_domain(CASES[0]["question"]) == RESIDENT_REGISTRATION
    )
    assert extract_query_service_domain(CASES[1]["question"]) == HEALTH_INSURANCE
    assert extract_query_service_domain(CASES[2]["question"]) == CHILDCARE_SUPPORT
    assert extract_query_service_domain(CASES[3]["question"]) == WASTE_RECYCLING
    assert extract_query_service_domain(CASES[4]["question"]) == CERTIFICATE_FEES
    assert extract_query_service_domain(CASES[6]["question"]) == DISASTER_PREPAREDNESS


def test_ingest_learns_domains_from_titles_and_text():
    for item in CORPUS:
        detected = detect_service_domain(
            filename=item["filename"],
            headings=[item["lines"][0]],
            text="\n".join(item["lines"]),
        )
        assert detected == item["domain"], item["filename"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_seven_file_session_domain_retrieval(client: TestClient, case: dict, caplog):
    company_id = "hikari-city-corpus"
    session = _create_session(client)
    _upload_all(client, session["id"], company_id)

    listed = client.get(f"/api/chat/sessions/{session['id']}/attachments")
    assert listed.status_code == 200
    assert len(listed.json()) == 7

    response = client.post(
        "/api/chat",
        json={
            "company_id": company_id,
            "session_id": session["id"],
            "include_company_docs": False,
            "question": case["question"],
            "top_k": 6,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    answer = payload["answer"]
    diagnostics = payload.get("diagnostics") or {}
    candidates = _top_candidates(payload)
    _log_candidates(caplog, case["id"], case["question"], candidates)

    assert diagnostics.get("service_domain") == case["domain"] or any(
        block.get("service_domain") == case["domain"]
        for block in diagnostics.get("sub_question_results") or []
    )

    assert candidates, "expected ranked candidates across attached files"
    top = candidates[0]
    top_name = top.get("filename") or top.get("document_name") or ""
    assert case["prefer_file"] in top_name or any(
        case["prefer_file"] in (c.get("filename") or c.get("document_name") or "")
        for c in candidates[:3]
    ), candidates[:5]

    if case.get("avoid_file"):
        # Current fee questions must not lead with the archived schedule.
        assert case["avoid_file"] not in top_name

    # Domain match should outrank conflicting municipal procedures in the top set.
    matching = [
        c
        for c in candidates[:8]
        if (c.get("service_domain") or "") == case["domain"]
        or case["prefer_file"] in (c.get("filename") or c.get("document_name") or "")
    ]
    assert matching, candidates[:8]
    for needle in case["must_include"]:
        assert needle.lower() in answer.lower(), (needle, answer)
    for banned in case["must_exclude"]:
        assert banned.lower() not in answer.lower(), (banned, answer)


def test_all_session_attachments_remain_searchable(client: TestClient):
    company_id = "hikari-searchable"
    session = _create_session(client)
    _upload_all(client, session["id"], company_id)

    seen_files = set()
    for case in CASES:
        response = client.post(
            "/api/chat",
            json={
                "company_id": company_id,
                "session_id": session["id"],
                "include_company_docs": False,
                "question": case["question"],
                "top_k": 6,
            },
        )
        assert response.status_code == 200
        for candidate in _top_candidates(response.json())[:5]:
            name = candidate.get("filename") or candidate.get("document_name")
            if name:
                seen_files.add(name)
        prefer = case["prefer_file"]
        assert any(
            prefer in (c.get("filename") or c.get("document_name") or "")
            for c in _top_candidates(response.json())[:5]
        ), case["id"]

    # Across the corpus questions, multiple files appear — not a single-file lock.
    assert len(seen_files) >= 5, seen_files
