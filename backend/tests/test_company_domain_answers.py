"""Company-domain grounding regressions (TalentBridge-style corpus).

Production code stays document-agnostic. Facts below exist only in fixtures.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import fitz
import pytest
from fastapi.testclient import TestClient

from app.generation.evidence_presentation import format_structured_content_as_prose
from app.generation.prompts import FALLBACK_ANSWER
from app.models.api import RetrievedChunk
from app.retrieval.answer_grounding import (
    detect_source_intent,
    filter_evidence_for_answer_grounding,
    format_office_not_found_answer,
    is_candidate_fee_evidence,
    is_employer_fee_block,
    is_fee_process_noise_only,
    is_office_location_evidence,
)
from app.retrieval.entity_validation import (
    extract_requested_service_phrases,
    filter_evidence_for_requested_entity,
    format_not_found_answer,
    infer_not_found_contact_name,
)
from app.retrieval.query_understanding import classify_query


@pytest.fixture()
def client(monkeypatch, tmp_path):
    database_path = tmp_path / "talentbridge.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("NOT_FOUND_CONTACT_NAME", "")
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


def _create_session(client: TestClient) -> dict:
    response = client.post("/api/chat/sessions")
    assert response.status_code == 201, response.text
    return response.json()


def _upload(client: TestClient, session_id: str, filename: str, data: bytes) -> None:
    response = client.post(
        f"/api/chat/sessions/{session_id}/attachments",
        data={"company_id": "seirai"},
        files={"file": (filename, data, "application/octet-stream")},
    )
    assert response.status_code == 201, response.text


def _ask(
    client: TestClient, session_id: str, question: str
) -> Dict[str, Any]:
    response = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"company_id": "seirai", "content": question, "top_k": 8},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return {
        "answer": body["assistant"]["content"],
        "diagnostics": body.get("diagnostics") or {},
        "raw": body,
    }


def _talentbridge_corpus() -> Dict[str, bytes]:
    fees_pdf = _pdf(
        [
            "TalentBridge Partners — Services & Fee Schedule",
            "Effective 1 April 2026 (current version).",
            "1. Recruitment Services & Fees",
            "Fees for permanent placements are calculated as a percentage of the "
            "candidate's first-year base salary.",
            "Service: Permanent placement (contingent)",
            "Fee (from 1 Apr 2026): 22% of first-year base salary",
            "Minimum fee: SGD 9,000",
            "Service: Executive search (retained)",
            "Fee (from 1 Apr 2026): 30% of first-year total compensation",
            "Minimum fee: SGD 30,000",
            "2. Replacement Guarantee",
            "Permanent placements carry a 90-day replacement guarantee.",
            "3. Payment Terms",
            "For retained searches, instalments are billed at engagement, "
            "shortlist delivery, and placement.",
        ]
    )
    placement_md = (
        "# TalentBridge Partners — Our Placement Process\n\n"
        "Deliver a qualified shortlist within 10 business days of engagement.\n"
        "Our 2025 offer-acceptance rate was 94%.\n"
        "## Replacement Guarantee\n"
        "All permanent placements carry a 90-day replacement guarantee.\n"
    ).encode("utf-8")
    candidate_faq = (
        "<!DOCTYPE html><html><body>"
        "<h1>Candidate FAQ — TalentBridge Partners</h1>"
        "<h2>Do I have to pay TalentBridge anything?</h2>"
        "<p>No. Our services are completely free for candidates. Our fees are "
        "paid entirely by hiring companies. This applies to CV review, interview "
        "coaching, and salary negotiation support.</p>"
        "<h2>Which locations do you cover?</h2>"
        "<p>We place candidates through our offices in Singapore (headquarters), "
        "London, Sydney, and Ho Chi Minh City.</p>"
        "</body></html>"
    ).encode("utf-8")
    jobs_csv = (
        "job_id,title,practice,location,employment_type,salary_range,remote_option,posted_date\n"
        "TB-2026-041,Senior Backend Engineer (Go),Technology,Singapore,Permanent,"
        '"SGD 120,000 - 150,000",Hybrid,2026-06-02\n'
        "TB-2026-042,DevOps Engineer,Technology,Singapore,Permanent,"
        '"SGD 95,000 - 125,000",Hybrid,2026-06-09\n'
        "TB-2026-046,Finance Manager - FP&A,Finance,Singapore,Permanent,"
        '"SGD 110,000 - 135,000",Hybrid,2026-06-05\n'
    ).encode("utf-8")
    overview = _pdf(
        [
            "TalentBridge Partners company overview",
            "TalentBridge Partners was founded in Singapore in 2011 by Amelia Chong "
            "and David Reyes.",
            "Headquarters: 12 Raffles Place, Singapore",
            "Other offices: London, Sydney, Ho Chi Minh City",
            "Contact: hello@talentbridge.example.com | +65 6555 0114",
            "Shortlist within 10 business days. 94% offer-acceptance rate.",
            "90-day replacement guarantee on all permanent placements.",
        ]
    )
    return {
        "02_services_and_fees_2026.pdf": fees_pdf,
        "06_placement_process.md": placement_md,
        "04_candidate_faq.html": candidate_faq,
        "05_open_positions.csv": jobs_csv,
        "01_company_overview.pdf": overview,
    }


def test_a_unsupported_company_question_uses_org_neutral_fallback():
    text = format_not_found_answer(None)
    assert "the city" not in text.lower()
    assert "hikari" not in text.lower()
    assert "municipal" not in text.lower()
    assert "organization directly" in text.lower()
    assert "the city" not in FALLBACK_ANSWER.lower()

    named = format_not_found_answer("TalentBridge")
    assert "TalentBridge directly" in named
    assert "the city" not in named.lower()

    inferred = infer_not_found_contact_name(
        question="Who is the CEO of TalentBridge now?",
        configured_name="",
    )
    assert inferred == "TalentBridge"

    office = format_office_not_found_answer("Tokyo", contact_name="TalentBridge")
    assert "Tokyo office" in office
    assert "TalentBridge directly" in office
    assert "the city" not in office.lower()


def test_permanent_placement_rejects_process_noise():
    question = "What fee does TalentBridge charge for a permanent placement?"
    assert detect_source_intent(question) == "employer_fee"

    noise = RetrievedChunk(
        content=(
            "All permanent placements carry a 90-day replacement guarantee. "
            "Shortlist within 10 business days. Offer-acceptance rate was 94%."
        ),
        document_name="06_placement_process.md",
        score=0.95,
        chunk_id="noise",
        document_id="n",
    )
    fee = RetrievedChunk(
        content=(
            "Service: Permanent placement (contingent) "
            "Fee (from 1 Apr 2026): 22% of first-year base salary "
            "Minimum fee: SGD 9,000"
        ),
        document_name="02_services_and_fees_2026.pdf",
        score=0.8,
        chunk_id="fee",
        document_id="f",
        document_status="current",
    )
    assert is_fee_process_noise_only(noise.content)
    assert is_employer_fee_block(fee.content, service_phrases=["permanent placement"])

    kept, diag = filter_evidence_for_answer_grounding(
        [noise, fee], question, fact_type="price"
    )
    assert any("22%" in (c.content or "") for c in kept), diag
    assert all("replacement guarantee" not in (c.content or "").lower() for c in kept)


def test_job_listing_prose_keeps_salary_range():
    row = (
        "job_id: TB-2026-042; title: DevOps Engineer; practice: Technology; "
        "location: Singapore; employment_type: Permanent; "
        "salary_range: SGD 95,000 - 125,000; "
        "remote_option: Hybrid (2 days office); posted_date: 2026-06-09"
    )
    prose = format_structured_content_as_prose(row)
    assert "DevOps Engineer" in prose
    assert "Singapore" in prose
    assert "95,000" in prose and "125,000" in prose


def test_salary_guide_intent_and_open_position_rejection():
    q = "According to the 2026 salary guide, what does a senior DevOps Engineer earn?"
    assert detect_source_intent(q) == "salary_guide"
    assert classify_query(q) == "price"

    open_row = RetrievedChunk(
        content=(
            "job_id: TB-2026-041; title: Senior Backend Engineer (Go); "
            "location: Singapore; salary_range: SGD 120,000 - 150,000"
        ),
        document_name="05_open_positions.csv",
        score=0.9,
        chunk_id="job",
        document_id="j",
        content_type="structured_table_row",
        row_number=1,
    )
    junior = RetrievedChunk(
        content=(
            "job_id: TB-2026-042; title: DevOps Engineer; location: Singapore; "
            "salary_range: SGD 95,000 - 125,000"
        ),
        document_name="05_open_positions.csv",
        score=0.85,
        chunk_id="job2",
        document_id="j",
        content_type="structured_table_row",
        row_number=2,
    )
    kept, diag = filter_evidence_for_answer_grounding(
        [open_row, junior], q, fact_type="price"
    )
    assert kept == []
    assert diag.get("rejection_reason") in {
        "open_position_not_salary_guide",
        "salary_guide_required_but_absent",
        "role_or_seniority_mismatch",
        "not_salary_guide_evidence",
    }


def test_tokyo_office_requires_explicit_location():
    q = "Does TalentBridge have an office in Tokyo?"
    assert detect_source_intent(q) == "office_location"
    assert classify_query(q) == "location"

    contact = RetrievedChunk(
        content="Contact: hello@talentbridge.example.com | +65 6555 0114",
        document_name="01_company_overview.pdf",
        score=0.9,
        chunk_id="c",
        document_id="o",
    )
    offices = RetrievedChunk(
        content="Other offices: London, Sydney, Ho Chi Minh City",
        document_name="01_company_overview.pdf",
        score=0.8,
        chunk_id="o2",
        document_id="o",
    )
    assert not is_office_location_evidence(contact.content, "Tokyo")
    assert not is_office_location_evidence(offices.content, "Tokyo")
    assert is_office_location_evidence(
        "We have an office in Tokyo at 1 Chiyoda.", "Tokyo"
    )

    kept, diag = filter_evidence_for_answer_grounding(
        [contact, offices], q, fact_type="location"
    )
    assert kept == []
    assert diag.get("rejection_reason")


def test_candidate_fee_evidence_detection():
    q = "Do I have to pay TalentBridge anything as a job seeker?"
    assert detect_source_intent(q) == "candidate_fee"
    assert classify_query(q) == "policy"
    text = (
        "No. Our services are completely free for candidates. Our fees are "
        "paid entirely by hiring companies."
    )
    assert is_candidate_fee_evidence(text)


def test_a_b_c_d_live_grounding(client: TestClient):
    session = _create_session(client)
    for name, data in _talentbridge_corpus().items():
        _upload(client, session["id"], name, data)

    def _diag_summary(result: Dict[str, Any]) -> Dict[str, Any]:
        d = result.get("diagnostics") or {}
        ag = d.get("answer_grounding") or {}
        return {
            "answer": result.get("answer"),
            "top": [
                {
                    "document_name": c.get("document_name"),
                    "preview": (c.get("content") or c.get("preview") or "")[:100],
                }
                for c in (d.get("top_candidates_across_files") or [])[:5]
            ],
            "selected": [
                {
                    "document_name": c.get("document_name"),
                    "source_type": c.get("content_type") or c.get("record_type"),
                    "preview": (c.get("content") or "")[:100],
                }
                for c in (d.get("evidence_after_fact_filter") or [])[:5]
            ],
            "source_intent": ag.get("source_intent"),
            "entity_match_ok": ag.get("entity_match_ok"),
            "answer_type_ok": ag.get("answer_type_ok"),
            "rejection": ag.get("rejection_reason") or d.get("rejection_reason"),
            "rejected": [r.get("rejection_reason") for r in (ag.get("rejected") or [])][
                :6
            ],
            "rescue": (d.get("validation_rescue") or {}).get("reasons"),
        }

    # A. Permanent placement fee
    fee = _ask(
        client,
        session["id"],
        "What fee does TalentBridge charge for a permanent placement?",
    )
    fee_answer = fee["answer"]
    assert "could not find" not in fee_answer.lower(), _diag_summary(fee)
    assert "22%" in fee_answer or "22 percent" in fee_answer.lower(), _diag_summary(fee)
    assert "9,000" in fee_answer or "9000" in fee_answer.replace(",", ""), _diag_summary(
        fee
    )
    assert "first-year" in fee_answer.lower() or "base salary" in fee_answer.lower()
    assert "shortlist" not in fee_answer.lower()
    assert "acceptance" not in fee_answer.lower()
    assert "guarantee" not in fee_answer.lower()
    assert "30%" not in fee_answer

    # B. Job seeker fee — identical query five times
    seeker_q = "Do I have to pay TalentBridge anything as a job seeker?"
    for i in range(5):
        result = _ask(client, session["id"], seeker_q)
        answer = result["answer"].lower()
        assert "could not find" not in answer, (i, _diag_summary(result))
        assert (
            "free" in answer
            or "no fee" in answer
            or "no charge" in answer
            or answer.strip().startswith("no")
        ), (i, _diag_summary(result))

    # C. Senior DevOps salary guide — no guide doc uploaded → grounded not-found
    guide = _ask(
        client,
        session["id"],
        "According to the 2026 salary guide, what does a senior DevOps Engineer earn?",
    )
    guide_answer = guide["answer"].lower()
    assert "could not find" in guide_answer, _diag_summary(guide)
    assert "backend" not in guide_answer
    assert "devops directly" not in guide_answer
    assert "120,000" not in guide["answer"]
    assert "95,000" not in guide["answer"]
    assert "the city" not in guide_answer
    assert "organization directly" in guide_answer or "talentbridge directly" in guide_answer

    # D. Tokyo office — not in documents; must not return contact-only
    tokyo = _ask(
        client,
        session["id"],
        "Does TalentBridge have an office in Tokyo?",
    )
    tokyo_answer = tokyo["answer"]
    lower = tokyo_answer.lower()
    assert "tokyo office" in lower or "could not find confirmation" in lower, _diag_summary(
        tokyo
    )
    assert "hello@talentbridge" not in lower
    assert "+65" not in tokyo_answer
    assert "the city" not in lower
    assert "talentbridge directly" in lower or "organization directly" in lower


def test_open_role_salary_still_works(client: TestClient):
    """Open-role salary questions may still use job listings."""
    session = _create_session(client)
    for name, data in _talentbridge_corpus().items():
        _upload(client, session["id"], name, data)

    result = _ask(
        client,
        session["id"],
        "What is the salary range for the DevOps Engineer job in Singapore?",
    )
    answer = result["answer"]
    assert "could not find" not in answer.lower()
    assert "95,000" in answer and "125,000" in answer
    assert "DevOps" in answer
    assert "120,000" not in answer
