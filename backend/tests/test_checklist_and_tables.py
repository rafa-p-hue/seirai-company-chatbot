"""Regression coverage for exhaustive checklists and semantic PDF table rows."""

from __future__ import annotations

import re

import fitz
import pytest
from fastapi.testclient import TestClient

from app.generation.evidence_validation import (
    checklist_answer,
    checklist_answer_is_complete,
)
from app.ingestion.pdf_loader import extract_pdf_pages
from app.ingestion.universal_chunker import create_universal_chunks
from app.models.api import ExtractedPage, RetrievedChunk


@pytest.fixture()
def client(monkeypatch, tmp_path):
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


def _table_pdf(
    *,
    item: str,
    columns: list[str],
    values: list[str],
    effective_date: str = "April 1, 2026",
) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((48, 48), "Certificates and Fees", fontsize=14)
    page.insert_text((48, 66), f"Effective date: {effective_date}", fontsize=10)

    headers = ["Certificate", *columns]
    cells = [item, *values]
    widths = [220] + [110] * len(columns)
    xs = [48]
    for width in widths:
        xs.append(xs[-1] + width)
    ys = [86, 116, 148]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for index, header in enumerate(headers):
        page.insert_text((xs[index] + 5, 106), header, fontsize=8)
    for index, value in enumerate(cells):
        page.insert_text((xs[index] + 5, 137), value, fontsize=8)

    data = document.tobytes()
    document.close()
    return data


def _upload(client: TestClient, company_id: str, name: str, data: bytes):
    response = client.post(
        "/api/documents/upload",
        data={"company_id": company_id},
        files={"file": (name, data, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _text_pdf(lines: list[str]) -> bytes:
    document = fitz.open()
    page = document.new_page()
    y = 50
    for line in lines:
        # Wrap long lines so PyMuPDF insert_text does not clip at the page edge.
        remaining = line
        while remaining:
            chunk = remaining
            if len(chunk) > 78:
                split_at = chunk.rfind(" ", 0, 78)
                if split_at <= 0:
                    split_at = 78
                chunk, remaining = remaining[:split_at], remaining[split_at:].lstrip()
            else:
                remaining = ""
            page.insert_text((48, y), chunk, fontsize=10)
            y += 14
            if y > 780:
                page = document.new_page()
                y = 50
    data = document.tobytes()
    document.close()
    return data


def _fused_moving_in_pdf() -> bytes:
    """One complete Moving In paragraph (action, deadline, window, documents)."""
    return _text_pdf(
        [
            "City Resident Services Guide",
            "Overview",
            "This guide explains registration and certificates.",
            "Moving In",
            (
                "New residents must file a move-in notification and register "
                "within 14 days at Service Window 3. Bring your passport or "
                "national identity card, a lease agreement or utility bill, and "
                "a guardian consent form only if the applicant is under 18. "
                "Identification is required for all household members."
            ),
            "Office Hours",
            "Monday through Friday: 8:30 a.m. to 5:00 p.m.",
            "Certificates and Fees",
            "Residence certificates are available at the counter.",
        ]
    )


def test_checklist_chunk_and_answer_preserve_all_conditions():
    page = ExtractedPage(
        page_number=1,
        text=(
            "Registration Requirements\n"
            "Bring the following:\n"
            "• Passport or national identity card\n"
            "• Lease agreement or utility bill\n"
            "• Guardian consent form, only if the applicant is under 18\n"
            "• Identification for all household members"
        ),
    )
    chunks = create_universal_chunks(
        pages=[page],
        company_id="generic",
        document_id="requirements",
        document_name="guide.pdf",
    )
    lists = [chunk for chunk in chunks if chunk.content_type == "list"]
    assert len(lists) == 1
    assert "Passport or national identity card" in lists[0].content
    assert "only if the applicant is under 18" in lists[0].content
    assert "for all household members" in lists[0].content

    evidence = [
        RetrievedChunk(
            content=lists[0].content,
            document_name="guide.pdf",
            page_number=1,
            score=1.0,
            content_type="list",
        )
    ]
    answer = checklist_answer(evidence)
    assert checklist_answer_is_complete(answer, evidence)
    assert "Passport or national identity card" in answer
    assert "Lease agreement or utility bill" in answer
    assert "only if the applicant is under 18" in answer
    assert "for all household members" in answer
    assert "proof of address" not in answer.lower()


def test_checklist_chat_returns_exact_items_without_generic_substitutions(client):
    _upload(
        client,
        "requirements-chat",
        "registration.pdf",
        _text_pdf(
            [
                "Registration Requirements",
                "Bring the following:",
                "• Passport or national identity card",
                "• Lease agreement or utility bill",
                "• Guardian consent form, only if the applicant is under 18",
                "• Identification for all household members",
            ]
        ),
    )
    response = client.post(
        "/api/chat",
        json={
            "company_id": "requirements-chat",
            "question": "What do I bring for registration?",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["answer"]
    assert "Passport or national identity card" in answer
    assert "Lease agreement or utility bill" in answer
    assert "only if the applicant is under 18" in answer
    assert "for all household members" in answer
    assert "proof of address" not in answer.lower()


def test_compound_deadline_and_checklist_are_answered_independently(client):
    _upload(
        client,
        "resident-guide",
        "new-resident-guide.pdf",
        _text_pdf(
            [
                "New Resident Registration",
                "Residents must register within 14 days at Service Window 3.",
                "Required documents:",
                "• Passport or national identity card",
                "• Lease agreement or utility bill",
                "• Guardian consent form, only if the applicant is under 18",
            ]
        ),
    )
    response = client.post(
        "/api/chat",
        json={
            "company_id": "resident-guide",
            "question": (
                "I just moved to Lakeside. When must I register, "
                "and what do I bring?"
            ),
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "14 days" in body["answer"]
    assert "Window 3" in body["answer"]
    assert "Passport or national identity card" in body["answer"]
    assert "Lease agreement or utility bill" in body["answer"]
    assert "only if the applicant is under 18" in body["answer"]
    displayed = {source["number"] for source in body["sources"]}
    inline = {
        int(value)
        for value in __import__("re").findall(r"\[(\d+)\]", body["answer"])
    }
    assert inline == displayed


def test_checklist_follow_up_inherits_active_section_and_rejects_hours(client):
    _upload(
        client,
        "section-follow-up",
        "resident-services.pdf",
        _text_pdf(
            [
                "Moving In",
                "Residents must register within 14 days after moving in.",
                "Required documents:",
                "• Passport or national identity card",
                "• Lease agreement or utility bill",
                "• Guardian consent form, only if the applicant is under 18",
                "Office Hours",
                "• Monday through Friday: 8:30 a.m. to 5:00 p.m.",
                "• Saturday and Sunday: Closed",
                "Contact Information",
                "• Main office telephone line",
            ]
        ),
    )
    first_question = "When must I register after moving in?"
    first = client.post(
        "/api/chat",
        json={
            "company_id": "section-follow-up",
            "question": first_question,
            "history": [],
        },
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert "14 days" in first_body["answer"]
    assert "confirms" not in first_body["answer"].lower()

    second = client.post(
        "/api/chat",
        json={
            "company_id": "section-follow-up",
            "question": "What do I bring?",
            "history": [
                {"role": "user", "content": first_question},
                {"role": "assistant", "content": first_body["answer"]},
            ],
        },
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert "Passport or national identity card" in body["answer"]
    assert "Lease agreement or utility bill" in body["answer"]
    assert "only if the applicant is under 18" in body["answer"]
    assert "Monday" not in body["answer"]
    assert "Closed" not in body["answer"]
    assert "telephone" not in body["answer"].lower()
    assert body["diagnostics"]["context_question"] == first_question
    assert body["diagnostics"]["active_section"] == "Moving In"


def test_pdf_table_row_keeps_headers_values_and_effective_date(tmp_path):
    path = tmp_path / "fees.pdf"
    path.write_bytes(
        _table_pdf(
            item="Residence certificate (juminhyo)",
            columns=["Counter fee", "Kiosk fee"],
            values=["¥300", "¥200"],
        )
    )
    pages, metadata = extract_pdf_pages(path)
    assert metadata["table_row_count"] == 1
    row = pages[0].table_rows[0]
    assert row.heading == "Certificates and Fees"
    assert row.effective_date == "April 1, 2026"
    assert row.column_labels == ["Certificate", "Counter fee", "Kiosk fee"]
    assert row.cells == ["Residence certificate (juminhyo)", "¥300", "¥200"]
    assert "Counter fee: ¥300" in row.human_text
    assert "Kiosk fee: ¥200" in row.human_text

    chunks = create_universal_chunks(
        pages=pages,
        company_id="generic",
        document_id="fees",
        document_name="fees.pdf",
    )
    table_chunks = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].table_data
    assert table_chunks[0].content.count("¥") == 2


def test_price_query_returns_all_table_columns_with_consistent_citations(client):
    from tests.eval.semantic_eval import BenchmarkCase, assert_eval_rating, evaluate_answer, fact

    upload = _upload(
        client,
        "table-fees",
        "fees.pdf",
        _table_pdf(
            item="Residence certificate (juminhyo)",
            columns=["Counter fee", "Kiosk fee"],
            values=["¥300", "¥200"],
        ),
    )
    chunks = upload["chunks"]
    assert any(chunk["content_type"] == "table" for chunk in chunks)
    assert any(chunk["content_type"] == "structured_table_row" for chunk in chunks)

    response = client.post(
        "/api/chat",
        json={
            "company_id": "table-fees",
            "question": "How much does a residence certificate (juminhyo) cost?",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Fixture-local semantic ground truth (not a verbatim answer template).
    case = BenchmarkCase(
        test_id="FIXTURE-FEE-TABLE",
        question="How much does a residence certificate (juminhyo) cost?",
        expected_facts=(
            fact(
                "counter_300",
                "counter fee ¥300",
                ("300", "counter"),
                ("300 yen",),
                ("¥300",),
            ),
            fact(
                "kiosk_200",
                "kiosk fee ¥200",
                ("200", "kiosk"),
                ("200 yen",),
                ("¥200",),
            ),
            fact(
                "effective_2026",
                "effective April 1, 2026",
                ("april", "2026"),
                ("2026",),
                critical=False,
            ),
        ),
        required_sources=("fees.pdf", "fee"),
        forbidden_facts=(),
        evaluation_notes="Uses uploaded fixture amounts; paraphrases allowed.",
    )
    result = evaluate_answer(
        case,
        answer=body["answer"],
        sources=body.get("sources") or [],
        diagnostics=body.get("diagnostics"),
    )
    assert_eval_rating(result, minimum="PARTIAL")
    assert "could not find" not in body["answer"].lower()
    displayed = {source["number"] for source in body["sources"]}
    inline = {
        int(value)
        for value in __import__("re").findall(r"\[(\d+)\]", body["answer"])
    }
    assert inline == displayed

    diagnostics = body["diagnostics"]
    assert diagnostics["extracted_table_rows"]
    assert diagnostics["table_chunks_created"]
    candidate = next(
        item
        for item in diagnostics["initial_candidates"]
        if "juminhyo" in item["content"].lower()
    )
    assert candidate["exact_lexical_scores"]["residence certificate"] == 1.0
    assert candidate["exact_lexical_scores"]["juminhyo"] == 1.0
    assert candidate["numeric_currency_score"] > 0
    assert diagnostics["evidence_sent_to_llm"]


def _resident_services_pdf() -> bytes:
    return _text_pdf(
        [
            "City Resident Services Guide",
            "Overview",
            "This guide explains registration, certificates, and service windows for new residents.",
            "Moving In",
            "Residents must register within 14 days after moving in at Service Window 3.",
            "Required documents:",
            "• Passport or national identity card",
            "• Lease agreement or utility bill",
            "• Guardian consent form, only if the applicant is under 18",
            "• Identification for all household members",
            "Office Hours",
            "• Monday through Friday: 8:30 a.m. to 5:00 p.m.",
            "• Saturday and Sunday: Closed",
            "Contact Information",
            "• Main office telephone line",
            "Certificates and Fees",
            "Residence certificates are available at the counter and kiosk.",
        ]
    )


def test_document_summary_is_synthesized_not_office_hours_fragment(client):
    _upload(client, "summary-guide", "guide.pdf", _resident_services_pdf())
    response = client.post(
        "/api/chat",
        json={
            "company_id": "summary-guide",
            "question": "What is this document about?",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    answer = body["answer"]
    assert body["diagnostics"]["query_intent"] == "summary"
    assert "could not find" not in answer.lower()
    assert not answer.lstrip().startswith(("•", "-", "·"))
    assert "Monday" not in answer
    assert "Closed" not in answer
    assert re.search(r"(?i)\b(guide|covers|topics|registration|procedures?)\b", answer)
    assert len(answer.split()) >= 12


def test_register_deadline_retrieves_procedure_section(client):
    _upload(client, "deadline-guide", "guide.pdf", _resident_services_pdf())
    response = client.post(
        "/api/chat",
        json={
            "company_id": "deadline-guide",
            "question": "When must I register after moving?",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["answer"]
    assert "14 days" in answer
    assert "Window 3" in answer
    assert "City Resident Services Guide" not in answer
    assert "Monday" not in answer


def test_required_documents_checklist_is_complete(client):
    _upload(client, "docs-required", "guide.pdf", _resident_services_pdf())
    response = client.post(
        "/api/chat",
        json={
            "company_id": "docs-required",
            "question": "What documents are required?",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["answer"]
    assert "Passport or national identity card" in answer
    assert "Lease agreement or utility bill" in answer
    assert "only if the applicant is under 18" in answer
    assert "for all household members" in answer
    assert "Monday" not in answer
    assert "telephone" not in answer.lower()


def test_compound_register_and_bring_resolves_shared_context(client):
    _upload(client, "compound-guide", "guide.pdf", _resident_services_pdf())
    response = client.post(
        "/api/chat",
        json={
            "company_id": "compound-guide",
            "question": (
                "I just moved to Lakeside. When must I register, "
                "and what do I bring?"
            ),
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    answer = body["answer"]
    assert "14 days" in answer
    assert "Passport or national identity card" in answer
    assert "Lease agreement or utility bill" in answer
    assert "only if the applicant is under 18" in answer
    assert "City Resident Services Guide" not in answer
    assert "could not find" not in answer.lower()
    diagnostics = body["diagnostics"]
    assert diagnostics["query_intent"] == "multi"
    assert len(diagnostics["sub_questions"]) == 2
    assert diagnostics["resolved_sub_questions"]
    assert diagnostics["active_topic"]
    assert re.search(
        r"(?i)registration|moving", diagnostics["active_topic"] or ""
    )
    resolved = diagnostics["resolved_sub_questions"]
    assert any("deadline" in item.lower() for item in resolved)
    assert any(
        "documents" in item.lower() and "required" in item.lower() for item in resolved
    )
    assert not any(item.strip().lower() == "what do i bring?" for item in resolved)
    second = diagnostics["sub_question_results"][1]
    assert re.search(r"(?i)registration|moving", second["resolved_question"])
    assert "what do i bring?" not in second["resolved_question"].lower()
    assert second["top_candidates"]
    assert second["final_evidence"]


def test_fused_moving_in_paragraph_answers_deadline_and_checklist(client):
    upload = _upload(client, "fused-moving", "guide.pdf", _fused_moving_in_pdf())
    moving_chunks = [
        chunk
        for chunk in upload["chunks"]
        if (chunk.get("section_title") or "") == "Moving In"
        and "14 days" in chunk["content"]
    ]
    assert moving_chunks
    body_chunk = moving_chunks[0]["content"]
    assert "Moving In" in body_chunk or moving_chunks[0]["section_title"] == "Moving In"
    assert "move-in notification" in body_chunk.lower()
    assert "within 14 days" in body_chunk.lower()
    assert "Service Window 3" in body_chunk
    assert "Bring your passport" in body_chunk or "passport" in body_chunk.lower()

    deadline = client.post(
        "/api/chat",
        json={
            "company_id": "fused-moving",
            "question": "When must I register after moving in?",
            "history": [],
        },
    )
    assert deadline.status_code == 200, deadline.text
    assert "14 days" in deadline.json()["answer"]
    assert "Window 3" in deadline.json()["answer"]

    bring = client.post(
        "/api/chat",
        json={
            "company_id": "fused-moving",
            "question": "What documents do I need to bring?",
            "history": [],
        },
    )
    assert bring.status_code == 200, bring.text
    bring_answer = bring.json()["answer"]
    assert "passport" in bring_answer.lower()
    assert "lease agreement" in bring_answer.lower() or "utility bill" in bring_answer.lower()
    assert "under 18" in bring_answer
    assert "household members" in bring_answer.lower()
    assert "proof of address" not in bring_answer.lower()

    compound = client.post(
        "/api/chat",
        json={
            "company_id": "fused-moving",
            "question": (
                "I just moved to Hikari City. When must I register, "
                "and what do I bring?"
            ),
            "history": [],
        },
    )
    assert compound.status_code == 200, compound.text
    compound_body = compound.json()
    answer = compound_body["answer"]
    assert "14 days" in answer
    assert "Window 3" in answer
    assert "passport" in answer.lower()
    assert "under 18" in answer
    assert "household members" in answer.lower()
    assert "could not find" not in answer.lower()
    diagnostics = compound_body["diagnostics"]
    assert diagnostics["query_intent"] == "multi"
    assert diagnostics["moving_in_chunks"]
    assert diagnostics["resolved_sub_questions"]
    assert "moving-in" in (diagnostics.get("active_topic") or "").lower() or "register" in (
        diagnostics.get("active_topic") or ""
    ).lower()
    for sub in diagnostics["sub_question_results"]:
        assert sub["top_candidates"]
        assert sub["final_evidence"]
        assert len(sub["top_candidates"]) <= 20

    elliptical = client.post(
        "/api/chat",
        json={
            "company_id": "fused-moving",
            "question": "I moved recently. What is the deadline?",
            "history": [],
        },
    )
    assert elliptical.status_code == 200, elliptical.text
    assert "14 days" in elliptical.json()["answer"]

    notification = client.post(
        "/api/chat",
        json={
            "company_id": "fused-moving",
            "question": "What do I need for the move-in notification?",
            "history": [],
        },
    )
    assert notification.status_code == 200, notification.text
    note_answer = notification.json()["answer"]
    assert "passport" in note_answer.lower()
    assert "under 18" in note_answer


def test_procedure_chunk_keeps_deadline_with_requirements():
    pages = [
        ExtractedPage(
            page_number=1,
            text=(
                "Moving In\n"
                "Residents must register within 14 days after moving in at Service Window 3.\n"
                "Required documents:\n"
                "• Passport or national identity card\n"
                "• Lease agreement or utility bill\n"
                "• Guardian consent form, only if the applicant is under 18"
            ),
        )
    ]
    chunks = create_universal_chunks(
        pages=pages,
        company_id="generic",
        document_id="moving",
        document_name="guide.pdf",
    )
    assert any(chunk.section_title == "Moving In" for chunk in chunks)
    procedural = [
        chunk
        for chunk in chunks
        if "14 days" in chunk.content and "Passport" in chunk.content
    ]
    assert procedural, [chunk.content for chunk in chunks]
    assert "only if the applicant is under 18" in procedural[0].content
    # Short procedural sections keep the heading with the body unit.
    assert any(
        chunk.section_title == "Moving In"
        and "Moving In" in chunk.content
        and "14 days" in chunk.content
        for chunk in chunks
    )


def test_fused_prose_moving_in_chunk_is_one_searchable_unit():
    pages = [
        ExtractedPage(
            page_number=1,
            text=(
                "Moving In\n"
                "New residents must file a move-in notification and register within 14 days "
                "at Service Window 3. Bring your passport or national identity card, a lease "
                "agreement or utility bill, and a guardian consent form only if the applicant "
                "is under 18. Identification is required for all household members.\n"
                "Office Hours\n"
                "Monday through Friday: 8:30 a.m. to 5:00 p.m."
            ),
        )
    ]
    chunks = create_universal_chunks(
        pages=pages,
        company_id="generic",
        document_id="fused",
        document_name="guide.pdf",
    )
    moving = [
        chunk
        for chunk in chunks
        if chunk.section_title == "Moving In" and "14 days" in (chunk.content or "")
    ]
    assert len(moving) == 1
    content = moving[0].content
    assert "move-in notification" in content.lower()
    assert "within 14 days" in content.lower()
    assert "Service Window 3" in content
    assert "Bring your passport" in content or "passport or national identity card" in content
    assert "under 18" in content
    assert "household members" in content.lower()
    # Must not be split into deadline-only / bring-only fragments.
    assert not any(
        chunk.section_title == "Moving In"
        and "14 days" in (chunk.content or "")
        and "passport" not in (chunk.content or "").lower()
        for chunk in chunks
    )


def test_fee_answer_combines_matching_values_from_multiple_files(client):
    _upload(
        client,
        "multi-fees",
        "counter.pdf",
        _table_pdf(
            item="Residency record (resident-register copy)",
            columns=["Counter fee"],
            values=["$6"],
        ),
    )
    _upload(
        client,
        "multi-fees",
        "online.pdf",
        _table_pdf(
            item="Residency record (resident-register copy)",
            columns=["Online fee"],
            values=["$4"],
        ),
    )
    response = client.post(
        "/api/chat",
        json={
            "company_id": "multi-fees",
            "question": (
                "How much does a residency record "
                "(resident-register copy) cost?"
            ),
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "$6" in body["answer"]
    assert "$4" in body["answer"]
    assert {source["document_name"] for source in body["sources"]} == {
        "counter.pdf",
        "online.pdf",
    }
