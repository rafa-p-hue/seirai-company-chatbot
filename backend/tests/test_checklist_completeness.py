"""Regression: checklist completeness for procedural required-item questions."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.generation.answer_synthesis import (
    checklist_regeneration_instruction,
    extract_checklist_structure,
    missing_checklist_items,
    synthesize_checklist_answer,
)
from app.generation.evidence_validation import (
    checklist_answer_is_complete,
    extract_required_items,
    strip_checklist_deadline_padding,
)
from app.models.api import RetrievedChunk
from app.services.chat_service import _post_validate_answer


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


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "content": "",
        "document_name": "guide.pdf",
        "page_number": 1,
        "score": 1.0,
        "content_type": "list",
        "section_title": "Registration",
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


FULL_PROCEDURE = (
    "New residents must file a move-in notification and register "
    "within 14 days at Service Window 3. Bring your passport or "
    "national identity card, a lease agreement or utility bill, and "
    "a guardian consent form only if the applicant is under 18. "
    "Identification is required for all household members."
)


def test_one_required_item():
    evidence = [
        _chunk(
            content="Bring the following:\n• Passport",
            content_type="list",
        )
    ]
    items = extract_required_items(evidence)
    assert len(items) == 1
    assert "Passport" in items[0][1]
    answer = synthesize_checklist_answer(evidence)
    assert "Passport" in answer
    assert checklist_answer_is_complete(answer, evidence)


def test_multiple_required_items_from_same_passage():
    evidence = [
        _chunk(
            content=FULL_PROCEDURE,
            content_type="paragraph",
            section_title="Moving In",
        )
    ]
    items = [text for _, text in extract_required_items(evidence)]
    assert len(items) >= 3, items
    assert any("passport" in item.lower() for item in items)
    assert any("lease" in item.lower() or "utility" in item.lower() for item in items)
    assert any("guardian" in item.lower() or "consent" in item.lower() for item in items)
    assert any("household" in item.lower() for item in items)

    structure = extract_checklist_structure(evidence)
    assert len(structure) >= 3

    incomplete = "Bring your passport or national identity card. [1]"
    assert not checklist_answer_is_complete(incomplete, evidence)
    assert missing_checklist_items(incomplete, evidence)

    rewritten = _post_validate_answer(
        incomplete,
        "What do I bring?",
        "checklist",
        "checklist",
        evidence,
    )
    low = rewritten.lower()
    assert "passport" in low
    assert "lease" in low or "utility" in low
    assert "guardian" in low or "consent" in low
    assert "household" in low
    assert checklist_answer_is_complete(rewritten, evidence)


def test_or_alternative_preserved():
    evidence = [
        _chunk(
            content=(
                "Bring the following:\n"
                "• Passport or national identity card\n"
                "• Lease agreement or utility bill"
            )
        )
    ]
    structure = extract_checklist_structure(evidence)
    assert any(entry.get("alternative") for entry in structure)
    answer = synthesize_checklist_answer(evidence)
    assert " or " in answer.lower()
    assert "passport" in answer.lower()
    assert "national identity" in answer.lower() or "identity card" in answer.lower()
    assert "lease" in answer.lower()
    assert "utility" in answer.lower()


def test_conditional_item_preserved():
    evidence = [
        _chunk(
            content=(
                "Bring the following:\n"
                "• Passport or national identity card\n"
                "• Guardian consent form, only if the applicant is under 18"
            )
        )
    ]
    structure = extract_checklist_structure(evidence)
    assert any(
        (entry.get("condition") or "").lower().startswith("only if")
        or "only if" in (entry.get("item") or "").lower()
        for entry in structure
    )
    answer = synthesize_checklist_answer(evidence)
    assert "only if" in answer.lower()
    assert "under 18" in answer.lower()


def test_household_wide_requirement_preserved():
    evidence = [
        _chunk(
            content=(
                "Bring the following:\n"
                "• Residence card, or passport with landing permission\n"
                "• My Number cards or notification cards for all household members"
            )
        )
    ]
    structure = extract_checklist_structure(evidence)
    household = [
        entry
        for entry in structure
        if entry.get("type") == "household_document"
        or "household" in (entry.get("scope") or "").lower()
        or "household" in (entry.get("condition") or "").lower()
        or "household" in (entry.get("item") or "").lower()
    ]
    assert household, structure
    assert any(entry.get("scope") for entry in household) or any(
        "household" in (entry.get("item") or "").lower() for entry in household
    )
    answer = synthesize_checklist_answer(evidence)
    assert "my number" in answer.lower()
    assert "household" in answer.lower()
    assert checklist_answer_is_complete(answer, evidence)


def test_typed_structure_alt_conditional_and_household():
    """Evidence with A-or-B, conditional certificate, and household-wide item."""
    evidence = [
        _chunk(
            content=(
                "Moving In (Tennyu Todoke)\n"
                "Submit a move-in notification at Citizen Affairs Division "
                "Window 3 within 14 days of beginning to live at the new address.\n"
                "Bring the following:\n"
                "• Residence card, or passport with landing permission\n"
                "• Moving-Out Certificate when moving from another municipality\n"
                "• My Number cards or notification cards for all household members"
            ),
            section_title="Moving In (Tennyu Todoke)",
            content_type="list",
        )
    ]
    structure = extract_checklist_structure(evidence)
    assert len(structure) == 3, structure
    # Deadline/procedure prose must not become a checklist item.
    assert not any(
        "14 days" in (entry.get("item") or "").lower()
        or "window" in (entry.get("item") or "").lower()
        for entry in structure
    )

    by_type = {entry["type"]: entry for entry in structure}
    assert "identity_document" in by_type
    assert len(by_type["identity_document"]["alternatives"]) == 2
    assert by_type["identity_document"]["required"] is True

    assert "certificate" in by_type
    assert by_type["certificate"]["condition"]
    assert "when" in by_type["certificate"]["condition"].lower()

    assert "household_document" in by_type
    assert by_type["household_document"]["scope"]
    assert "household" in by_type["household_document"]["scope"].lower()

    incomplete = (
        "You must register within 14 days at Window 3. "
        "Bring your residence card or passport with landing permission."
    )
    assert not checklist_answer_is_complete(incomplete, evidence)
    rewritten = _post_validate_answer(
        incomplete,
        "What do I bring?",
        "checklist",
        "checklist",
        evidence,
    )
    low = rewritten.lower()
    assert "residence card" in low or "passport" in low
    assert "moving-out" in low or "moving out" in low
    assert "my number" in low
    assert "household" in low
    assert "14 days" not in low
    assert "window 3" not in low
    assert "identity_document" not in rewritten  # structure not exposed
    assert checklist_answer_is_complete(rewritten, evidence)


def test_deadline_padding_is_stripped_from_checklist_answers():
    padded = (
        "You must register within 14 days at Service Window 3. "
        "Bring your passport or national identity card and a lease agreement "
        "or utility bill. Identification is required for all household members."
    )
    stripped = strip_checklist_deadline_padding(padded)
    assert "14 days" not in stripped.lower()
    assert "window 3" not in stripped.lower()
    assert "passport" in stripped.lower()
    assert "lease" in stripped.lower() or "utility" in stripped.lower()


def test_incomplete_answer_gets_structured_fallback_not_deadline_repeat():
    evidence = [
        _chunk(
            content=FULL_PROCEDURE,
            content_type="paragraph",
            section_title="Moving In",
        )
    ]
    bad = (
        "Residents must register within 14 days at Service Window 3. "
        "Bring your passport or national identity card."
    )
    instruction = checklist_regeneration_instruction(evidence, bad)
    assert "lease" in instruction.lower() or "utility" in instruction.lower()
    assert "guardian" in instruction.lower() or "consent" in instruction.lower()

    rewritten = _post_validate_answer(
        bad,
        "What do I bring?",
        "checklist",
        "checklist",
        evidence,
    )
    # Must not pad missing items with another deadline sentence.
    deadline_hits = rewritten.lower().count("14 days")
    assert deadline_hits == 0
    assert "passport" in rewritten.lower()
    assert "lease" in rewritten.lower() or "utility" in rewritten.lower()
    assert "household" in rewritten.lower()


def test_compound_deadline_clause_unchanged_checklist_complete(client):
    from tests.test_checklist_and_tables import _text_pdf, _upload

    _upload(
        client,
        "checklist-complete-compound",
        "new-resident-guide.pdf",
        _text_pdf(
            [
                "New Resident Registration",
                "Residents must register within 14 days at Service Window 3.",
                "Required documents:",
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
            "company_id": "checklist-complete-compound",
            "question": (
                "I just moved to Lakeside. When must I register, "
                "and what do I bring?"
            ),
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["answer"]
    assert "14 days" in answer
    assert "Window 3" in answer
    assert "Passport or national identity card" in answer
    assert "Lease agreement or utility bill" in answer
    assert "only if the applicant is under 18" in answer
    assert "household members" in answer.lower()

    # Checklist section should not restate the deadline as a substitute for items.
    parts = answer.split("\n")
    checklist_parts = [
        part
        for part in parts
        if "bring" in part.lower()
        or "passport" in part.lower()
        or "lease" in part.lower()
        or "guardian" in part.lower()
        or "household" in part.lower()
    ]
    checklist_blob = "\n".join(checklist_parts).lower()
    # Allow deadline only in the dedicated when/register clause, not as the
    # sole content of the bring clause.
    bring_block = ""
    for part in parts:
        if "what do i bring" in part.lower():
            idx = parts.index(part)
            bring_block = "\n".join(parts[idx : idx + 6]).lower()
            break
    if bring_block:
        assert "passport" in bring_block
        assert "lease" in bring_block or "utility" in bring_block
        # Deadline may appear once in the compound answer overall, but the
        # bring block should still list documents rather than only restating it.
        assert "passport" in bring_block or "identity" in bring_block
