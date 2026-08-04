"""Regression tests for post-retrieval validation / compose failures.

Covers:
A) PPTX pet-friendly shelter evidence must survive and answer
B) Unsupported parking-permit fee must not fall back to Koseki
C) Move-in compound checklist must keep deadline + complete requirements
"""

from __future__ import annotations

import pytest

from app.generation.answer_composer import compose_answer
from app.generation.evidence_presentation import expand_sentence_boundaries
from app.generation.evidence_validation import (
    extract_required_items,
    relevant_price_evidence,
)
from app.models.api import RetrievedChunk
from app.retrieval.entity_validation import (
    answer_matches_requested_entity,
    filter_evidence_for_requested_entity,
)
from app.retrieval.query_understanding import understand_query
from app.services.chat_service import _penalize_conflicting_topic_evidence


def _chunk(**kwargs) -> RetrievedChunk:
    base = {
        "chunk_id": kwargs.get("chunk_id", "c1"),
        "company_id": "seirai",
        "document_id": kwargs.get("document_id", "doc"),
        "document_name": kwargs.get("document_name", "doc.txt"),
        "content": kwargs.get("content", ""),
        "score": kwargs.get("score", 1.0),
    }
    base.update(kwargs)
    return RetrievedChunk(**base)


def test_a_pet_shelter_pptx_compose_uses_pet_friendly_attribute():
    shelters = _chunk(
        chunk_id="sakuragi",
        document_name="06_disaster_preparedness.pptx",
        content=(
            "• Evacuation Shelters in Hikari City\n"
            "• Hikari Elementary School Gymnasium (2-4-1 Chuo) — main shelter, capacity 800\n"
            "• Sakuragi Community Center (1-9-2 Sakuragi) — capacity 300, pet-friendly area\n"
            "• Areas along the Hikari River are in the flood hazard zone."
        ),
        page_number=3,
        slide_number=3,
        content_type="list",
        score=9.5,
    )
    alert_levels = _chunk(
        chunk_id="alerts",
        document_name="06_disaster_preparedness.pptx",
        content=(
            "• Japan's 5 Alert Levels\n"
            "• Level 1 — Stay aware: early weather information issued\n"
            "• Level 5 — Emergency safety measures: act immediately to protect your life"
        ),
        page_number=2,
        slide_number=2,
        content_type="list",
        score=5.0,
    )

    # Different PPTX slides must not be glued solely because both are bullet lists.
    expanded = expand_sentence_boundaries([alert_levels, shelters])
    assert len(expanded) == 2
    assert expanded[0].chunk_id == "alerts"
    assert expanded[1].chunk_id == "sakuragi"

    understanding = understand_query("Which evacuation shelter accepts pets?")
    answer, sources = compose_answer(
        understanding=understanding, evidence=[shelters, alert_levels]
    )
    assert "could not find" not in answer.lower()
    assert "Sakuragi" in answer
    assert "pet-friendly" in answer.lower()
    assert sources
    assert any(
        (s.document_name or "").endswith(".pptx") or "disaster" in (s.document_name or "").lower()
        for s in sources
    )


def test_b_parking_permit_never_uses_unrelated_koseki_fee():
    question = "How much is a residential parking permit?"
    koseki = _chunk(
        chunk_id="koseki",
        document_name="05_city_fees_schedule.csv",
        content=(
            "certificate_or_service: Copy of family register (Koseki Tohon) - "
            "for registered domicile in Hikari; fee_jpy: 550; "
            "where_to_apply: Citizen Affairs Division Window 5"
        ),
        content_type="structured_table_row",
        document_status="current",
        score=8.0,
    )
    residence = _chunk(
        chunk_id="juminhyo",
        document_name="05_city_fees_schedule.csv",
        content=(
            "certificate_or_service: Residence certificate (Juminhyo); "
            "fee_jpy: 350; where_to_apply: Citizen Affairs Division Window 3"
        ),
        content_type="structured_table_row",
        document_status="current",
        score=7.0,
    )

    kept, diag = filter_evidence_for_requested_entity([koseki, residence], question)
    assert kept == []
    assert diag.get("rejection_reason") == "no_explicit_entity_match"
    assert relevant_price_evidence(question, [koseki, residence]) == []
    assert not answer_matches_requested_entity(
        "Copy of family register (Koseki Tohon) costs ¥550 per copy.",
        question,
    )

    understanding = understand_query(question)
    answer, sources = compose_answer(
        understanding=understanding, evidence=[koseki, residence]
    )
    assert "could not find that information" in answer.lower()
    assert "contact the organization directly" in answer.lower()
    assert "the city" not in answer.lower()
    assert "koseki" not in answer.lower()
    assert "family register" not in answer.lower()
    assert "¥" not in answer and "550" not in answer
    assert sources == []


def test_c_move_in_keeps_deadline_and_complete_checklist():
    question = (
        "I just moved to the city. When must I register, and what do I bring?"
    )
    move_in = _chunk(
        chunk_id="move_in",
        document_id="reg",
        document_name="01_resident_registration_guide.pdf",
        section_title="Moving In (Tennyu Todoke)",
        content=(
            "Moving In (Tennyu Todoke). If you move into Hikari City, submit a "
            "move-in notification within 14 days of moving at Citizen Affairs "
            "Division Window 3. Bring: your residence card (or passport with "
            "landing permission), Moving-Out Certificate from your previous city, "
            "and My Number documents for all household members."
        ),
        page_number=2,
        content_type="list",
        service_domain="resident_registration",
        score=9.0,
    )
    bring_sibling = _chunk(
        chunk_id="bring_more",
        document_id="reg",
        document_name="01_resident_registration_guide.pdf",
        section_title="Moving In (Tennyu Todoke)",
        content=(
            "• National Health Insurance enrollment form (if applicable)\n"
            "• Lease agreement or proof of address"
        ),
        page_number=2,
        content_type="list",
        service_domain="resident_registration",
        score=8.0,
    )
    move_out = _chunk(
        chunk_id="move_out",
        document_id="reg",
        document_name="01_resident_registration_guide.pdf",
        section_title="Moving Out / Moving Within the City",
        content=(
            "Moving Out / Moving Within the City Submit a move-out notification "
            "up to 14 days before you leave Hikari City; you will receive a "
            "Moving-Out Certificate."
        ),
        page_number=3,
        content_type="list",
        service_domain="resident_registration",
        score=8.5,
    )

    preferred = _penalize_conflicting_topic_evidence(
        [move_out, move_in, bring_sibling], question
    )
    assert preferred
    assert all(
        "Moving Out" not in (c.section_title or "")
        or "Moving In" in (c.content or "")
        for c in preferred
    )
    assert any("within 14 days of moving" in (c.content or "") for c in preferred)

    items = extract_required_items([move_in, bring_sibling])
    texts = " ".join(item for _, item in items).lower()
    assert "residence card" in texts or "passport" in texts
    assert "moving-out certificate" in texts or "my number" in texts
    # Sibling list without local "bring" wording still contributes items.
    assert "lease agreement" in texts or "national health insurance" in texts

    date_u = understand_query("When must I register?")
    date_u.query_type = "date"
    date_answer, _ = compose_answer(understanding=date_u, evidence=preferred)
    assert "could not find" not in date_answer.lower()
    assert "14 days" in date_answer.lower()

    checklist_u = understand_query("what do I bring?")
    checklist_u.query_type = "checklist"
    checklist_answer, _ = compose_answer(
        understanding=checklist_u, evidence=[move_in, bring_sibling]
    )
    assert "could not find" not in checklist_answer.lower()
    lower = checklist_answer.lower()
    assert "residence card" in lower or "passport" in lower
    # Must not be truncated mid-phrase.
    assert not lower.rstrip().endswith("landing")
    assert "(" not in lower or ")" in lower or "passport" in lower
