"""Focused regressions for move-in vs insurance evidence hard rejection."""

from __future__ import annotations

from app.retrieval.procedure_context import (
    is_move_in_registration_question,
    reject_insurance_evidence_for_move_in,
)


def test_move_in_question_detection():
    assert is_move_in_registration_question(
        "I just moved to Hikari City. When must I register, and what do I bring?"
    )
    assert not is_move_in_registration_question(
        "I left employer insurance. When must I enroll, and what do I bring?"
    )


def test_hard_reject_insurance_dominated_evidence_for_move_in():
    question = "I just moved to the city. When must I register, and what do I bring?"
    assert reject_insurance_evidence_for_move_in(
        question=question,
        content=(
            "If you leave your employer's insurance, enroll in national health "
            "insurance (NHI). Visit Insurance & Pension Division Window 6. "
            "Bring certificate of loss of insurance. The NHI card is mailed separately."
        ),
        section_title="Leaving employer insurance",
        service_domain="health_insurance",
    )
    assert not reject_insurance_evidence_for_move_in(
        question=question,
        content=(
            "Submit a move-in notification at Citizen Affairs Division Window 3. "
            "Register within 14 days. Bring a residence card or passport with "
            "landing permission and My Number cards for all household members."
        ),
        section_title="Move-in notification",
        service_domain="resident_registration",
    )
