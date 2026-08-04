"""Benchmark case definitions for semantic RAG evaluation (tests only).

These cases must not be imported by production generation or retrieval modules.
"""

from __future__ import annotations

from tests.eval.semantic_eval import BenchmarkCase, fact

HC_E3 = BenchmarkCase(
    test_id="HC-E3",
    question="How much does a residence certificate (juminhyo) cost?",
    expected_facts=(
        fact(
            "counter_fee_current",
            "counter fee is ¥350",
            ("350", "counter"),
            ("350 yen", "counter"),
            ("350", "in person"),
            ("counter fee", "350"),
            critical=True,
        ),
        fact(
            "kiosk_fee_current",
            "convenience-store kiosk fee is ¥250",
            ("250", "kiosk"),
            ("250 yen", "kiosk"),
            ("250", "convenience"),
            ("kiosk fee", "250"),
            critical=True,
        ),
        fact(
            "effective_april_2026",
            "fees are effective April 1, 2026",
            ("april", "1", "2026"),
            ("april 1 2026",),
            ("effective", "2026"),
            critical=False,
        ),
    ),
    required_sources=(
        "fee schedule",
        "certificates and fees",
        "current",
        "resident registration",
        "Certificates_and_Fees",
    ),
    forbidden_facts=(
        fact(
            "archived_counter_300",
            "counter fee ¥300 presented as current",
            ("300", "counter"),
            ("300 yen", "counter"),
            ("counter fee", "300"),
            critical=True,
        ),
        fact(
            "archived_kiosk_200",
            "kiosk fee ¥200 presented as current",
            ("200", "kiosk"),
            ("200 yen", "kiosk"),
            ("kiosk fee", "200"),
            critical=True,
        ),
    ),
    optional_facts=(
        fact(
            "mentions_juminhyo",
            "mentions residence certificate / juminhyo",
            ("residence certificate",),
            ("juminhyo",),
            critical=False,
        ),
    ),
    evaluation_notes=(
        "Critical: both current prices and rejection of archived prices. "
        "Paraphrases and reordering are acceptable."
    ),
    compound_parts=1,
)


HC_E4 = BenchmarkCase(
    test_id="HC-E4",
    question=(
        "I just moved to Hikari City. When must I register, and what do I bring?"
    ),
    expected_facts=(
        fact(
            "move_in_notification",
            "submit a move-in notification",
            ("move in notification",),
            ("move-in notification",),
            ("moving in notification",),
            ("submit", "notification"),
            critical=True,
        ),
        fact(
            "window_3",
            "go to Citizen Affairs Window 3",
            ("window 3",),
            ("citizen affairs", "window 3"),
            ("citizen services", "window 3"),
            critical=True,
        ),
        fact(
            "deadline_14_days",
            "deadline is within 14 days of beginning to live at the new address",
            ("14 days",),
            ("within 14",),
            ("fourteen days",),
            critical=True,
        ),
        fact(
            "residence_or_passport",
            "bring a residence card, or passport with landing permission",
            ("residence card",),
            ("passport", "landing"),
            ("passport", "landing permission"),
            critical=True,
        ),
        fact(
            "moving_out_certificate",
            "bring a Moving-Out Certificate when moving from another municipality",
            ("moving out certificate",),
            ("moving-out certificate",),
            ("move out certificate",),
            critical=True,
        ),
        fact(
            "my_number_household",
            "bring My Number cards or notification cards for all household members",
            ("my number", "household"),
            ("notification card", "household"),
            ("my number", "all household"),
            ("identification", "household members"),
            critical=True,
        ),
    ),
    required_sources=(
        "resident registration",
        "moving in",
        "Resident_Registration",
        "new resident",
    ),
    forbidden_facts=(
        fact(
            "insurance_window_6",
            "Insurance & Pension Window 6",
            ("window 6",),
            ("insurance", "pension", "window"),
            critical=True,
        ),
        fact(
            "health_insurance_enrollment",
            "health-insurance enrollment instructions",
            ("health insurance", "enroll"),
            ("national health insurance",),
            ("employer", "insurance"),
            critical=True,
        ),
        fact(
            "loss_of_employer_insurance",
            "certificate of loss of employer insurance",
            ("certificate of loss",),
            ("loss of eligibility",),
            ("loss of employer",),
            critical=True,
        ),
        fact(
            "nhi_card_mailing",
            "NHI card mailing details",
            ("nhi card",),
            ("insurance card", "mail"),
            ("mailed separately",),
            critical=True,
        ),
    ),
    optional_facts=(),
    evaluation_notes=(
        "Critical: 14-day deadline, move-in notification, correct procedure/source, "
        "and required-document list. Bullets, paragraphs, and reordered facts are OK."
    ),
    compound_parts=2,
)


BENCHMARK_CASES = {
    HC_E3.test_id: HC_E3,
    HC_E4.test_id: HC_E4,
}
