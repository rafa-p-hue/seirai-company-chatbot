"""Regression: clean answer synthesis for procedures, CSV fees, truncation, multi-fee."""

from __future__ import annotations

from app.generation.answer_synthesis import (
    extract_checklist_structure,
    is_retrieval_dump_answer,
    normalize_evidence_bundle,
    synthesize_checklist_answer,
    synthesize_fee_answer,
)
from app.generation.evidence_presentation import (
    answer_exposes_internal_field_keys,
    format_fee_fields_as_prose,
    format_structured_content_as_prose,
    looks_like_answer_fragment,
    prepare_evidence_for_generation,
    repair_passage_text,
)
from app.generation.evidence_validation import (
    checklist_answer,
    checklist_answer_is_complete,
    extract_required_items,
    price_answer,
    price_answer_is_complete,
)
from app.models.api import RetrievedChunk
from app.services.chat_service import _post_validate_answer


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "content": "",
        "document_name": "doc.pdf",
        "page_number": 1,
        "score": 1.0,
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


def test_a_procedure_checklist_spans_chunks_without_fragments():
    evidence = [
        _chunk(
            content=(
                "Moving In (Tennyu Todoke)\n"
                "Submit a move-in notification at Citizen Affairs Division Window 3 "
                "within 14 days of beginning to live at the new address.\n"
                "Bring the following:\n"
                "• Residence card, or passport with landing permission"
            ),
            document_name="01_resident_registration_guide.pdf",
            section_title="Moving In (Tennyu Todoke)",
            content_type="list",
            page_number=2,
            service_domain="resident_registration",
        ),
        _chunk(
            content=(
                "• Moving-Out Certificate when moving from another municipality\n"
                "• My Number cards or notification cards for all household members"
            ),
            document_name="01_resident_registration_guide.pdf",
            section_title="Moving In (Tennyu Todoke)",
            content_type="list",
            page_number=2,
            service_domain="resident_registration",
        ),
    ]
    prepared, diag = normalize_evidence_bundle(
        evidence, question="What do I bring when I move in?"
    )
    assert len(prepared) == 1
    assert "Residence card" in prepared[0].content
    assert "Moving-Out Certificate" in prepared[0].content
    assert "My Number" in prepared[0].content

    items = extract_required_items(prepared)
    texts = [item for _, item in items]
    assert len(items) >= 3, texts
    assert any("Residence card" in item or "passport" in item.lower() for item in texts)
    assert any("Moving-Out Certificate" in item for item in texts)
    assert any("My Number" in item for item in texts)

    structure = extract_checklist_structure(prepared)
    assert len(structure) >= 3
    assert diag["checklist_structure"]

    answer = checklist_answer(prepared)
    assert "Required items:" not in answer
    assert "Applicable fees:" not in answer
    assert not answer.startswith("of the day")
    assert "Moving In (Tennyu Todoke) If you" not in answer
    low = answer.lower()
    assert "residence card" in low or "passport" in low
    assert "moving-out certificate" in low or "moving out certificate" in low
    assert "my number" in low
    assert checklist_answer_is_complete(answer, prepared)
    assert not looks_like_answer_fragment(answer)
    assert not answer_exposes_internal_field_keys(answer)
    assert not is_retrieval_dump_answer(answer)


def test_b_csv_fee_rows_become_natural_language():
    evidence = [
        _chunk(
            content=(
                "certificate_or_service: Residence certificate (Juminhyo) - counter; "
                "fee_jpy: 350; where_to_apply: Citizen Affairs Division Window 3; "
                "notes: Per copy. Effective April 1 2026"
            ),
            document_name="05_city_fees_schedule.csv",
            file_type="csv",
            content_type="table",
            row_number=1,
            document_status="current",
            table_data={
                "column_labels": [
                    "certificate_or_service",
                    "fee_jpy",
                    "where_to_apply",
                    "notes",
                ],
                "cells": [
                    "Residence certificate (Juminhyo) - counter",
                    "350",
                    "Citizen Affairs Division Window 3",
                    "Per copy. Effective April 1 2026",
                ],
            },
        ),
        _chunk(
            content=(
                "certificate_or_service: Residence certificate (Juminhyo) - "
                "convenience store kiosk; fee_jpy: 250; where_to_apply: Any konbini "
                "multicopy machine nationwide; notes: Requires My Number Card. "
                "Effective April 1 2026"
            ),
            document_name="05_city_fees_schedule.csv",
            file_type="csv",
            content_type="table",
            row_number=2,
            document_status="current",
        ),
    ]
    question = "How much does a residence certificate (juminhyo) cost?"
    answer = price_answer(question, evidence)
    assert "Applicable fees:" not in answer
    assert "350" in answer
    assert "250" in answer
    assert "¥350" in answer or "350" in answer
    assert "Effective" in answer or "April" in answer
    for banned in (
        "certificate_or_service",
        "fee_jpy",
        "where_to_apply",
    ):
        assert banned not in answer
    assert "notes:" not in answer
    assert price_answer_is_complete(answer, question, evidence)
    assert not answer_exposes_internal_field_keys(answer)
    assert not is_retrieval_dump_answer(answer)

    prepared = prepare_evidence_for_generation(evidence, question=question)
    for chunk in prepared:
        assert "fee_jpy" not in chunk.content
        assert "certificate_or_service" not in chunk.content
        assert "costs" in chunk.content.lower() or "¥" in chunk.content


def test_c_truncated_source_chunk_reconstructed_before_generation():
    sibling_a = _chunk(
        content=(
            "Submit a move-in notification within 14 days "
            "of the day you start living at your new address. Bring: your residence "
            "card (or passport with landing"
        ),
        document_name="01_resident_registration_guide.pdf",
        section_title="Moving In (Tennyu Todoke)",
        page_number=2,
        content_type="paragraph",
    )
    sibling_b = _chunk(
        content=(
            "permission), your Moving-Out Certificate when moving from another "
            "municipality, and My Number cards for all household members."
        ),
        document_name="01_resident_registration_guide.pdf",
        section_title="Moving In (Tennyu Todoke)",
        page_number=2,
        content_type="paragraph",
    )
    alone = prepare_evidence_for_generation(
        [
            _chunk(
                content=(
                    "of the day you start living at your new address. Bring: your "
                    "residence card (or passport with landing"
                ),
                document_name="guide.pdf",
                section_title="Moving In",
                page_number=1,
            )
        ]
    )
    repaired = alone[0].content
    assert not repaired.lower().startswith("of the day")
    assert repair_passage_text(
        "of the day you start living at your new address. Bring your card."
    ).startswith("Bring")

    prepared = prepare_evidence_for_generation([sibling_a, sibling_b])
    assert len(prepared) == 1
    fused = prepared[0].content
    assert "landing permission" in fused.lower()
    assert "Moving-Out Certificate" in fused or "moving-out certificate" in fused.lower()
    assert "My Number" in fused
    assert not fused.lower().startswith("of the day")
    assert not fused.rstrip().endswith("landing")


def test_d_multiple_fee_options_preserve_amounts_and_conditions():
    evidence = [
        _chunk(
            content=(
                "certificate_or_service: Residence certificate (Juminhyo) - counter; "
                "fee_jpy: 350; where_to_apply: Citizen Affairs Division Window 3; "
                "notes: Per copy. Effective April 1 2026"
            ),
            document_name="05_city_fees_schedule.csv",
            file_type="csv",
            content_type="table",
            row_number=1,
            document_status="current",
        ),
        _chunk(
            content=(
                "certificate_or_service: Residence certificate (Juminhyo) - "
                "convenience store kiosk; fee_jpy: 250; where_to_apply: Any konbini "
                "multicopy machine nationwide; notes: Requires My Number Card. "
                "Effective April 1 2026"
            ),
            document_name="05_city_fees_schedule.csv",
            file_type="csv",
            content_type="table",
            row_number=2,
            document_status="current",
        ),
    ]
    question = "How much does a residence certificate (juminhyo) cost?"
    answer = synthesize_fee_answer(question, evidence)
    assert "350" in answer and "250" in answer
    assert "April" in answer or "2026" in answer
    assert (
        "My Number" in answer
        or "kiosk" in answer.lower()
        or "konbini" in answer.lower()
    )
    assert "certificate_or_service" not in answer
    assert "fee_jpy" not in answer


def test_c_snake_case_metadata_hidden_from_users():
    raw = (
        "certificate_or_service: Residence certificate (Juminhyo); "
        "fee_jpy: 350; where_to_apply: Window 3; notes: Per copy"
    )
    prose = format_structured_content_as_prose(
        raw,
        _chunk(content=raw, file_type="csv", content_type="table", row_number=1),
    )
    assert "fee_jpy" not in prose
    assert "certificate_or_service" not in prose
    assert "where_to_apply" not in prose
    assert "¥350" in prose or "350" in prose
    assert "Residence certificate" in prose

    assert answer_exposes_internal_field_keys(
        "Applicable fees: - certificate_or_service: Foo; fee_jpy: 350"
    )
    assert not answer_exposes_internal_field_keys(
        "Residence certificate (Juminhyo) costs ¥350 per copy at Window 3."
    )

    rewritten = _post_validate_answer(
        "Applicable fees:\n- certificate_or_service: Residence certificate; fee_jpy: 350 [1]",
        "How much does a residence certificate (juminhyo) cost?",
        "price",
        "price",
        [
            _chunk(
                content=raw,
                document_name="05_city_fees_schedule.csv",
                file_type="csv",
                content_type="table",
                row_number=1,
                document_status="current",
            )
        ],
    )
    assert "fee_jpy" not in rewritten
    assert "certificate_or_service" not in rewritten
    assert "Applicable fees:" not in rewritten
    assert "350" in rewritten


def test_fee_fields_formatter_example():
    prose = format_fee_fields_as_prose(
        {
            "certificate_or_service": "Residence certificate (Juminhyo)",
            "fee_jpy": "350",
            "where_to_apply": "Citizen Affairs Division Window 3",
            "notes": "Per copy. Effective April 1 2026",
        }
    )
    assert "Residence certificate (Juminhyo)" in prose
    assert "costs" in prose.lower()
    assert "¥350" in prose
    assert "per copy" in prose.lower()
    assert "Citizen Affairs Division Window 3" in prose
    assert "Effective" in prose
    assert "fee_jpy" not in prose


def test_post_validate_rejects_mid_sentence_checklist_dump():
    evidence = [
        _chunk(
            content=(
                "Bring the following:\n"
                "• Residence card, or passport with landing permission\n"
                "• Moving-Out Certificate when moving from another municipality\n"
                "• My Number cards for all household members"
            ),
            document_name="01_resident_registration_guide.pdf",
            section_title="Moving In (Tennyu Todoke)",
            content_type="list",
        )
    ]
    rewritten = _post_validate_answer(
        "of the day you start living at your new address. Bring: your residence "
        "card (or passport with landing",
        "What do I bring for move-in?",
        "checklist",
        "checklist",
        evidence,
    )
    assert not rewritten.lower().startswith("of the day")
    assert "landing" not in rewritten.lower() or "permission" in rewritten.lower()
    assert checklist_answer_is_complete(rewritten, evidence)
    assert "My Number" in rewritten or "my number" in rewritten.lower()


def test_synthesize_checklist_is_not_passage_dump():
    evidence = [
        _chunk(
            content=(
                "Moving In (Tennyu Todoke) If you move to Hikari City, submit a "
                "notification. Bring the following:\n"
                "• Residence card, or passport with landing permission\n"
                "• Moving-Out Certificate when moving from another municipality"
            ),
            section_title="Moving In (Tennyu Todoke)",
            content_type="list",
        )
    ]
    answer = synthesize_checklist_answer(evidence)
    assert "Moving In (Tennyu Todoke) If you move" not in answer
    assert answer.startswith("Bring")
    assert "Residence card" in answer or "passport" in answer.lower()
