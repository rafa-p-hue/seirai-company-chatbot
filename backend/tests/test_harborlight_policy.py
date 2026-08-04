"""Multi-section policy/handbook retrieval regressions.

Uses a Harborlight-style community handbook as a *test fixture only*.
No Harborlight names, prices, or facts appear in production code.
"""

from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.generation.text_scrub import scrub_internal_metadata
from app.retrieval.query_understanding import classify_query


# Invented multi-section handbook for section-aware retrieval tests.
HARBORLIGHT_FIXTURE = """
Community Handbook Overview
Welcome to the community market and garden program. This overview summarizes
the organization at a high level and is not a substitute for the detailed
policies in the sections below.

Membership
Annual individual membership costs $120 per year.
Household membership costs $200 per year.
Pets are allowed in outdoor garden areas only when leashed.
Pets are not allowed inside the indoor market hall.

Volunteer Program
New volunteers must complete a 90-minute orientation before their first shift.
Volunteer shifts are scheduled on Saturday mornings from 9:00 a.m. to 12:00 p.m.

Food Donation Program
The program redistributes approximately 18,000 pounds of donated food each year
to partner pantries.

Sustainability Practices
Accepted compost materials include fruit scraps, vegetable scraps, and coffee grounds.
The following items are prohibited in compost bins:
- Meat and dairy products
- Plastic bags and packaging
- Treated wood scraps

Event Rentals
Private event rentals are available for the pavilion after market hours.
Alcohol is permitted only with a pre-approved catering license.
Event rentals may not extend past 10:00 p.m.

Cancellation and Refund Policy
Cancellations made at least 14 days before an event receive a full refund.
Cancellations made fewer than 14 days before an event are not eligible for a refund.

Accessibility
The indoor market hall has wheelchair-accessible entrances and restrooms.
Accessible parking is available near the north entrance.

Future Plans
The planned opening period for the expanded garden wing is Spring 2027.
The final street address for the expansion site has not been announced.
"""


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.chdir(tmp_path)

    from app.config import get_settings
    from app import dependencies
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


def _pdf_bytes(text: str) -> bytes:
    safe = text.replace("–", "-").replace("—", "-")
    doc = fitz.open()
    page = doc.new_page()
    y = 56
    for line in safe.splitlines():
        if not line.strip():
            y += 10
            continue
        page.insert_text((48, y), line[:110], fontsize=10)
        y += 14
        if y > 780:
            page = doc.new_page()
            y = 56
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client: TestClient, company_id: str = "harbor-test"):
    response = client.post(
        "/api/documents/upload",
        data={"company_id": company_id},
        files={
            "file": (
                "community-handbook.pdf",
                _pdf_bytes(HARBORLIGHT_FIXTURE),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _chat(client: TestClient, question: str, company_id: str = "harbor-test"):
    response = client.post(
        "/api/chat",
        json={"company_id": company_id, "question": question, "history": []},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_policy_question_intents():
    assert classify_query("How much does membership cost?") == "price"
    assert classify_query("Are pets allowed?") == "policy"
    assert classify_query("When is volunteer orientation?") == "policy"
    assert classify_query("How many pounds of donated food each year?") == "quantity"
    assert classify_query("What is prohibited in compost?") == "policy"
    assert classify_query("Can events serve alcohol?") == "policy"
    assert classify_query("What is the cancellation and refund policy?") == "policy"
    assert classify_query("Is the location accessible?") == "accessibility"
    assert classify_query("When is the planned opening period?") == "date"
    assert classify_query("What is the expansion address?") in {"location", "policy", "accessibility"}


def test_scrub_internal_metadata():
    raw = "Record type: profile\nPerson: Sample Name\nProfile Description:\nHello world"
    cleaned = scrub_internal_metadata(raw)
    assert "record type" not in cleaned.lower()
    assert "person:" not in cleaned.lower()
    assert "profile description" not in cleaned.lower()
    assert "hello world" in cleaned.lower()


def test_membership_price_not_overview(client: TestClient):
    _upload(client)
    body = _chat(client, "How much does membership cost?")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "$120" in answer or "120" in answer
    assert "welcome to the community" not in answer
    assert "record type" not in answer


def test_pet_policy(client: TestClient):
    _upload(client, "harbor-pets")
    body = _chat(client, "Are pets allowed?", "harbor-pets")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "pet" in answer or "leash" in answer or "allowed" in answer
    assert "indoor" in answer or "not allowed" in answer or "market hall" in answer


def test_volunteer_schedule(client: TestClient):
    _upload(client, "harbor-vol")
    body = _chat(client, "When is volunteer orientation and when are shifts?", "harbor-vol")
    print("VOLUNTEER BODY:", body)
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "orientation" in answer or "90" in answer or "saturday" in answer


def test_donation_quantity(client: TestClient):
    _upload(client, "harbor-food")
    body = _chat(client, "How many pounds of donated food each year?", "harbor-food")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "18,000" in answer or "18000" in answer or "18 000" in answer


def test_compost_prohibitions(client: TestClient):
    _upload(client, "harbor-compost")
    body = _chat(client, "What items are prohibited in compost?", "harbor-compost")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "meat" in answer or "plastic" in answer or "dairy" in answer
    assert "record type" not in answer


def test_event_rental_restrictions(client: TestClient):
    _upload(client, "harbor-rent")
    body = _chat(client, "Can private events serve alcohol?", "harbor-rent")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "alcohol" in answer
    assert "license" in answer or "permit" in answer or "approved" in answer


def test_cancellation_refund_rules(client: TestClient):
    _upload(client, "harbor-refund")
    body = _chat(client, "What is the cancellation and refund policy?", "harbor-refund")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "14" in answer
    assert "refund" in answer


def test_accessibility_location(client: TestClient):
    _upload(client, "harbor-access")
    body = _chat(client, "Is the market hall accessible for wheelchairs?", "harbor-access")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "wheelchair" in answer or "accessible" in answer


def test_planned_opening_period(client: TestClient):
    _upload(client, "harbor-open")
    body = _chat(client, "When is the planned opening period?", "harbor-open")
    answer = body["answer"].lower()
    assert "could not find" not in answer
    assert "2027" in answer or "spring" in answer


def test_unannounced_future_address(client: TestClient):
    _upload(client, "harbor-addr")
    body = _chat(client, "What is the expansion site address?", "harbor-addr")
    answer = body["answer"].lower()
    assert "not been announced" in answer or "not announced" in answer
    assert "record type" not in answer


def test_unsupported_question_fallback(client: TestClient):
    _upload(client, "harbor-unsup")
    body = _chat(client, "What is the CEO's zodiac sign?", "harbor-unsup")
    assert "could not find" in body["answer"].lower()


def test_overview_does_not_dominate_specific_fact(client: TestClient):
    _upload(client, "harbor-ov")
    body = _chat(client, "What compost materials are prohibited?", "harbor-ov")
    answer = body["answer"].lower()
    diagnostics = body.get("diagnostics") or {}
    final = diagnostics.get("final_chunks_sent_to_llm") or []
    joined = " ".join(chunk.get("content", "").lower() for chunk in final)
    assert "welcome to the community" not in answer
    assert "meat" in answer or "plastic" in answer or "prohibited" in joined
