"""Regression: checklist sibling assembly for split requirement lists."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.embeddings.local_provider import HashingFallbackEmbeddingProvider
from app.generation.answer_synthesis import (
    extract_checklist_structure,
    normalize_evidence_bundle,
)
from app.models.api import DocumentChunk, SourceType
from app.retrieval.checklist_assembly import (
    expand_checklist_siblings,
    merged_checklist_text,
    requirement_list_looks_incomplete,
)
from app.retrieval.procedure_context import build_procedure_context
from app.retrieval.retriever import Retriever
from app.vector_store.memory_store import InMemoryVectorStore


SECTION = "Moving In (Tennyu Todoke)"

# Semantic expectation groups for the regression (test-only; not production facts).
EXPECTED_GROUPS = {
    "identity_or_alternative": ("residence card", "passport"),
    "conditional_certificate": ("moving-out", "certificate"),
    "household_document": ("my number", "household"),
}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.chdir(tmp_path)

    from app import dependencies
    from app.main import create_app

    get_settings.cache_clear()
    for name in (
        "get_embedding_provider",
        "get_vector_store",
        "get_llm_provider",
    ):
        getattr(dependencies, name).cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
    for name in (
        "get_embedding_provider",
        "get_vector_store",
        "get_llm_provider",
    ):
        getattr(dependencies, name).cache_clear()


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    content: str,
    content_type: str = "list",
    section_title: str = SECTION,
    company_id: str = "sibling-check",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        company_id=company_id,
        document_id=document_id,
        document_name="resident-guide.pdf",
        page_number=1,
        section_title=section_title,
        chunk_index=chunk_index,
        content=content,
        content_hash=chunk_id,
        source_type=SourceType.pdf,
        content_type=content_type,
        uploaded_at=datetime.utcnow(),
        service_domain="resident_registration",
    )


def _split_chunks(company_id: str = "sibling-check") -> list[DocumentChunk]:
    return [
        _chunk(
            chunk_id=f"{company_id}-c-intro",
            document_id=f"{company_id}-doc",
            chunk_index=0,
            company_id=company_id,
            content=(
                "Moving In (Tennyu Todoke)\n"
                "Submit a move-in notification at Citizen Affairs Division "
                "Window 3 within 14 days of beginning to live at the new address.\n"
                "Bring the following:\n"
                "• Residence card, or passport with landing permission"
            ),
        ),
        _chunk(
            chunk_id=f"{company_id}-c-tail",
            document_id=f"{company_id}-doc",
            chunk_index=1,
            company_id=company_id,
            content=(
                "• Moving-Out Certificate when moving from another municipality\n"
                "• My Number cards or notification cards for all household members"
            ),
        ),
        _chunk(
            chunk_id=f"{company_id}-c-hours",
            document_id=f"{company_id}-doc",
            chunk_index=2,
            company_id=company_id,
            content="Monday through Friday: 8:30 a.m. to 5:00 p.m.",
            content_type="key_value",
            section_title="Office Hours",
        ),
    ]


async def _seed_store(
    store: InMemoryVectorStore,
    *,
    company_id: str = "sibling-check",
    dimensions: int = 384,
) -> None:
    embeddings = HashingFallbackEmbeddingProvider(dimensions=dimensions)
    chunks = _split_chunks(company_id)
    vectors = await embeddings.embed_texts([c.content for c in chunks])
    await store.upsert_chunks(chunks, vectors)


def _assert_requirement_groups_present(blob: str) -> None:
    low = blob.lower()
    assert any(token in low for token in EXPECTED_GROUPS["identity_or_alternative"])
    assert any(token in low for token in EXPECTED_GROUPS["conditional_certificate"])
    assert "my number" in low
    assert "household" in low


def test_requirement_list_incomplete_when_only_first_bullet():
    text = (
        "Bring the following:\n"
        "• Residence card, or passport with landing permission"
    )
    assert requirement_list_looks_incomplete(text)
    assert not requirement_list_looks_incomplete(
        text
        + "\n• Moving-Out Certificate when moving from another municipality\n"
        "• My Number cards or notification cards for all household members"
    )


@pytest.mark.asyncio
async def test_expand_checklist_siblings_loads_tail_bullets():
    store = InMemoryVectorStore()
    await _seed_store(store)
    payloads = await store.list_payloads("sibling-check")
    selected = [
        {
            "payload": next(
                p for p in payloads if str(p["chunk_id"]).endswith("c-intro")
            ),
            "combined": 1.0,
            "rerank": 1.0,
        }
    ]
    assert "Moving-Out" not in selected[0]["payload"]["content"]
    assert "My Number" not in selected[0]["payload"]["content"]

    expanded, diag = expand_checklist_siblings(selected, payloads, limit=8)
    merged = merged_checklist_text(expanded)
    assert "Residence card" in merged or "passport" in merged.lower()
    assert "Moving-Out Certificate" in merged
    assert "My Number" in merged
    assert diag["siblings_added"]
    assert any(str(row["chunk_id"]).endswith("c-tail") for row in diag["siblings_added"])
    assert "8:30" not in merged


@pytest.mark.asyncio
async def test_retriever_checklist_evidence_includes_sibling_groups():
    store = InMemoryVectorStore()
    await _seed_store(store, dimensions=384)
    embeddings = HashingFallbackEmbeddingProvider(dimensions=384)
    settings = Settings(
        APP_ENV="development",
        VECTOR_STORE="memory",
        EMBEDDING_PROVIDER="hash",
        LLM_PROVIDER="deterministic",
        EMBEDDING_DIMENSION=384,
    )
    retriever = Retriever(settings=settings, embeddings=embeddings, store=store)
    procedure = build_procedure_context(
        "I just moved to Hikari City. When must I register, and what do I bring?"
    )
    procedure.active_document = "resident-guide.pdf"
    procedure.active_section = SECTION

    evidence, _understanding, inspection = await retriever.retrieve(
        company_id="sibling-check",
        question="What documents are required for the same move-in procedure?",
        top_k=5,
        procedure_context=procedure,
        domain_source_question=(
            "I just moved to Hikari City. When must I register, and what do I bring?"
        ),
    )

    payload_blob = "\n".join(chunk.content or "" for chunk in evidence)
    # Temporary hard assertion: evidence payload must contain all requirement
    # groups before generation begins.
    _assert_requirement_groups_present(payload_blob)

    prepared, _diag = normalize_evidence_bundle(
        evidence, question="What do I bring?"
    )
    structure = extract_checklist_structure(prepared)
    assert len(structure) >= 3, structure
    assert any(
        entry.get("condition") or entry.get("type") == "certificate"
        for entry in structure
    )
    assert any(
        entry.get("scope") or entry.get("type") == "household_document"
        for entry in structure
    )
    assert inspection.checklist_assembly.get("siblings_added") or len(evidence) >= 2


def test_compound_chat_with_split_siblings_returns_complete_checklist(client):
    """End-to-end: split siblings must reach the checklist LLM payload."""
    import asyncio

    from app.dependencies import get_embedding_provider, get_vector_store

    store = get_vector_store()
    assert isinstance(store, InMemoryVectorStore)
    embeddings = get_embedding_provider()
    chunks = _split_chunks("sibling-check")
    vectors = asyncio.get_event_loop().run_until_complete(
        embeddings.embed_texts([c.content for c in chunks])
    )
    asyncio.get_event_loop().run_until_complete(store.upsert_chunks(chunks, vectors))

    response = client.post(
        "/api/chat",
        json={
            "company_id": "sibling-check",
            "question": (
                "I just moved to Hikari City. When must I register, "
                "and what do I bring?"
            ),
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    answer = body["answer"]
    low = answer.lower()
    assert "14 days" in low or "window" in low
    assert "residence card" in low or "passport" in low
    assert "moving-out" in low or "moving out" in low
    assert "my number" in low
    assert "household" in low

    subs = body.get("diagnostics", {}).get("sub_question_results") or []
    bring = next(
        (
            sub
            for sub in subs
            if "bring" in (sub.get("question") or "").lower()
            or sub.get("fact_type") == "checklist"
        ),
        None,
    )
    assert bring is not None, subs
    evidence_blob = "\n".join(
        (item.get("content") or "")
        for item in (bring.get("evidence_sent_to_llm") or [])
    )
    _assert_requirement_groups_present(evidence_blob)
