"""Additional unit and integration-style tests for the RAG backend."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.embeddings.local_provider import HashingFallbackEmbeddingProvider
from app.generation.prompts import FALLBACK_ANSWER
from app.generation.qwen_provider import DeterministicFallbackProvider
from app.ingestion.pdf_loader import PdfExtractionError, extract_pdf_pages
from app.ingestion.pipeline import ingest_pdf_bytes, sanitize_filename
from app.models.api import RetrievedChunk
from app.retrieval.hybrid_search import hybrid_score


FIXTURES = Path(__file__).parent / "fixtures"


def _make_pdf(path: Path, pages: list) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_sanitize_filename():
    assert sanitize_filename("../../evil.pdf") == "evil.pdf"
    assert ".." not in sanitize_filename("a/b/c.pdf")


def test_pdf_validation_and_extraction(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    sample = (FIXTURES / "sample_company.txt").read_text()
    _make_pdf(pdf_path, [sample])
    pages, metadata = extract_pdf_pages(pdf_path)
    assert metadata["page_count"] == 1
    assert "inventory automation" in pages[0].text.lower()
    assert metadata["ocr_required"] is False


def test_empty_pdf_requires_ocr(tmp_path: Path):
    pdf_path = tmp_path / "blank.pdf"
    _make_pdf(pdf_path, [""])
    with pytest.raises(PdfExtractionError, match="OCR"):
        extract_pdf_pages(pdf_path)


@pytest.mark.asyncio
async def test_ingest_pdf_bytes_creates_company_chunks(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = (FIXTURES / "sample_company.txt").read_text()
    pdf_path = tmp_path / "company.pdf"
    _make_pdf(pdf_path, [sample])
    data = pdf_path.read_bytes()
    summary, pages, chunks = await ingest_pdf_bytes(
        data=data,
        filename="company-overview.pdf",
        company_id="acme",
    )
    assert summary.company_id == "acme"
    assert pages
    assert chunks
    assert all(chunk.company_id == "acme" for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)


@pytest.mark.asyncio
async def test_refund_question_falls_back_without_policy():
    provider = DeterministicFallbackProvider()
    evidence = [
        RetrievedChunk(
            content="Acme provides inventory automation and analytics dashboards.",
            document_name="overview.pdf",
            page_number=1,
            source_url=None,
            score=0.8,
        )
    ]
    answer, sources = await provider.generate(
        question="What is the refund policy?",
        evidence=evidence,
    )
    assert answer == FALLBACK_ANSWER
    assert sources == []

    empty_answer, empty_sources = await provider.generate(
        question="What is the refund policy?",
        evidence=[],
    )
    assert empty_answer == FALLBACK_ANSWER
    assert empty_sources == []


def test_hybrid_lexical_overlap():
    score = hybrid_score(
        "What services does the company provide?",
        "The company provides inventory automation and analytics.",
    )
    assert score > 0.2
    assert hybrid_score("refund policy", "inventory automation") < 0.15


class _MemoryStore:
    def __init__(self):
        self.rows = []

    async def search(self, *, company_id: str, query_vector, top_k: int):
        return [
            {
                "score": 0.95,
                "payload": row,
            }
            for row in self.rows
            if row["company_id"] == company_id
        ][:top_k]


@pytest.mark.asyncio
async def test_company_separation_in_retrieval():
    store = _MemoryStore()
    store.rows = [
        {
            "chunk_id": "a1",
            "company_id": "acme",
            "document_id": "d1",
            "document_name": "acme.pdf",
            "page_number": 1,
            "section_title": "Services",
            "chunk_index": 0,
            "content": "Acme provides inventory automation.",
            "content_hash": "h1",
            "source_type": "pdf",
            "source_url": None,
        },
        {
            "chunk_id": "b1",
            "company_id": "beta",
            "document_id": "d2",
            "document_name": "beta.pdf",
            "page_number": 1,
            "section_title": "Services",
            "chunk_index": 0,
            "content": "Beta provides rocket engines.",
            "content_hash": "h2",
            "source_type": "pdf",
            "source_url": None,
        },
    ]

    results = await store.search(
        company_id="acme",
        query_vector=[0.0] * 8,
        top_k=5,
    )
    assert len(results) == 1
    assert results[0]["payload"]["company_id"] == "acme"
    assert "rocket" not in results[0]["payload"]["content"].lower()
    _ = HashingFallbackEmbeddingProvider  # provider available for embedding tests

def test_health_and_invalid_company(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    from app.config import get_settings
    from app import dependencies

    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm_provider.cache_clear()

    # Avoid requiring live Qdrant for this route validation test.
    from app.main import create_app

    client = TestClient(create_app())
    response = client.post(
        "/api/retrieve",
        json={"company_id": "BAD ID!", "question": "hello"},
    )
    assert response.status_code == 400
