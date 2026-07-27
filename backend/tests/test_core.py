import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_pages, clean_text
from app.ingestion.web_loader import UnsafeUrlError, validate_crawl_url
from app.models.api import ExtractedPage, SourceType
from app.generation.prompts import FALLBACK_ANSWER, build_user_prompt
from app.generation.qwen_provider import DeterministicFallbackProvider, format_sources
from app.models.api import RetrievedChunk


def test_clean_text_removes_nulls_and_normalizes_whitespace():
    text = clean_text("Hello\u0000   world\r\n\r\n\r\nNext")
    assert "\u0000" not in text
    assert "Hello world" in text
    assert "\n\n\n" not in text


def test_repeated_headers_removed_but_content_kept():
    pages = [
        ExtractedPage(page_number=1, text="Shared Header\nRefunds are reviewed within thirty days.\nPage 1"),
        ExtractedPage(page_number=2, text="Shared Header\nApprovals require proof of purchase.\nPage 2"),
        ExtractedPage(page_number=3, text="Shared Header\nDenied requests include a written reason.\nPage 3"),
    ]
    cleaned = clean_pages(pages)
    assert "Shared Header" not in cleaned[0].text
    assert "Refunds are reviewed" in cleaned[0].text


def test_chunk_overlap_and_deterministic_ids():
    pages = [
        ExtractedPage(
            page_number=1,
            text="SERVICES\n"
            + ("The company provides inventory automation and analytics. " * 40),
        ),
        ExtractedPage(
            page_number=2,
            text="CONTACT\n"
            + ("Customers can email support for help with dashboards. " * 40),
        ),
    ]
    first = chunk_pages(
        pages=pages,
        company_id="acme",
        document_id="doc-1",
        document_name="overview.pdf",
        target_min=50,
        target_max=120,
        overlap=20,
        min_useful=10,
    )
    second = chunk_pages(
        pages=pages,
        company_id="acme",
        document_id="doc-1",
        document_name="overview.pdf",
        target_min=50,
        target_max=120,
        overlap=20,
        min_useful=10,
    )
    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.company_id == "acme" for chunk in first)
    assert all(chunk.content_hash for chunk in first)


def test_unsafe_urls_rejected():
    with pytest.raises(UnsafeUrlError):
        validate_crawl_url("file:///etc/passwd")
    with pytest.raises(UnsafeUrlError):
        validate_crawl_url("http://localhost/admin")
    with pytest.raises(UnsafeUrlError):
        validate_crawl_url("http://127.0.0.1/secret")


def test_allowlist_enforced():
    with pytest.raises(UnsafeUrlError):
        validate_crawl_url("https://example.com", allowlist=["trusted.example"])


@pytest.mark.asyncio
async def test_prompt_and_fallback_without_evidence():
    provider = DeterministicFallbackProvider()
    answer, sources = await provider.generate(question="What is the refund policy?", evidence=[])
    assert answer == FALLBACK_ANSWER
    assert sources == []


@pytest.mark.asyncio
async def test_citation_formatting():
    evidence = [
        RetrievedChunk(
            content="Acme provides inventory automation.",
            document_name="overview.pdf",
            page_number=1,
            source_url=None,
            score=0.9,
        )
    ]
    sources = format_sources(evidence)
    assert sources[0].number == 1
    assert sources[0].document_name == "overview.pdf"
    prompt = build_user_prompt("What services does the company provide?", evidence)
    assert "[1]" in prompt
    assert "overview.pdf" in prompt
