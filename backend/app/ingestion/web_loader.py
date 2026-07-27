from __future__ import annotations

import ipaddress
import logging
import socket
from typing import List, Optional
from urllib.parse import urlparse

from app.models.api import ExtractedPage

logger = logging.getLogger(__name__)


class UnsafeUrlError(ValueError):
    """Raised when a URL fails SSRF / allowlist checks."""


def validate_crawl_url(url: str, allowlist: Optional[List[str]] = None) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must include a hostname.")

    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        raise UnsafeUrlError("Localhost crawling is blocked.")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Unable to resolve host: {host}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise UnsafeUrlError("Private or reserved network targets are blocked.")

    allowlist = [item.lower() for item in (allowlist or [])]
    if allowlist and host not in allowlist and not any(host.endswith(f".{d}") for d in allowlist):
        raise UnsafeUrlError(
            "Domain is not on the approved crawl allowlist. Set ALLOWED_CRAWL_DOMAINS."
        )

    return parsed.geturl()


async def crawl_website(
    url: str,
    *,
    max_pages: int = 20,
    allowlist: Optional[List[str]] = None,
) -> List[ExtractedPage]:
    """Crawl a website with Crawl4AI when available.

    Falls back to a single-page httpx fetch if Crawl4AI is unavailable so the
    rest of the pipeline remains testable.
    """
    validated = validate_crawl_url(url, allowlist=allowlist)

    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig  # type: ignore
    except Exception:  # noqa: BLE001
        logger.warning("Crawl4AI unavailable; using single-page httpx fallback")
        return await _httpx_single_page(validated)

    pages: List[ExtractedPage] = []
    config = CrawlerRunConfig(
        word_count_threshold=10,
        exclude_external_links=True,
        process_iframes=False,
    )

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=validated, config=config)
        markdown = (getattr(result, "markdown", None) or getattr(result, "cleaned_html", "") or "").strip()
        title = ""
        metadata = getattr(result, "metadata", None) or {}
        if isinstance(metadata, dict):
            title = str(metadata.get("title") or "")
        text = f"{title}\n\n{markdown}".strip() if title else markdown
        if text:
            pages.append(ExtractedPage(page_number=1, text=text))

        # Bounded breadth: follow same-host links up to max_pages when available.
        links = []
        result_links = getattr(result, "links", None) or {}
        if isinstance(result_links, dict):
            links = result_links.get("internal") or []

        seen = {validated}
        for link in links:
            if len(pages) >= max_pages:
                break
            href = link.get("href") if isinstance(link, dict) else str(link)
            if not href or href in seen:
                continue
            try:
                href = validate_crawl_url(href, allowlist=allowlist or [urlparse(validated).hostname or ""])
            except UnsafeUrlError:
                continue
            seen.add(href)
            child = await crawler.arun(url=href, config=config)
            child_md = (getattr(child, "markdown", None) or "").strip()
            if child_md:
                pages.append(ExtractedPage(page_number=len(pages) + 1, text=child_md))

    if not pages:
        raise ValueError("No usable text was extracted from the website.")

    return pages[:max_pages]


async def _httpx_single_page(url: str) -> List[ExtractedPage]:
    import httpx
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: List[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):  # noqa: ANN001
            if tag in {"script", "style", "noscript"}:
                self._skip = True

        def handle_endtag(self, tag):  # noqa: ANN001
            if tag in {"script", "style", "noscript"}:
                self._skip = False

        def handle_data(self, data):  # noqa: ANN001
            if not self._skip:
                text = data.strip()
                if text:
                    self.parts.append(text)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        parser = _TextExtractor()
        parser.feed(response.text)
        text = "\n".join(parser.parts).strip()
        if not text:
            raise ValueError("No usable text was extracted from the website.")
        return [ExtractedPage(page_number=1, text=text)]
