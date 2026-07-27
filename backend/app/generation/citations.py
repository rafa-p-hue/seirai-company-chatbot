"""Citation extraction and source deduplication helpers."""

from __future__ import annotations

import re
from typing import List, Sequence, Set, Tuple

from app.models.api import CitationSource, RetrievedChunk, SourceType


CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citation_numbers(answer: str) -> List[int]:
    seen: Set[int] = set()
    ordered: List[int] = []
    for match in CITATION_RE.finditer(answer or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            ordered.append(number)
    return ordered


def sources_from_answer(
    answer: str,
    evidence: Sequence[RetrievedChunk],
    *,
    query_type: str | None = None,
) -> List[CitationSource]:
    """Return only cited sources, deduped by document+page.

    If the model omitted citations but produced a grounded answer, fall back to
    a small set of top evidence pages (1–2 for one-fact, up to 4 for summaries).
    """
    numbers = extract_citation_numbers(answer)
    selected_evidence: List[RetrievedChunk] = []
    if numbers:
        for number in numbers:
            if 1 <= number <= len(evidence):
                selected_evidence.append(evidence[number - 1])
    else:
        limit = 4 if query_type in {"identity", "summary", "general"} else 2
        selected_evidence = list(evidence[:limit])

    return dedupe_sources(selected_evidence)


def dedupe_sources(evidence: Sequence[RetrievedChunk]) -> List[CitationSource]:
    sources: List[CitationSource] = []
    seen: Set[Tuple[str, int | None]] = set()
    for item in evidence:
        key = (item.document_name, item.page_number)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            CitationSource(
                number=len(sources) + 1,
                document_name=item.document_name,
                page_number=item.page_number,
                source_url=item.source_url,
                source_type=SourceType.website if item.source_url else SourceType.pdf,
            )
        )
    return sources


def cited_evidence(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> List[RetrievedChunk]:
    numbers = extract_citation_numbers(answer)
    if not numbers:
        return list(evidence[:2])
    out: List[RetrievedChunk] = []
    for number in numbers:
        if 1 <= number <= len(evidence):
            out.append(evidence[number - 1])
    return out
