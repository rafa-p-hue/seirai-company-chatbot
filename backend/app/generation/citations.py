"""Citation extraction and source deduplication helpers."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set, Tuple

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
    allow_uncited_fallback: bool = False,
) -> List[CitationSource]:
    """Return only cited sources, deduped by document+page.

    By default, omit sources when the answer has no citation markers so unused
    retrieved chunks are not shown. Broad summaries may opt into a small fallback.
    """
    numbers = extract_citation_numbers(answer)
    selected_evidence: List[RetrievedChunk] = []
    if numbers:
        for number in numbers:
            if 1 <= number <= len(evidence):
                selected_evidence.append(evidence[number - 1])
    elif allow_uncited_fallback or query_type in {"identity", "summary"}:
        limit = 4 if query_type in {"identity", "summary"} else 2
        selected_evidence = list(evidence[:limit])

    return dedupe_sources(selected_evidence)


def dedupe_sources(evidence: Sequence[RetrievedChunk]) -> List[CitationSource]:
    sources: List[CitationSource] = []
    seen: Set[Tuple[str, int | None, int | None, int | None]] = set()
    for item in evidence:
        key = (
            item.document_name,
            item.page_number,
            item.slide_number,
            item.row_number,
        )
        if key in seen:
            continue
        seen.add(key)
        source_type = item.source_type
        if source_type is None:
            if item.source_url:
                source_type = SourceType.website
            elif item.file_type:
                try:
                    source_type = SourceType(item.file_type)
                except ValueError:
                    source_type = SourceType.pdf
            else:
                source_type = SourceType.pdf
        sources.append(
            CitationSource(
                number=len(sources) + 1,
                document_name=item.document_name,
                page_number=item.page_number,
                slide_number=item.slide_number,
                row_number=item.row_number,
                section_title=item.section_title,
                source_url=item.source_url,
                source_type=source_type,
                effective_date=item.effective_date,
                document_status=item.document_status,
            )
        )
    return sources


def cited_evidence(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> List[RetrievedChunk]:
    numbers = extract_citation_numbers(answer)
    if not numbers:
        return []
    out: List[RetrievedChunk] = []
    for number in numbers:
        if 1 <= number <= len(evidence):
            out.append(evidence[number - 1])
    return out


def remap_answer_citations(answer: str, mapping: Dict[int, int]) -> str:
    """Rewrite [n] markers using an old→new number map; drop unmapped markers."""

    def _replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        new = mapping.get(old)
        if new is None:
            return ""
        return f"[{new}]"

    remapped = CITATION_RE.sub(_replace, answer or "")
    remapped = re.sub(r"[ \t]{2,}", " ", remapped)
    remapped = re.sub(r" +\n", "\n", remapped)
    return remapped.strip()


def renumber_citations_contiguous(answer: str) -> Tuple[str, Dict[int, int]]:
    """Map first-seen citation numbers to 1..N in appearance order."""
    numbers = extract_citation_numbers(answer)
    mapping = {old: index for index, old in enumerate(numbers, start=1)}
    return remap_answer_citations(answer, mapping), mapping


def reconcile_answer_citations(
    answer: str, evidence: Sequence[RetrievedChunk]
) -> Tuple[str, List[CitationSource], List[RetrievedChunk]]:
    """Align inline evidence indices with a deduplicated displayed source list."""
    mapping: Dict[int, int] = {}
    sources: List[CitationSource] = []
    cited: List[RetrievedChunk] = []
    source_numbers: Dict[Tuple[str, int | None, str | None], int] = {}
    for old_number in extract_citation_numbers(answer):
        if not (1 <= old_number <= len(evidence)):
            continue
        item = evidence[old_number - 1]
        key = (item.document_name, item.page_number, item.source_url)
        new_number = source_numbers.get(key)
        if new_number is None:
            new_number = len(sources) + 1
            source_numbers[key] = new_number
            sources.append(
                CitationSource(
                    number=new_number,
                    document_name=item.document_name,
                    page_number=item.page_number,
                    source_url=item.source_url,
                    source_type=SourceType.website
                    if item.source_url
                    else SourceType.pdf,
                )
            )
            cited.append(item)
        mapping[old_number] = new_number
    return remap_answer_citations(answer, mapping), sources, cited
