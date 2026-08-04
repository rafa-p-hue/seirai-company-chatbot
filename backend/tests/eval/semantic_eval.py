"""Structured semantic RAG evaluation for benchmarks (tests only).

Benchmark expected answers are treated as semantic ground truth, not verbatim
templates. This module must never be imported by production generation code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

Rating = Literal["PASS", "PARTIAL", "FAIL"]


@dataclass(frozen=True)
class FactSpec:
    """One evaluable fact with paraphrase-tolerant match alternatives.

    A fact is present when ANY alternative succeeds. An alternative succeeds
    when ALL of its cue phrases appear in the normalized answer.
    """

    fact_id: str
    description: str
    alternatives: Tuple[Tuple[str, ...], ...]
    critical: bool = True


@dataclass(frozen=True)
class BenchmarkCase:
    test_id: str
    question: str
    expected_facts: Tuple[FactSpec, ...]
    required_sources: Tuple[str, ...]
    forbidden_facts: Tuple[FactSpec, ...] = ()
    optional_facts: Tuple[FactSpec, ...] = ()
    evaluation_notes: str = ""
    # Minimum distinct answer parts expected for compound questions (soft).
    compound_parts: int = 1


@dataclass
class EvalScores:
    factual_coverage: float
    source_grounding: float
    contradiction_score: float
    completeness: float
    citation_accuracy: float


@dataclass
class EvalResult:
    test_id: str
    rating: Rating
    generated_answer: str
    required_facts_found: List[str]
    required_facts_missing: List[str]
    optional_facts_found: List[str]
    forbidden_facts_detected: List[str]
    selected_source_files: List[str]
    citation_result: str
    scores: EvalScores
    critical_missing: List[str]
    noncritical_missing: List[str]
    evaluation_notes: str = ""

    def report(self) -> str:
        lines = [
            f"=== {self.test_id} → {self.rating} ===",
            f"Question parts / completeness: {self.scores.completeness:.0%}",
            f"Factual coverage: {self.scores.factual_coverage:.0%}",
            f"Source grounding: {self.scores.source_grounding:.0%}",
            f"Contradiction score: {self.scores.contradiction_score:.0%} (0% = clean)",
            f"Citation accuracy: {self.scores.citation_accuracy:.0%}",
            "",
            "Generated answer:",
            self.generated_answer.strip() or "(empty)",
            "",
            "Required facts found:",
            *([f"  - {item}" for item in self.required_facts_found] or ["  (none)"]),
            "Required facts missing:",
            *([f"  - {item}" for item in self.required_facts_missing] or ["  (none)"]),
            "Forbidden facts detected:",
            *(
                [f"  - {item}" for item in self.forbidden_facts_detected]
                or ["  (none)"]
            ),
            "Selected source files:",
            *([f"  - {item}" for item in self.selected_source_files] or ["  (none)"]),
            f"Citation result: {self.citation_result}",
        ]
        if self.evaluation_notes:
            lines.extend(["", f"Notes: {self.evaluation_notes}"])
        return "\n".join(lines)


_CITATION_RE = re.compile(r"\[\d+\]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_answer_text(text: str) -> str:
    """Normalize wording for paraphrase-tolerant cue matching."""
    value = (text or "").lower()
    value = _CITATION_RE.sub(" ", value)
    value = value.replace("–", "-").replace("—", "-")
    # Currency before stripping symbols: "¥350" / "350¥" → "350 yen"
    value = re.sub(r"[¥￥]\s*(\d+(?:[.,]\d+)?)", r"\1 yen", value)
    value = re.sub(r"(\d+(?:[.,]\d+)?)\s*[¥￥]", r"\1 yen", value)
    value = re.sub(r"(\d+(?:[.,]\d+)?)\s*(?:yen|円)", r"\1 yen", value)
    value = value.replace("&", " and ")
    value = _NON_ALNUM_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


def fact_present(answer: str, fact: FactSpec) -> bool:
    normalized = normalize_answer_text(answer)
    if not normalized:
        return False
    for alternative in fact.alternatives:
        cues = [normalize_answer_text(cue) for cue in alternative if cue.strip()]
        if cues and all(_cue_in_text(cue, normalized) for cue in cues):
            return True
    return False


def _cue_in_text(cue: str, normalized_answer: str) -> bool:
    if not cue:
        return True
    # Allow flexible spacing inside multi-word cues.
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in cue.split()) + r"\b"
    if re.search(pattern, normalized_answer):
        return True
    # Fallback: contiguous substring after space collapse (for "14 days").
    return cue in normalized_answer


def extract_source_names(
    sources: Sequence[Any] | None,
    *,
    answer: str = "",
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[str]:
    names: List[str] = []
    for source in sources or []:
        if isinstance(source, dict):
            name = source.get("document_name") or source.get("filename") or ""
        else:
            name = getattr(source, "document_name", "") or ""
        if name:
            names.append(str(name))
    if diagnostics:
        for key in (
            "evidence_sent_to_llm",
            "final_chunks_sent_to_llm",
        ):
            for item in diagnostics.get(key) or []:
                if isinstance(item, dict):
                    name = item.get("document_name") or item.get("filename") or ""
                    if name:
                        names.append(str(name))
        # Multi-question diagnostics nest evidence under sub_question_results.
        for block in diagnostics.get("sub_question_results") or []:
            if not isinstance(block, dict):
                continue
            for item in block.get("evidence_sent_to_llm") or []:
                if isinstance(item, dict):
                    name = item.get("document_name") or item.get("filename") or ""
                    if name:
                        names.append(str(name))
    # Deduplicate preserving order.
    seen = set()
    ordered: List[str] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    return ordered


def source_grounded(
    selected_sources: Sequence[str], required_sources: Sequence[str]
) -> bool:
    if not required_sources:
        return True
    blob = " | ".join(s.lower() for s in selected_sources)
    return any(req.lower() in blob for req in required_sources)


def citation_accuracy(
    *,
    answer: str,
    selected_sources: Sequence[str],
    required_sources: Sequence[str],
    sources: Sequence[Any] | None = None,
) -> Tuple[float, str]:
    """Score whether inline citations / source list align with required files."""
    has_inline = bool(re.search(r"\[\d+\]", answer or ""))
    grounded = source_grounded(selected_sources, required_sources)
    listed = bool(sources)
    if not required_sources:
        return 1.0, "no required sources"
    if grounded and (has_inline or listed):
        return 1.0, "citations grounded in required source"
    if grounded and not has_inline and not listed:
        return 0.7, "required source used but no citation markers"
    if not grounded and (has_inline or listed):
        return 0.2, "citations present but required source missing"
    return 0.0, "required source not selected"


def completeness_score(
    *,
    case: BenchmarkCase,
    found_fact_ids: Sequence[str],
) -> float:
    """Soft completeness: share of critical facts found (proxy for compound parts)."""
    critical = [fact for fact in case.expected_facts if fact.critical]
    if not critical:
        return 1.0
    found = {fid for fid in found_fact_ids}
    covered = sum(1 for fact in critical if fact.fact_id in found)
    ratio = covered / len(critical)
    if case.compound_parts <= 1:
        return ratio
    # Compound questions need broad critical coverage.
    return ratio


def rate_result(
    *,
    scores: EvalScores,
    critical_missing: Sequence[str],
    noncritical_missing: Sequence[str],
    forbidden_detected: Sequence[str],
    unsupported_fallback: bool,
) -> Rating:
    if unsupported_fallback or forbidden_detected or critical_missing:
        return "FAIL"
    if scores.source_grounding < 1.0:
        return "FAIL"
    if noncritical_missing or scores.citation_accuracy < 0.7:
        return "PARTIAL"
    if scores.completeness >= 0.999 and scores.contradiction_score <= 0.001:
        return "PASS"
    return "PARTIAL"


def evaluate_answer(
    case: BenchmarkCase,
    *,
    answer: str,
    sources: Sequence[Any] | None = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """Compare a generated answer to structured semantic ground truth."""
    found: List[str] = []
    missing: List[str] = []
    critical_missing: List[str] = []
    noncritical_missing: List[str] = []
    for fact in case.expected_facts:
        label = f"{fact.fact_id}: {fact.description}"
        if fact_present(answer, fact):
            found.append(label)
        else:
            missing.append(label)
            if fact.critical:
                critical_missing.append(label)
            else:
                noncritical_missing.append(label)

    optional_found = [
        f"{fact.fact_id}: {fact.description}"
        for fact in case.optional_facts
        if fact_present(answer, fact)
    ]

    forbidden_hit = [
        f"{fact.fact_id}: {fact.description}"
        for fact in case.forbidden_facts
        if fact_present(answer, fact)
    ]

    selected = extract_source_names(sources, answer=answer, diagnostics=diagnostics)
    grounded = source_grounded(selected, case.required_sources)
    cite_score, cite_msg = citation_accuracy(
        answer=answer,
        selected_sources=selected,
        required_sources=case.required_sources,
        sources=sources,
    )

    total_required = len(case.expected_facts) or 1
    factual_coverage = len(found) / total_required
    contradiction = 1.0 if forbidden_hit else 0.0
    completeness = completeness_score(case=case, found_fact_ids=[f.fact_id for f in case.expected_facts if fact_present(answer, f)])

    fallback = bool(
        re.search(
            r"(?i)could not find|i don'?t have|no information|unsupported",
            answer or "",
        )
    ) and factual_coverage < 0.5

    scores = EvalScores(
        factual_coverage=factual_coverage,
        source_grounding=1.0 if grounded else 0.0,
        contradiction_score=contradiction,
        completeness=completeness,
        citation_accuracy=cite_score,
    )
    rating = rate_result(
        scores=scores,
        critical_missing=critical_missing,
        noncritical_missing=noncritical_missing,
        forbidden_detected=forbidden_hit,
        unsupported_fallback=fallback,
    )
    return EvalResult(
        test_id=case.test_id,
        rating=rating,
        generated_answer=answer or "",
        required_facts_found=found,
        required_facts_missing=missing,
        optional_facts_found=optional_found,
        forbidden_facts_detected=forbidden_hit,
        selected_source_files=selected,
        citation_result=cite_msg,
        scores=scores,
        critical_missing=critical_missing,
        noncritical_missing=noncritical_missing,
        evaluation_notes=case.evaluation_notes,
    )


def assert_eval_rating(
    result: EvalResult,
    *,
    minimum: Rating = "PASS",
) -> None:
    """Pytest helper: fail with a full semantic report."""
    order = {"FAIL": 0, "PARTIAL": 1, "PASS": 2}
    if order[result.rating] < order[minimum]:
        raise AssertionError(
            f"Expected rating >= {minimum}, got {result.rating}\n\n{result.report()}"
        )


def fact(
    fact_id: str,
    description: str,
    *alternatives: Tuple[str, ...],
    critical: bool = True,
) -> FactSpec:
    """Convenience builder for FactSpec."""
    if not alternatives:
        raise ValueError(f"Fact {fact_id} needs at least one alternative cue tuple")
    return FactSpec(
        fact_id=fact_id,
        description=description,
        alternatives=tuple(alternatives),
        critical=critical,
    )
