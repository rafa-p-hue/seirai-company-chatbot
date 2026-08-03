"""Maximal marginal relevance and topic diversity for final evidence selection."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

from app.retrieval.chunk_quality import content_fingerprint
from app.retrieval.hybrid_search import tokenize


TOPIC_PATTERNS = {
    "identity": re.compile(
        r"(?i)\b(full\s*name|email|profile|phone|address|nominee)\b"
    ),
    "education": re.compile(
        r"(?i)\b(education|major|minor|degree|university|college|school|stud(?:y|ies))\b"
    ),
    "experience": re.compile(
        r"(?i)\b(intern|assistant|advisor|role|position|experience|responsib|"
        r"founder|chair|orientation|resident|manager|officer)\b"
    ),
    "research": re.compile(
        r"(?i)\b(research|lab|fellow|project|study|honors)\b"
    ),
    "affiliation": re.compile(
        r"(?i)\b(chapter|fraternity|sorority|organization|affiliation|society|club)\b"
    ),
    "interest": re.compile(
        r"(?i)\b(music|ensemble|band|performance|musician|choir|orchestra|hobby|interest)\b"
    ),
    "policy": re.compile(r"(?i)\b(policy|procedure|leave|guideline|terms)\b"),
    "product": re.compile(r"(?i)\b(feature|manual|calibrat|install|product|device)\b"),
    "registration": re.compile(
        r"(?i)\b(register|registration|moving in|required documents?|resident)\b"
    ),
    "fees": re.compile(r"(?i)\b(fee|fees|cost|price|certificate|¥|\$)\b"),
    "hours": re.compile(r"(?i)\b(office hours?|opening hours?|schedule)\b"),
    "overview": re.compile(
        r"(?i)\b(overview|this (?:guide|document|handbook)|purpose|introduction)\b"
    ),
}


def infer_topic(content: str, payload: Dict[str, Any] | None = None) -> str:
    text = content or ""
    record_type = str((payload or {}).get("record_type") or "")
    if record_type in TOPIC_PATTERNS and record_type not in {"universal", "section", "unknown"}:
        # Map structured types onto buckets.
        mapping = {
            "profile": "identity",
            "education": "education",
            "experience": "experience",
            "internship": "experience",
            "research": "research",
            "leadership": "experience",
            "skills": "interest",
        }
        if record_type in mapping:
            return mapping[record_type]
    for topic, pattern in TOPIC_PATTERNS.items():
        if pattern.search(text):
            return topic
    return "other"


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def mmr_select(
    scored: Sequence[Dict[str, Any]],
    *,
    limit: int,
    lambda_mult: float = 0.72,
    prefer_topic_diversity: bool = False,
) -> List[Dict[str, Any]]:
    """Select diverse high-scoring chunks via MMR (+ optional topic coverage)."""
    if not scored or limit <= 0:
        return []

    candidates = list(scored)
    selected: List[Dict[str, Any]] = []
    selected_fps: Set[str] = set()
    selected_topics: Set[str] = set()
    token_cache: Dict[int, Set[str]] = {}

    def tokens_for(item: Dict[str, Any]) -> Set[str]:
        key = id(item)
        if key not in token_cache:
            content = str(item["payload"].get("content") or "")
            token_cache[key] = tokenize(content)
        return token_cache[key]

    while candidates and len(selected) < limit:
        best_item = None
        best_score = float("-inf")
        for item in candidates:
            content = str(item["payload"].get("content") or "")
            fp = content_fingerprint(content)
            if not fp or fp in selected_fps:
                continue
            if any(fp in existing or existing in fp for existing in selected_fps):
                continue

            relevance = float(item.get("rerank", item.get("combined", 0.0)))
            if not selected:
                mmr = relevance
            else:
                redundancy = max(
                    jaccard(tokens_for(item), tokens_for(chosen)) for chosen in selected
                )
                mmr = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy

            topic = infer_topic(content, item.get("payload"))
            if prefer_topic_diversity and topic in selected_topics and topic != "other":
                mmr -= 0.18
            elif prefer_topic_diversity and topic not in selected_topics:
                mmr += 0.12

            if mmr > best_score:
                best_score = mmr
                best_item = item

        if best_item is None:
            break
        content = str(best_item["payload"].get("content") or "")
        selected.append(best_item)
        selected_fps.add(content_fingerprint(content))
        selected_topics.add(infer_topic(content, best_item.get("payload")))
        candidates = [item for item in candidates if item is not best_item]

    return selected
