from __future__ import annotations

from typing import Any, Dict, List

from app.retrieval.hybrid_search import tokenize


def simple_rerank(question: str, scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lightweight lexical reranker (no external cross-encoder dependency).

    Extension point: replace with a cross-encoder model later.
    """
    q_tokens = tokenize(question)
    for item in scored:
        content = str((item.get("payload") or {}).get("content") or "").lower()
        content_tokens = tokenize(content)
        overlap = len(q_tokens & content_tokens) / max(len(q_tokens), 1)
        item["rerank"] = float(item.get("combined", 0.0)) + overlap * 0.2
    return scored
