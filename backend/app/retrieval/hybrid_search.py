from __future__ import annotations

import re
from typing import Set


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "what",
    "how",
    "does",
    "do",
    "of",
    "to",
    "and",
    "in",
    "for",
    "with",
    "about",
    "that",
    "this",
    "it",
    "as",
    "on",
    "by",
    "from",
}


def tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def hybrid_score(question: str, content: str) -> float:
    q_tokens = tokenize(question)
    c_tokens = tokenize(content)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return min(1.0, overlap / max(len(q_tokens), 1))
