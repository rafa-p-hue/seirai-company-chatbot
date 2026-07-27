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


def phrase_score(question: str, content: str) -> float:
    """Exact phrase / multi-word match boost."""
    q = re.sub(r"\s+", " ", question.lower()).strip()
    c = content.lower()
    if len(q) < 4:
        return 0.0
    if q in c:
        return 1.0
    # Quoted or multi-word spans
    words = [w for w in re.findall(r"[a-z0-9]+", q) if w not in STOPWORDS]
    if len(words) >= 2:
        bigram = " ".join(words[:2])
        if bigram in c:
            return 0.7
    return 0.0


def label_boost(query_type: str, content: str, payload: dict) -> float:
    """Boost labeled fields and headings relevant to the query intent."""
    text = content.lower()
    heading = str(payload.get("section_title") or "").lower()
    title = str(payload.get("title") or "").lower()
    score = 0.0
    if re.search(r"(?m)^[^:\n]{1,80}:\s*\S+", content):
        score += 0.2
    if heading:
        score += 0.05
    intent_terms = {
        "education": ("education", "school", "university", "college", "degree", "major", "minor"),
        "identity": ("name", "profile", "email", "title", "about"),
        "research": ("research", "lab", "project", "study", "fellow", "assistant"),
        "leadership": (
            "role",
            "position",
            "leadership",
            "activities",
            "involvement",
            "responsibility",
        ),
        "experience": (
            "experience",
            "employment",
            "role",
            "position",
            "job",
            "intern",
            "responsibility",
            "activities",
            "research",
            "leadership",
        ),
        "internship": ("intern", "internship", "role", "position"),
        "policy": (
            "policy",
            "procedure",
            "guideline",
            "terms",
            "refund",
            "cancellation",
            "membership",
            "volunteer",
            "compost",
            "rental",
            "accessibility",
        ),
        "price": ("price", "cost", "fee", "membership", "dues", "$"),
        "quantity": ("quantity", "pounds", "tons", "donation", "amount"),
        "date": ("date", "opening", "timeline", "schedule", "hours"),
        "accessibility": ("accessibility", "accessible", "disability", "ada", "ramp"),
        "product": ("product", "feature", "manual", "specification"),
    }
    # For experience intents, labeled contact fields should not get a boost.
    if query_type in {"experience", "internship", "research", "leadership"}:
        if re.search(
            r"(?i)\b(full\s*name|email|minor|phone|address)\b\s*:",
            content,
        ) and not re.search(r"(?i)\b(intern|assistant|advisor|fellow|chair)\b", content):
            return 0.0
    for term in intent_terms.get(query_type, ()):
        if term in text or term in heading or term in title:
            score += 0.08
    return min(1.0, score)
