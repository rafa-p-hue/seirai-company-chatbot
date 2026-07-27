"""Split multi-question user messages into ordered sub-questions."""

from __future__ import annotations

import re
from typing import List


def split_questions(message: str) -> List[str]:
    """Decompose a message into individual questions, preserving order.

    Handles:
    - multiple '?'-terminated questions in one line
    - newline-separated questions
    - mixed punctuation
    Returns a single-item list when the message is one question.
    """
    text = (message or "").strip()
    if not text:
        return []

    # Normalize fancy punctuation.
    text = text.replace("？", "?").replace("\r\n", "\n").replace("\r", "\n")

    candidates: List[str] = []
    # Prefer newline splits when multiple non-empty lines look like questions.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1 and sum(1 for line in lines if "?" in line or _looks_like_question(line)) >= 2:
        for line in lines:
            candidates.extend(_split_on_question_marks(line))
    else:
        candidates = _split_on_question_marks(text)

    cleaned: List[str] = []
    for item in candidates:
        item = re.sub(r"\s+", " ", item).strip(" \t-•")
        if not item:
            continue
        # Restore trailing ? for question-like clauses.
        if _looks_like_question(item) and not item.endswith("?"):
            item = item + "?"
        cleaned.append(item)

    return cleaned or [text]


def _split_on_question_marks(text: str) -> List[str]:
    # Keep the question mark with each clause by splitting after it.
    parts = re.split(r"(?<=\?)\s+", text.strip())
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) <= 1:
        # Also split "Q1? Q2" already handled; try "Q1. Q2?" patterns with
        # interrogatives on later sentences.
        sentences = re.split(
            r"(?<=[.!])\s+(?=(?:what|who|where|when|why|how|is|does|do|did|can|which)\b)",
            text,
            flags=re.I,
        )
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1 and sum(1 for s in sentences if _looks_like_question(s)) >= 2:
            return sentences
        # Compound one-liner: "What is X major and minor, what fraternity..., and which..."
        compound = re.split(
            r"\s*(?:,\s*|\band\s+)(?=(?:what|which|who|where|when|why|how)\b)",
            text,
            flags=re.I,
        )
        compound = [c.strip(" ,") for c in compound if c.strip(" ,")]
        if len(compound) > 1 and sum(1 for c in compound if _looks_like_question(c)) >= 2:
            return compound
        return [text.strip()]
    return parts


def _looks_like_question(text: str) -> bool:
    lower = text.lower().strip()
    if lower.endswith("?"):
        return True
    return bool(
        re.match(
            r"^(what|who|where|when|why|how|is|are|does|do|did|can|could|which|whose)\b",
            lower,
        )
    )
