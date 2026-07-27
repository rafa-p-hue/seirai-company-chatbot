"""Detect currency, dates, quantities, and related numeric answer types."""

from __future__ import annotations

import re
from typing import Optional


CURRENCY_RE = re.compile(
    r"(?i)(?:\$\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s*(?:usd|dollars?))"
)
PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%")
DATE_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|(?:spring|summer|fall|autumn|winter)\s+\d{4}"
    r"|\b(?:q[1-4]|early|late|mid)\s+\d{4}"
    r"|\b(?:20\d{2}|19\d{2})\b"
    r")"
)
TIME_RE = re.compile(
    r"(?i)\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"
)
QUANTITY_RE = re.compile(
    r"(?i)\b\d[\d,]*(?:\.\d+)?\s*(?:"
    r"pounds?|lbs?|tons?|kg|hours?|days?|weeks?|months?|years?|"
    r"people|members?|volunteers?|items?|units?|percent"
    r")\b|"
    r"\b(?:up to|at least|more than|over|under)\s+\d"
)
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][a-zA-Z0-9.'\- ]{2,40}\s+"
    r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|ct|court)\b|"
    r"\b(?:address|located at|location)\s*:",
    re.I,
)
UNANNOUNCED_RE = re.compile(
    r"(?i)\b("
    r"has not (?:yet )?been announced|not (?:yet )?announced|"
    r"not (?:yet )?determined|to be (?:announced|determined)|tba|tbd|"
    r"final address (?:is|has) not|"
    r"address (?:is|has) not (?:yet )?(?:been )?(?:announced|released|confirmed)"
    r")\b"
)
POLICY_RULE_RE = re.compile(
    r"(?i)\b("
    r"must|may not|cannot|can not|prohibited|not allowed|allowed|permitted|"
    r"required|forbidden|except|unless|policy|rule|shall|will not|"
    r"no\s+(?:pets|alcohol|refunds?)|refunds? (?:are|will)|cancel|"
    r"pre-approved|license"
    r")\b"
)
ACCESSIBILITY_RE = re.compile(
    r"(?i)\b("
    r"accessib(?:le|ility)|wheelchair|ada|ramp|elevator|disabilit(?:y|ies)|"
    r"assistive|barrier[- ]free|accessible\s+(?:entrance|parking|restroom|location)"
    r")\b"
)


def content_has_currency(text: str) -> bool:
    return bool(CURRENCY_RE.search(text or ""))


def content_has_date_or_period(text: str) -> bool:
    return bool(DATE_RE.search(text or "") or TIME_RE.search(text or ""))


def content_has_quantity(text: str) -> bool:
    return bool(QUANTITY_RE.search(text or "") or PERCENT_RE.search(text or ""))


def content_has_address_or_place(text: str) -> bool:
    return bool(
        ADDRESS_RE.search(text or "")
        or re.search(
            r"(?i)\b(located (?:at|in)|address|campus|facility|site|venue)\b",
            text or "",
        )
    )


def content_has_policy_rule(text: str) -> bool:
    return bool(POLICY_RULE_RE.search(text or ""))


def content_has_accessibility(text: str) -> bool:
    return bool(ACCESSIBILITY_RE.search(text or ""))


def content_states_unannounced(text: str) -> bool:
    return bool(UNANNOUNCED_RE.search(text or ""))


def numeric_answer_boost(question: str, content: str) -> float:
    """Boost chunks that contain the numeric/date shape the question needs."""
    q = (question or "").lower()
    text = content or ""
    boost = 0.0
    if re.search(r"\b(price|cost|fee|fees|how much|\$|pricing|dues)\b", q):
        if content_has_currency(text):
            boost += 0.55
    if re.search(r"\b(when|date|opening|period|year|month|schedule|hours|time)\b", q):
        if content_has_date_or_period(text):
            boost += 0.45
    if re.search(
        r"\b(how many|quantity|amount|pounds|tons|percent|%|number of)\b", q
    ):
        if content_has_quantity(text):
            boost += 0.5
    if re.search(r"\b(where|address|located|location)\b", q):
        if content_has_address_or_place(text) or content_states_unannounced(text):
            boost += 0.45
    return min(0.9, boost)


def detect_numeric_fact_type(question: str) -> Optional[str]:
    lower = (question or "").lower()
    if re.search(r"\b(price|cost|fee|fees|how much|pricing|dues|\$)\b", lower):
        return "price"
    if re.search(
        r"\b(how many|quantity|amount|pounds|tons|percent|number of)\b", lower
    ):
        return "quantity"
    if re.search(
        r"\b(when|what date|opening (?:date|period)|what year|timeline)\b", lower
    ):
        return "date"
    return None
