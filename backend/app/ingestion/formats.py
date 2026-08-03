"""Supported upload formats and file-type detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from app.models.api import SourceType


@dataclass(frozen=True)
class FormatSpec:
    extension: str
    file_type: str
    source_type: SourceType
    mime_types: Tuple[str, ...]


SUPPORTED_FORMATS: Tuple[FormatSpec, ...] = (
    FormatSpec(".pdf", "pdf", SourceType.pdf, ("application/pdf",)),
    FormatSpec(
        ".html",
        "html",
        SourceType.html,
        ("text/html", "application/xhtml+xml"),
    ),
    FormatSpec(
        ".htm",
        "html",
        SourceType.html,
        ("text/html", "application/xhtml+xml"),
    ),
    FormatSpec(
        ".docx",
        "docx",
        SourceType.docx,
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    ),
    FormatSpec(".md", "markdown", SourceType.markdown, ("text/markdown", "text/x-markdown")),
    FormatSpec(".markdown", "markdown", SourceType.markdown, ("text/markdown",)),
    FormatSpec(".csv", "csv", SourceType.csv, ("text/csv", "application/csv")),
    FormatSpec(
        ".pptx",
        "pptx",
        SourceType.pptx,
        (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ),
)

SUPPORTED_EXTENSIONS: Set[str] = {spec.extension for spec in SUPPORTED_FORMATS}
SUPPORTED_MIME_TYPES: Set[str] = {
    mime for spec in SUPPORTED_FORMATS for mime in spec.mime_types
}
ACCEPT_ATTRIBUTE = ",".join(
    sorted(
        {
            *SUPPORTED_EXTENSIONS,
            *SUPPORTED_MIME_TYPES,
        }
    )
)


class UnsupportedFormatError(ValueError):
    """Raised when an upload extension/MIME is not supported."""


class EmptyDocumentError(ValueError):
    """Raised when a file has no usable text."""


class PasswordProtectedError(ValueError):
    """Raised when a document cannot be opened without a password."""


def extension_of(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def detect_format(
    filename: str, data: bytes = b"", content_type: Optional[str] = None
) -> FormatSpec:
    """Resolve a supported format from filename, MIME, or magic bytes."""
    ext = extension_of(filename)
    by_ext = {spec.extension: spec for spec in SUPPORTED_FORMATS}
    if ext in by_ext:
        return by_ext[ext]

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    for spec in SUPPORTED_FORMATS:
        if mime and mime in spec.mime_types:
            return spec

    if data.startswith(b"%PDF"):
        return by_ext[".pdf"]
    head = data[:2048].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return by_ext[".html"]
    if data[:2] == b"PK":
        # OOXML packages — require a known extension.
        pass

    raise UnsupportedFormatError(
        "Unsupported file format. Supported formats: PDF, DOCX, HTML, MD, CSV, PPTX."
    )


def source_type_for(file_type: str) -> SourceType:
    mapping: Dict[str, SourceType] = {
        "pdf": SourceType.pdf,
        "html": SourceType.html,
        "docx": SourceType.docx,
        "markdown": SourceType.markdown,
        "csv": SourceType.csv,
        "pptx": SourceType.pptx,
        "website": SourceType.website,
    }
    return mapping.get(file_type, SourceType.pdf)
