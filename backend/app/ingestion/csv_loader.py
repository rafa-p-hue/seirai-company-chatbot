"""CSV parser — one semantic record per logical row."""

from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Tuple

from app.ingestion.formats import EmptyDocumentError
from app.models.api import ExtractedPage, ExtractedTableRow

META_COLUMNS = {
    "effective_date": "effective_date",
    "effective": "effective_date",
    "status": "status",
    "document_status": "status",
    "version": "version",
    "archived": "status",
    "valid_from": "valid_from",
    "valid_to": "valid_to",
}


def extract_csv_bytes(
    data: bytes, *, filename: str = "document.csv"
) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
    if not data.strip():
        raise EmptyDocumentError("CSV file is empty.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise EmptyDocumentError("CSV file has no header row.")

    headers = [str(h).strip() for h in reader.fieldnames if h is not None and str(h).strip()]
    if not headers:
        raise EmptyDocumentError("CSV file has no usable column names.")

    pages: List[ExtractedPage] = []
    doc_meta: Dict[str, Any] = {"filename": filename, "columns": headers}
    for row_number, raw in enumerate(reader, start=1):
        values = {
            key: (raw.get(key) or "").strip()
            for key in headers
            if (raw.get(key) or "").strip()
        }
        if not values:
            continue
        parts = [f"{key}: {values[key]}" for key in headers if key in values]
        human = "; ".join(parts)
        row_meta = _row_metadata(values)
        for key, value in row_meta.items():
            doc_meta.setdefault(key, value)
        table_row = ExtractedTableRow(
            table_index=0,
            row_index=row_number,
            heading=None,
            effective_date=row_meta.get("effective_date"),
            column_labels=headers,
            cells=[values.get(key, "") for key in headers],
            row_label=next(iter(values.values()), None),
            human_text=human,
        )
        pages.append(
            ExtractedPage(
                page_number=row_number,
                text=human,
                table_rows=[table_row],
                row_number=row_number,
                metadata={
                    "filename": filename,
                    "columns": headers,
                    **row_meta,
                },
            )
        )

    if not pages:
        raise EmptyDocumentError("CSV file contains no data rows.")
    return pages, {
        "page_count": len(pages),
        "row_count": len(pages),
        "columns": headers,
        **{k: v for k, v in doc_meta.items() if k in META_COLUMNS.values()},
    }


def _row_metadata(values: Dict[str, str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key, value in values.items():
        normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
        mapped = META_COLUMNS.get(normalized)
        if not mapped:
            continue
        if mapped == "status" and normalized == "archived":
            meta["status"] = (
                "archived"
                if value.lower() in {"1", "true", "yes", "archived", "y"}
                else value
            )
        else:
            meta[mapped] = value
    return meta
