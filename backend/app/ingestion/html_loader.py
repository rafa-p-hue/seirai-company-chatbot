"""HTML file parser — semantic structure without chrome/boilerplate."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.formats import EmptyDocumentError
from app.models.api import ExtractedPage, ExtractedTableRow

DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
}
DROP_ATTR_RE = re.compile(
    r"(?i)(cookie|banner|consent|newsletter|sidebar|nav|menu|footer|header|promo)"
)


def extract_html_bytes(
    data: bytes, *, filename: str = "document.html", source_url: str | None = None
) -> Tuple[List[ExtractedPage], Dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    if not text.strip():
        raise EmptyDocumentError("HTML file is empty.")

    soup = BeautifulSoup(text, "lxml")
    for tag in list(soup.find_all(list(DROP_TAGS))):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        attrs = " ".join(
            [
                str(tag.get("id") or ""),
                " ".join(tag.get("class") or []),
                str(tag.get("role") or ""),
            ]
        )
        if DROP_ATTR_RE.search(attrs):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    root = soup.body or soup
    lines: List[str] = []
    table_rows: List[ExtractedTableRow] = []
    current_heading = title or None
    table_index = 0

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        name = node.name.lower()
        if name in {"h1", "h2", "h3", "h4"}:
            heading = _text(node)
            if heading:
                current_heading = heading
                lines.append(heading)
        elif name == "p":
            paragraph = _text(node)
            if paragraph:
                lines.append(paragraph)
        elif name in {"li"}:
            item = _text(node)
            if item:
                lines.append(f"• {item}")
        elif name == "table":
            extracted, human_lines = _table_to_rows(
                node, table_index=table_index, heading=current_heading
            )
            table_rows.extend(extracted)
            lines.extend(human_lines)
            table_index += 1

    # Prefer direct children walk if descendant walk produced nothing useful.
    if len("\n".join(lines).strip()) < 20:
        lines = [_text(root)]

    content = "\n".join(line for line in lines if line and line.strip())
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if len(content) < 12 and not table_rows:
        raise EmptyDocumentError("No useful text could be extracted from this HTML file.")

    page = ExtractedPage(
        page_number=1,
        text=content,
        table_rows=table_rows,
        section_heading=title or None,
        metadata={
            "title": title or None,
            "filename": filename,
            "source_url": source_url,
        },
    )
    return [page], {"page_count": 1, "title": title, "source_url": source_url}


def _text(node: Tag | NavigableString | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _table_to_rows(
    table: Tag, *, table_index: int, heading: str | None
) -> Tuple[List[ExtractedTableRow], List[str]]:
    rows_el = table.find_all("tr")
    if not rows_el:
        return [], []
    headers: List[str] = []
    header_cells = rows_el[0].find_all(["th", "td"])
    if header_cells:
        headers = [_text(cell) or f"Column {i+1}" for i, cell in enumerate(header_cells)]
    body = rows_el[1:] if headers and rows_el[0].find("th") is not None else rows_el
    if not headers and body:
        width = len(body[0].find_all(["td", "th"]))
        headers = [f"Column {i+1}" for i in range(width)]

    out: List[ExtractedTableRow] = []
    human: List[str] = []
    for row_index, row in enumerate(body, start=1):
        cells = [_text(cell) for cell in row.find_all(["td", "th"])]
        if not any(cells):
            continue
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[: len(headers)]
        parts = [f"{headers[i]}: {cells[i]}" for i in range(len(headers)) if cells[i]]
        human_text = "; ".join(parts)
        out.append(
            ExtractedTableRow(
                table_index=table_index,
                row_index=row_index,
                heading=heading,
                column_labels=headers,
                cells=cells,
                row_label=cells[0] if cells else None,
                human_text=human_text,
            )
        )
        human.append(human_text)
    return out, human
