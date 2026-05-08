"""Convert HTML `<table>` elements into markdown / JSON for chunking.

Same threshold policy as `src.parse.tables` (5-row inline / 50-row sidecar)
but operates on lowercase HTML tags emitted by the DART web viewer.
"""

from __future__ import annotations

import json
from typing import Literal

from lxml.html import HtmlElement

INLINE_MAX_ROWS = 5
SIDECAR_MIN_ROWS = 50

TableMode = Literal["inline", "chunk", "sidecar"]


def table_size(table: HtmlElement) -> int:
    return len(table.xpath(".//tr"))


def table_to_rows(table: HtmlElement) -> list[list[str]]:
    """Extract a row-major matrix of cell text from an HTML `<table>`."""
    rows: list[list[str]] = []
    for tr in table.xpath(".//tr"):
        cells: list[str] = []
        for cell in tr.xpath(".//td | .//th"):
            cells.append((cell.text_content() or "").strip())
        rows.append(cells)
    return rows


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    out: list[str] = []
    for i, r in enumerate(rows):
        padded = r + [""] * (width - len(r))
        out.append("| " + " | ".join(padded) + " |")
        if i == 0:
            out.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(out)


def rows_to_json(rows: list[list[str]]) -> str:
    return json.dumps(rows, ensure_ascii=False)


def classify_table(table: HtmlElement) -> TableMode:
    n = table_size(table)
    if n <= INLINE_MAX_ROWS:
        return "inline"
    if n < SIDECAR_MIN_ROWS:
        return "chunk"
    return "sidecar"
