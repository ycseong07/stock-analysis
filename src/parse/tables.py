"""Convert `<TABLE>` elements into text or JSON for downstream chunking.

Threshold policy:

  - rows ≤ INLINE_MAX_ROWS (5)         → markdown text, inlined into the
    surrounding section's text-run chunk.
  - INLINE_MAX_ROWS < rows < SIDECAR_MIN_ROWS  → standalone `is_table` chunk;
    its content is a JSON-serialized 2D row matrix.
  - rows ≥ SIDECAR_MIN_ROWS (50)       → JSON sidecar uploaded to a separate
    GCS object; the chunk records only `table_uri`.

These thresholds are heuristics tuned for DART periodic filings — a
typical financial statement spans 20–60 rows; very long tables are usually
계열사 목록 / 보유 자산 명세 (1 000+ rows) and don't belong inside a chunk.
"""

from __future__ import annotations

import json
from typing import Literal

from lxml import etree

INLINE_MAX_ROWS = 5
SIDECAR_MIN_ROWS = 50

TableMode = Literal["inline", "chunk", "sidecar"]


def table_size(table: etree._Element) -> int:
    return len(list(table.iter("TR")))


def table_to_rows(table: etree._Element) -> list[list[str]]:
    """Extract a row-major matrix of cell text from a `<TABLE>`."""
    rows: list[list[str]] = []
    for tr in table.iter("TR"):
        cells: list[str] = []
        for cell in tr.iter():
            if cell.tag in ("TD", "TH", "TU"):
                cells.append("".join(cell.itertext()).strip())
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


def classify_table(table: etree._Element) -> TableMode:
    n = table_size(table)
    if n <= INLINE_MAX_ROWS:
        return "inline"
    if n < SIDECAR_MIN_ROWS:
        return "chunk"
    return "sidecar"
