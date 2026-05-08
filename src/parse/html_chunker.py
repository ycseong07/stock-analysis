"""Build `Chunk`s from HTML-sourced `Section`s.

Reuses `_iter_parent_stacks` and `_split_oversized_text` from the legacy
XML chunker; only the per-section element walk differs (lowercase HTML
tags `p`/`table` instead of upper-case DART XML `P`/`TABLE`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.ingest.dart_client import ReportType
from src.parse.chunker import (
    MAX_CHARS,
    _iter_parent_stacks,
    _split_oversized_text,
)
from src.parse.html_tables import (
    classify_table,
    rows_to_json,
    rows_to_markdown,
    table_size,
    table_to_rows,
)
from src.parse.schemas import Chunk
from src.parse.sections import Section

log = logging.getLogger(__name__)


def chunk_html_report(
    sections: list[Section],
    *,
    rcept_no: str,
    corp_code: str,
    corp_name: str,
    fiscal_year: int,
    report_type: ReportType,
    sidecar_uri_for: Callable[[int], str] | None = None,
    sidecar_payloads: dict[str, str] | None = None,
) -> list[Chunk]:
    """Build chunks for one HTML-sourced report."""
    base = {
        "rcept_no": rcept_no,
        "corp_code": corp_code,
        "corp_name": corp_name,
        "fiscal_year": fiscal_year,
        "report_type": report_type,
    }
    chunks: list[Chunk] = []
    sidecar_n = 0

    for parents, section in zip(_iter_parent_stacks(sections), sections, strict=True):
        path = " > ".join(s.title for s in [*parents, section])
        section_chunks, sidecar_n = _chunk_html_section(
            section,
            path=path,
            rcept_no=rcept_no,
            base=base,
            sidecar_uri_for=sidecar_uri_for,
            sidecar_payloads=sidecar_payloads,
            sidecar_n_start=sidecar_n,
        )
        chunks.extend(section_chunks)
    return chunks


def _chunk_html_section(
    section: Section,
    *,
    path: str,
    rcept_no: str,
    base: dict[str, object],
    sidecar_uri_for: Callable[[int], str] | None,
    sidecar_payloads: dict[str, str] | None,
    sidecar_n_start: int,
) -> tuple[list[Chunk], int]:
    chunks: list[Chunk] = []
    text_buf: list[str] = []
    idx = 0
    sidecar_n = sidecar_n_start

    def buf_len() -> int:
        return sum(len(s) for s in text_buf)

    def flush() -> None:
        nonlocal idx
        if not text_buf:
            return
        for piece in _split_oversized_text("\n\n".join(text_buf)):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{rcept_no}#{path}#{idx}",
                    section=path,
                    is_table=False,
                    content=piece,
                    **base,  # type: ignore[arg-type]
                )
            )
            idx += 1
        text_buf.clear()

    for el in section.elements:
        if el.tag == "p":
            t = (el.text_content() or "").strip()
            if not t:
                continue
            if buf_len() + len(t) > MAX_CHARS and text_buf:
                flush()
            text_buf.append(t)
        elif el.tag == "table":
            rows = table_to_rows(el)
            mode = classify_table(el)
            if mode == "inline":
                md = rows_to_markdown(rows)
                if not md:
                    continue
                if buf_len() + len(md) > MAX_CHARS and text_buf:
                    flush()
                text_buf.append(md)
            else:
                flush()
                if mode == "chunk":
                    chunks.append(
                        Chunk(
                            chunk_id=f"{rcept_no}#{path}#{idx}",
                            section=path,
                            is_table=True,
                            content=rows_to_json(rows),
                            **base,  # type: ignore[arg-type]
                        )
                    )
                    idx += 1
                else:  # sidecar
                    sidecar_n += 1
                    uri = (
                        sidecar_uri_for(sidecar_n) if sidecar_uri_for else f"sidecar://{sidecar_n}"
                    )
                    if sidecar_payloads is not None:
                        sidecar_payloads[uri] = rows_to_json(rows)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{rcept_no}#{path}#{idx}",
                            section=path,
                            is_table=True,
                            content=f"[표 — {table_size(el)}행, sidecar 참조]",
                            table_uri=uri,
                            **base,  # type: ignore[arg-type]
                        )
                    )
                    idx += 1

    flush()
    return chunks, sidecar_n
