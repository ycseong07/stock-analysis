"""Build `Chunk` objects from a parsed list of `Section`s.

Strategy:

  - Walk sections left-to-right with a parent stack so each chunk records
    its full breadcrumb (e.g. ``I. 회사의 개요 > 1. 회사의 연혁``).
  - Inside each section, accumulate `<P>` text into a buffer; small
    `<TABLE>` is inlined as markdown into the buffer; medium/large tables
    flush the buffer first then produce their own chunk(s).
  - The text buffer is also flushed when adding the next paragraph would
    push it past `MAX_CHARS`. We do not merge across section boundaries —
    that would erase the breadcrumb.
  - Korean chars-per-token heuristic ≈ 1.7 (Gemini tokenizer); target chunk
    size 200–800 tokens → roughly 340–1 360 characters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from src.ingest.dart_client import ReportType
from src.parse.schemas import Chunk
from src.parse.sections import Section
from src.parse.tables import (
    classify_table,
    rows_to_json,
    rows_to_markdown,
    table_size,
    table_to_rows,
)

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 1.7
MIN_CHARS = int(200 * CHARS_PER_TOKEN)
MAX_CHARS = int(800 * CHARS_PER_TOKEN)


def _iter_parent_stacks(sections: list[Section]) -> Iterator[list[Section]]:
    """For each section in order, yield the stack of parents above it.

    `level == 0` (unknown-prefix headings like '목차') are treated as orphans:
    they get an empty stack and do not enter the stack themselves.
    """
    stack: list[Section] = []
    for s in sections:
        if s.level == 0:
            yield []
            continue
        while stack and stack[-1].level >= s.level:
            stack.pop()
        yield list(stack)
        stack.append(s)


def _split_oversized_text(text: str) -> list[str]:
    """Split text > MAX_CHARS at paragraph boundaries; one piece otherwise."""
    if len(text) <= MAX_CHARS:
        return [text]
    paragraphs = text.split("\n\n")
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) > MAX_CHARS and cur:
            out.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_report(
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
    """Build chunks for one report.

    `sidecar_uri_for` and `sidecar_payloads` together let the caller decide
    where large-table JSON goes. If both are None, sidecar tables get a
    placeholder URI (``sidecar://N``) and their payload is discarded.
    """
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
        section_chunks, sidecar_n = _chunk_section(
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


def _chunk_section(
    section: Section,
    *,
    path: str,
    rcept_no: str,
    base: dict[str, object],
    sidecar_uri_for: Callable[[int], str] | None,
    sidecar_payloads: dict[str, str] | None,
    sidecar_n_start: int,
) -> tuple[list[Chunk], int]:
    """Build chunks for one section. Returns `(chunks, next_sidecar_n)`."""
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
        if el.tag == "P":
            t = "".join(el.itertext()).strip()
            if not t:
                continue
            if buf_len() + len(t) > MAX_CHARS and text_buf:
                flush()
            text_buf.append(t)
        elif el.tag == "TABLE":
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
