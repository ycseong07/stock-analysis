"""Citation tag format used in agent answers.

Per CLAUDE.md citation contract:

    [corp:{corp_name}|year:{yyyy}|report:{사업|반기|분기}|section:{...}|chunk:{id}]

`chunk_id` is the load-bearing field — `faithfulness_check` re-fetches that
chunk and verifies cited claims appear in its content. The other fields are
human-readable redundancy (you can read the tag without dereferencing the id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieve.hybrid import RetrievedChunk

_TAG = re.compile(
    r"\[corp:(?P<corp>[^|\]]+)\|"
    r"year:(?P<year>\d{4})\|"
    r"report:(?P<report>[^|\]]+)\|"
    r"section:(?P<section>[^|\]]*)\|"
    r"chunk:(?P<chunk>[^\]]+)\]"
)


@dataclass(frozen=True)
class Citation:
    """Parsed view of one citation tag."""

    corp_name: str
    fiscal_year: int
    report_type: str
    section: str
    chunk_id: str


def format_citation(chunk: RetrievedChunk) -> str:
    """Render the citation tag for one retrieved chunk."""
    return (
        f"[corp:{chunk.corp_name}|year:{chunk.fiscal_year}|"
        f"report:{chunk.report_type}|section:{chunk.section}|"
        f"chunk:{chunk.chunk_id}]"
    )


def parse_citations(text: str) -> list[Citation]:
    """Extract every citation tag in `text`, in order of appearance."""
    return [
        Citation(
            corp_name=m["corp"].strip(),
            fiscal_year=int(m["year"]),
            report_type=m["report"].strip(),
            section=m["section"].strip(),
            chunk_id=m["chunk"].strip(),
        )
        for m in _TAG.finditer(text)
    ]
