"""Chunk data model — the contract between Stage 2 and Stage 3 (indexing)."""

from __future__ import annotations

from pydantic import BaseModel

from src.ingest.dart_client import ReportType


class Chunk(BaseModel):
    """One indexable unit of a DART report.

    `content` is either:
      - prose text (plus inlined small tables in markdown), when `is_table=False`,
      - a JSON-serialized 2D row matrix when `is_table=True` and the table fit
        as a standalone chunk,
      - a short pointer description when the table was offloaded to a sidecar
        (see `table_uri`).
    """

    chunk_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    fiscal_year: int
    report_type: ReportType
    section: str
    is_table: bool = False
    content: str
    table_uri: str | None = None
