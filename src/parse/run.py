"""Stage 2 orchestrator: parse every GCS report → chunks.jsonl per report.

For each row in the SQLite `filings` table:
  1. Download the report zip from GCS.
  2. Parse + section + chunk.
  3. Upload chunks.jsonl to ``gs://<bucket>/_chunks/{rcept_no}.jsonl``.
  4. Upload large-table sidecars to ``gs://<bucket>/_tables/{rcept_no}/{n}.json``.
  5. Update the SQLite row with ``extractions_json`` (the 20 ACODE values).

Idempotent: re-running overwrites GCS objects and updates SQLite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.common.config import get_infra
from src.ingest.dart_client import ReportType
from src.ingest.store import download_zip, open_meta_db, upload_text
from src.parse.chunker import chunk_report
from src.parse.extractions import extract_metadata
from src.parse.sections import split_into_sections
from src.parse.xml_loader import load_report_zip

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseStats:
    rcept_no: str
    n_sections: int
    n_chunks: int
    n_table_chunks: int
    n_sidecars: int


def _ensure_extractions_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(filings)")}
    if "extractions_json" not in cols:
        conn.execute("ALTER TABLE filings ADD COLUMN extractions_json TEXT")


def parse_one(
    *,
    bucket: str,
    rcept_no: str,
    gcs_uri: str,
    corp_code: str,
    corp_name: str,
    fiscal_year: int,
    report_type: ReportType,
) -> tuple[ParseStats, dict[str, str]]:
    """Parse one report end-to-end. Returns `(stats, extractions)`."""
    report = load_report_zip(download_zip(gcs_uri))
    sections = split_into_sections(report.main)
    extractions = extract_metadata(report)

    sidecar_payloads: dict[str, str] = {}

    def sidecar_uri_for(n: int) -> str:
        return f"gs://{bucket}/_tables/{rcept_no}/{n}.json"

    chunks = chunk_report(
        sections,
        rcept_no=rcept_no,
        corp_code=corp_code,
        corp_name=corp_name,
        fiscal_year=fiscal_year,
        report_type=report_type,
        sidecar_uri_for=sidecar_uri_for,
        sidecar_payloads=sidecar_payloads,
    )

    for uri, payload in sidecar_payloads.items():
        upload_text(uri, payload, content_type="application/json")

    jsonl = "\n".join(c.model_dump_json(exclude_none=True) for c in chunks) + "\n"
    upload_text(
        f"gs://{bucket}/_chunks/{rcept_no}.jsonl",
        jsonl,
        content_type="application/x-ndjson",
    )

    stats = ParseStats(
        rcept_no=rcept_no,
        n_sections=len(sections),
        n_chunks=len(chunks),
        n_table_chunks=sum(1 for c in chunks if c.is_table),
        n_sidecars=len(sidecar_payloads),
    )
    log.info(
        "parsed",
        extra={
            "rcept_no": rcept_no,
            "sections": stats.n_sections,
            "chunks": stats.n_chunks,
            "table_chunks": stats.n_table_chunks,
            "sidecars": stats.n_sidecars,
        },
    )
    return stats, extractions


def parse_all(*, db_path: Path) -> list[ParseStats]:
    """Iterate every row in `filings`, parse to chunks.jsonl, update SQLite."""
    bucket = get_infra().raw_bucket
    all_stats: list[ParseStats] = []

    with open_meta_db(db_path) as conn:
        _ensure_extractions_column(conn)
        rows = list(
            conn.execute(
                "SELECT rcept_no, gcs_uri, corp_code, corp_name, "
                "fiscal_year, report_type FROM filings"
            )
        )
        for row in rows:
            stats, extractions = parse_one(
                bucket=bucket,
                rcept_no=row["rcept_no"],
                gcs_uri=row["gcs_uri"],
                corp_code=row["corp_code"],
                corp_name=row["corp_name"],
                fiscal_year=row["fiscal_year"],
                report_type=cast(ReportType, row["report_type"]),
            )
            conn.execute(
                "UPDATE filings SET extractions_json = ? WHERE rcept_no = ?",
                (json.dumps(extractions, ensure_ascii=False), row["rcept_no"]),
            )
            all_stats.append(stats)

    log.info("parse_all_done", extra={"n": len(all_stats)})
    return all_stats
