"""Stage 2 (HTML edition): web-viewer HTML → chunks.jsonl per report.

Per filing:
  1. Download ``gs://<bucket>/_html/<rcept_no>.html``.
  2. Parse + sectionize (CSS class section-1/-2/-3) + chunkify.
  3. Upload chunks.jsonl to ``gs://<bucket>/_chunks/<rcept_no>.jsonl``.
  4. Upload large-table sidecars to ``gs://<bucket>/_tables/<rcept_no>/<n>.json``.
  5. EXTRACTION metadata is still derived from the original zip — the web
     viewer doesn't surface ACODE values, so we keep the zip path for that.

Idempotent: overwrites GCS objects and updates SQLite extractions_json.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.common.config import get_infra
from src.ingest.dart_client import ReportType
from src.ingest.store import download_zip, open_meta_db, upload_text
from src.parse.extractions import extract_metadata
from src.parse.html_chunker import chunk_html_report
from src.parse.html_loader import load_html
from src.parse.html_sections import split_html_into_sections
from src.parse.xml_loader import load_report_zip

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseStatsHtml:
    rcept_no: str
    n_sections: int
    n_chunks: int
    n_table_chunks: int
    n_sidecars: int
    elapsed_ms: int


def _ensure_extractions_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(filings)")}
    if "extractions_json" not in cols:
        conn.execute("ALTER TABLE filings ADD COLUMN extractions_json TEXT")


def parse_one(
    *,
    bucket: str,
    rcept_no: str,
    html_uri: str,
    zip_uri: str,
    corp_code: str,
    corp_name: str,
    fiscal_year: int,
    report_type: ReportType,
) -> tuple[ParseStatsHtml, dict[str, str]]:
    """Parse one report end-to-end. Returns `(stats, extractions)`."""
    t0 = time.perf_counter()

    # 1. HTML body → sections + chunks
    html_text = download_zip(html_uri).decode("utf-8")
    root = load_html(html_text)
    sections = split_html_into_sections(root)

    # 2. EXTRACTION metadata from original zip
    zip_bytes = download_zip(zip_uri)
    extractions = extract_metadata(load_report_zip(zip_bytes))

    # 3. Chunk
    sidecar_payloads: dict[str, str] = {}

    def sidecar_uri_for(n: int) -> str:
        return f"gs://{bucket}/_tables/{rcept_no}/{n}.json"

    chunks = chunk_html_report(
        sections,
        rcept_no=rcept_no,
        corp_code=corp_code,
        corp_name=corp_name,
        fiscal_year=fiscal_year,
        report_type=report_type,
        sidecar_uri_for=sidecar_uri_for,
        sidecar_payloads=sidecar_payloads,
    )

    # 4. Upload sidecars
    for uri, payload in sidecar_payloads.items():
        upload_text(uri, payload, content_type="application/json")

    # 5. Upload chunks.jsonl
    jsonl = "\n".join(c.model_dump_json(exclude_none=True) for c in chunks) + "\n"
    upload_text(
        f"gs://{bucket}/_chunks/{rcept_no}.jsonl",
        jsonl,
        content_type="application/x-ndjson",
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    stats = ParseStatsHtml(
        rcept_no=rcept_no,
        n_sections=len(sections),
        n_chunks=len(chunks),
        n_table_chunks=sum(1 for c in chunks if c.is_table),
        n_sidecars=len(sidecar_payloads),
        elapsed_ms=elapsed_ms,
    )
    log.info(
        "parsed_html",
        extra={
            "rcept_no": rcept_no,
            "sections": stats.n_sections,
            "chunks": stats.n_chunks,
            "table_chunks": stats.n_table_chunks,
            "sidecars": stats.n_sidecars,
            "elapsed_ms": elapsed_ms,
        },
    )
    return stats, extractions


def parse_all(*, db_path: Path) -> list[ParseStatsHtml]:
    """Iterate every row with a populated `html_uri`; produce chunks + meta."""
    bucket = get_infra().raw_bucket
    out: list[ParseStatsHtml] = []

    with open_meta_db(db_path) as conn:
        _ensure_extractions_column(conn)
        rows = list(
            conn.execute(
                "SELECT rcept_no, gcs_uri, html_uri, corp_code, corp_name, "
                "fiscal_year, report_type FROM filings "
                "WHERE html_uri IS NOT NULL "
                "ORDER BY rcept_no"
            )
        )
        for row in rows:
            stats, extractions = parse_one(
                bucket=bucket,
                rcept_no=row["rcept_no"],
                html_uri=row["html_uri"],
                zip_uri=row["gcs_uri"],
                corp_code=row["corp_code"],
                corp_name=row["corp_name"],
                fiscal_year=row["fiscal_year"],
                report_type=cast(ReportType, row["report_type"]),
            )
            conn.execute(
                "UPDATE filings SET extractions_json = ? WHERE rcept_no = ?",
                (json.dumps(extractions, ensure_ascii=False), row["rcept_no"]),
            )
            out.append(stats)

    log.info("parse_all_html_done", extra={"n": len(out)})
    return out
