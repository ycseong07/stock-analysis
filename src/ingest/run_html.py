"""Fetch all filings as rendered HTML from the DART web viewer and persist.

Per filing:
  1. dart_viewer.fetch_report(rcept_no)  →  one concatenated HTML document.
  2. Upload to gs://<bucket>/_html/<rcept_no>.html.
  3. Update the SQLite filings row's `html_uri` column.

Idempotent and resumable: by default re-running skips reports whose GCS
object already exists. Pass `force=True` to re-fetch.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from google.cloud.storage import Client as StorageClient

from src.common.config import get_infra
from src.ingest.dart_viewer import DartViewer
from src.ingest.store import open_meta_db, split_gcs_uri, upload_text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HtmlFetchStats:
    rcept_no: str
    bytes: int
    elapsed_ms: int
    skipped: bool = False


def _ensure_html_uri_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(filings)")}
    if "html_uri" not in cols:
        conn.execute("ALTER TABLE filings ADD COLUMN html_uri TEXT")


def _gcs_object_exists(uri: str, *, client: StorageClient) -> bool:
    bucket_name, key = split_gcs_uri(uri)
    return bool(client.bucket(bucket_name).blob(key).exists())


def fetch_one(
    *,
    bucket: str,
    rcept_no: str,
    viewer: DartViewer,
) -> HtmlFetchStats:
    """Fetch one report's HTML, upload to GCS. Returns stats (no SQLite write)."""
    t0 = time.perf_counter()
    html = viewer.fetch_report(rcept_no)
    uri = f"gs://{bucket}/_html/{rcept_no}.html"
    upload_text(uri, html, content_type="text/html; charset=utf-8")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return HtmlFetchStats(rcept_no=rcept_no, bytes=len(html), elapsed_ms=elapsed_ms)


def fetch_all(*, db_path: Path, force: bool = False) -> list[HtmlFetchStats]:
    """Iterate filings; fetch HTML, persist to GCS + SQLite.

    By default skips reports whose `gs://<bucket>/_html/<rcept_no>.html`
    already exists (resumable). Pass `force=True` to re-fetch all rows.
    """
    bucket = get_infra().raw_bucket
    storage_client = StorageClient()
    out: list[HtmlFetchStats] = []
    with open_meta_db(db_path) as conn, DartViewer() as viewer:
        _ensure_html_uri_column(conn)
        rows = list(conn.execute("SELECT rcept_no FROM filings ORDER BY rcept_no"))
        for row in rows:
            rcept_no = row["rcept_no"]
            uri = f"gs://{bucket}/_html/{rcept_no}.html"
            if not force and _gcs_object_exists(uri, client=storage_client):
                conn.execute(
                    "UPDATE filings SET html_uri = ? WHERE rcept_no = ?",
                    (uri, rcept_no),
                )
                out.append(HtmlFetchStats(rcept_no=rcept_no, bytes=0, elapsed_ms=0, skipped=True))
                log.info("html_skipped_exists", extra={"rcept_no": rcept_no})
                continue
            stats = fetch_one(bucket=bucket, rcept_no=rcept_no, viewer=viewer)
            conn.execute(
                "UPDATE filings SET html_uri = ? WHERE rcept_no = ?",
                (uri, rcept_no),
            )
            out.append(stats)
            log.info(
                "html_fetched",
                extra={
                    "rcept_no": rcept_no,
                    "bytes": stats.bytes,
                    "elapsed_ms": stats.elapsed_ms,
                },
            )
    return out
