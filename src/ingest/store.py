"""Persistent storage for the ingestion stage.

Two stores, both idempotent:
  - GCS: raw filing zips at gs://<bucket>/<corp_code>/<fiscal_year>/<report_type>/<rcept_no>.zip
  - SQLite: filing metadata (one row per rcept_no), local file `<db_path>`.

Module boundary: this layer knows nothing about DART or fiscal-year mapping.
The caller (src.ingest.run) constructs `FilingRecord` and `gcs_uri()` and
hands them to the store functions.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from google.cloud.storage import Client as StorageClient
from pydantic import BaseModel

from src.ingest.dart_client import ReportType

log = logging.getLogger(__name__)


class FilingRecord(BaseModel):
    """Metadata stored in SQLite for one ingested filing."""

    rcept_no: str
    corp_code: str
    corp_name: str
    stock_code: str | None
    fiscal_year: int
    report_type: ReportType
    report_nm: str
    rcept_dt: str
    gcs_uri: str
    bytes: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    rcept_no TEXT PRIMARY KEY,
    corp_code TEXT NOT NULL,
    corp_name TEXT NOT NULL,
    stock_code TEXT,
    fiscal_year INTEGER NOT NULL,
    report_type TEXT NOT NULL,
    report_nm TEXT NOT NULL,
    rcept_dt TEXT NOT NULL,
    gcs_uri TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_filings_corp_year
    ON filings(corp_code, fiscal_year, report_type);
"""


def gcs_uri(
    bucket: str,
    corp_code: str,
    fiscal_year: int,
    report_type: ReportType,
    rcept_no: str,
) -> str:
    return f"gs://{bucket}/{corp_code}/{fiscal_year}/{report_type}/{rcept_no}.zip"


def split_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a `gs://bucket/key` URI into `(bucket, key)`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri}")
    rest = uri[len("gs://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed gs:// URI: {uri}")
    return bucket, key


def upload_zip(
    uri: str,
    payload: bytes,
    *,
    client: StorageClient | None = None,
) -> None:
    """Upload `payload` to the GCS object at `uri`. Overwrites existing object."""
    storage_client = client or StorageClient()
    bucket_name, key = split_gcs_uri(uri)
    blob = storage_client.bucket(bucket_name).blob(key)
    blob.upload_from_string(payload, content_type="application/zip")
    log.info("gcs_upload", extra={"uri": uri, "bytes": len(payload)})


def download_zip(
    uri: str,
    *,
    client: StorageClient | None = None,
) -> bytes:
    """Download a GCS object as bytes."""
    storage_client = client or StorageClient()
    bucket_name, key = split_gcs_uri(uri)
    data: bytes = storage_client.bucket(bucket_name).blob(key).download_as_bytes()
    log.info("gcs_download", extra={"uri": uri, "bytes": len(data)})
    return data


def upload_text(
    uri: str,
    payload: str,
    *,
    content_type: str = "application/json",
    client: StorageClient | None = None,
) -> None:
    """Upload a UTF-8 text payload to a GCS object. Overwrites."""
    storage_client = client or StorageClient()
    bucket_name, key = split_gcs_uri(uri)
    blob = storage_client.bucket(bucket_name).blob(key)
    blob.upload_from_string(payload.encode("utf-8"), content_type=content_type)
    log.info("gcs_upload_text", extra={"uri": uri, "bytes": len(payload)})


@contextmanager
def open_meta_db(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the metadata SQLite DB and ensure schema exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_filing(conn: sqlite3.Connection, record: FilingRecord) -> None:
    """Insert or update a filing metadata row, keyed by `rcept_no`."""
    conn.execute(
        """
        INSERT INTO filings (
            rcept_no, corp_code, corp_name, stock_code, fiscal_year,
            report_type, report_nm, rcept_dt, gcs_uri, bytes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(rcept_no) DO UPDATE SET
            corp_code=excluded.corp_code,
            corp_name=excluded.corp_name,
            stock_code=excluded.stock_code,
            fiscal_year=excluded.fiscal_year,
            report_type=excluded.report_type,
            report_nm=excluded.report_nm,
            rcept_dt=excluded.rcept_dt,
            gcs_uri=excluded.gcs_uri,
            bytes=excluded.bytes
        """,
        (
            record.rcept_no,
            record.corp_code,
            record.corp_name,
            record.stock_code,
            record.fiscal_year,
            record.report_type,
            record.report_nm,
            record.rcept_dt,
            record.gcs_uri,
            record.bytes,
        ),
    )
