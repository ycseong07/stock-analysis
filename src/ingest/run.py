"""Stage 1 orchestrator: matrix run over (corp × fiscal_year × report_type).

For each cell:
  1. Resolve corp_name → corp_code via the DART master.
  2. Map (fiscal_year, report_type) → DART acceptance-date window.
  3. List filings, drop amendments (정정), pick the latest.
  4. Download the zip from DART, upload to GCS, upsert metadata row.

Idempotent end to end: re-running with the same inputs overwrites the
GCS object and updates the SQLite row (keyed by rcept_no).

Fiscal-year → filing-date window (Korean disclosure rules; widened to
absorb late filings):
  - 사업 (annual):     FY YYYY → YYYY+1/01/01 ~ YYYY+1/06/30
  - 반기 (semi-annual): FY YYYY → YYYY/06/01    ~ YYYY/10/31
  - 분기 (quarterly):   FY YYYY → YYYY/04/01    ~ YYYY+1/01/31

This is the "fiscal-year mapping" that ADR-0001 deliberately keeps
out of `dart_client`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.common.config import get_infra, get_secrets
from src.ingest.corp_codes import find_listed_corp, load_corp_codes
from src.ingest.dart_client import DartClient, Filing, ReportType
from src.ingest.store import (
    FilingRecord,
    gcs_uri,
    open_meta_db,
    upload_zip,
    upsert_filing,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestRequest:
    corp_names: list[str]
    fiscal_years: list[int]
    report_types: list[ReportType]


def filing_window(fiscal_year: int, report_type: ReportType) -> tuple[str, str]:
    """Return (bgn_de, end_de) covering the filing window for this FY / report type."""
    if report_type == "사업":
        return f"{fiscal_year + 1}0101", f"{fiscal_year + 1}0630"
    if report_type == "반기":
        return f"{fiscal_year}0601", f"{fiscal_year}1031"
    if report_type == "분기":
        return f"{fiscal_year}0401", f"{fiscal_year + 1}0131"
    raise ValueError(f"unknown report_type: {report_type}")


def pick_latest_non_amended(filings: list[Filing]) -> Filing | None:
    """Prefer non-정정 filings; tiebreak by latest rcept_dt."""
    if not filings:
        return None
    non_amended = [f for f in filings if "정정" not in f.report_nm]
    pool = non_amended or filings
    return max(pool, key=lambda f: f.rcept_dt)


def ingest(request: IngestRequest, *, db_path: Path) -> int:
    """Run ingestion across the request matrix. Returns count of uploads."""
    infra = get_infra()
    api_key = get_secrets().dart_api_key.get_secret_value()
    bucket = infra.raw_bucket

    count = 0
    with DartClient(api_key) as dart:
        corps = load_corp_codes(dart)
        targets = [find_listed_corp(corps, n) for n in request.corp_names]
        log.info("ingest_resolved_corps", extra={"n": len(targets)})

        with open_meta_db(db_path) as conn:
            for corp in targets:
                for fy in request.fiscal_years:
                    for rt in request.report_types:
                        bgn, end = filing_window(fy, rt)
                        listings = dart.list_filings(
                            corp_code=corp.corp_code,
                            bgn_de=bgn,
                            end_de=end,
                            report_type=rt,
                        )
                        chosen = pick_latest_non_amended(listings)
                        if chosen is None:
                            log.info(
                                "ingest_miss",
                                extra={"corp": corp.corp_name, "fy": fy, "type": rt},
                            )
                            continue

                        zip_bytes = dart.download_filing(chosen.rcept_no)
                        uri = gcs_uri(bucket, corp.corp_code, fy, rt, chosen.rcept_no)
                        upload_zip(uri, zip_bytes)

                        record = FilingRecord(
                            rcept_no=chosen.rcept_no,
                            corp_code=corp.corp_code,
                            corp_name=corp.corp_name,
                            stock_code=corp.stock_code,
                            fiscal_year=fy,
                            report_type=rt,
                            report_nm=chosen.report_nm,
                            rcept_dt=chosen.rcept_dt,
                            gcs_uri=uri,
                            bytes=len(zip_bytes),
                        )
                        upsert_filing(conn, record)
                        count += 1
                        log.info(
                            "ingest_ok",
                            extra={
                                "corp": corp.corp_name,
                                "fy": fy,
                                "type": rt,
                                "rcept_no": chosen.rcept_no,
                                "bytes": len(zip_bytes),
                            },
                        )
    log.info("ingest_done", extra={"count": count})
    return count
