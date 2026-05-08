"""DART OPEN API client.

Thin synchronous mirror of the DART OPEN API. This module does NOT
translate fiscal year to filing-date windows — `list_filings` takes raw
`bgn_de`/`end_de` strings on the DART acceptance date. Fiscal-year
mapping lives in `src.ingest.run`.

Endpoints used:
  - GET /api/corpCode.xml  → zip(CORPCODE.xml): corp_code ↔ corp_name master.
  - GET /api/list.json     → list of filings filtered by corp/period/type.
  - GET /api/document.xml  → zip of the raw report for a given rcept_no.

Auth: the DART API key (`crtfc_key`) is provided by the caller. In this
project it comes from Secret Manager via `src.common.config.get_secrets`.

Status field convention (DART returns HTTP 200 for everything):
  - "000" → normal
  - "013" → empty result (not an error; we return an empty list)
  - other → raised as `DartApiError`

Binary endpoints (corpCode/document) sometimes return a JSON error payload
with HTTP 200; we detect this via Content-Type and raise the same error.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Literal

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

ReportType = Literal["사업", "반기", "분기"]

_REPORT_DETAIL: dict[ReportType, str] = {
    "사업": "A001",
    "반기": "A002",
    "분기": "A003",
}

_BASE_URL = "https://opendart.fss.or.kr/api"
_DEFAULT_TIMEOUT = 30.0
_STATUS_OK = "000"
_STATUS_EMPTY = "013"


class DartApiError(RuntimeError):
    """DART API returned a non-OK status code."""

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(f"DART API status={status}: {message}")


class Filing(BaseModel):
    """One periodic-disclosure filing as returned by /list.json."""

    corp_code: str
    corp_name: str
    stock_code: str | None = None
    report_nm: str
    rcept_no: str
    flr_nm: str | None = None
    rcept_dt: str
    rm: str | None = None


class DartClient:
    """Synchronous client for the DART OPEN API.

    Usage:
        with DartClient(api_key) as dart:
            zip_bytes = dart.fetch_corp_code_zip()
            filings = dart.list_filings(
                corp_code="00126380", year=2024, report_type="사업"
            )
            doc_zip = dart.download_filing(filings[0].rcept_no)
    """

    def __init__(self, api_key: str, *, timeout: float = _DEFAULT_TIMEOUT):
        self._key = api_key
        self._client = httpx.Client(timeout=timeout, base_url=_BASE_URL)

    def __enter__(self) -> DartClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def _get(self, path: str, params: Mapping[str, str | int]) -> httpx.Response:
        full_params = {"crtfc_key": self._key, **params}
        t0 = time.perf_counter()
        resp = self._client.get(path, params=full_params)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        log.info(
            "dart_call",
            extra={
                "endpoint": path,
                "http_status": resp.status_code,
                "bytes": len(resp.content),
                "latency_ms": latency_ms,
            },
        )
        return resp

    def _check_status(self, data: dict[str, Any]) -> None:
        status = data.get("status")
        if status in (_STATUS_OK, _STATUS_EMPTY):
            return
        raise DartApiError(str(status), str(data.get("message", "")))

    def fetch_corp_code_zip(self) -> bytes:
        """Return the raw zip containing CORPCODE.xml.

        Caller parses with `zipfile` + an XML parser. This zip is small
        (≈4 MB) and changes rarely; cache it locally between runs.
        """
        resp = self._get("/corpCode.xml", {})
        if "application/json" in resp.headers.get("content-type", ""):
            self._check_status(resp.json())
        return resp.content

    def list_filings(
        self,
        *,
        corp_code: str,
        bgn_de: str,
        end_de: str,
        report_type: ReportType,
        page_count: int = 100,
    ) -> list[Filing]:
        """List filings for `corp_code` whose receipt date falls in [bgn_de, end_de].

        `bgn_de` and `end_de` are YYYYMMDD strings on the DART acceptance date,
        not on fiscal year. Returns an empty list if DART responds with
        status 013 (no results).
        """
        params: dict[str, str | int] = {
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "pblntf_detail_ty": _REPORT_DETAIL[report_type],
            "page_count": page_count,
        }
        resp = self._get("/list.json", params)
        data = resp.json()
        self._check_status(data)
        if data.get("status") == _STATUS_EMPTY:
            return []
        return [Filing.model_validate(item) for item in data.get("list", [])]

    def download_filing(self, rcept_no: str) -> bytes:
        """Return the raw zip archive for a given filing receipt number."""
        resp = self._get("/document.xml", {"rcept_no": rcept_no})
        if "application/json" in resp.headers.get("content-type", ""):
            self._check_status(resp.json())
        return resp.content
