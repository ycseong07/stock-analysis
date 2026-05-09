"""DART loader — disclosures_market (공시 이벤트) + financials (재무 지표).

Uses ``OpenDartReader`` for both endpoints:
  - ``dart.list(corp, start, end)``    → recent filings → ``disclosures_market``
  - ``dart.finstate(corp, year, ...)`` → quarterly financials → ``financials``

M1 financials scope (decision 2026-05-09):
  - raw     : total_assets / total_liabilities / total_equity /
              revenue / operating_income / net_income
  - derived : debt_ratio (= total_liabilities / total_equity),
              roe        (= net_income / total_equity)
  - deferred (to M3): EPS / PER / PBR — need market cap + shares-outstanding
                      join, computed in disclosure_and_financial_events node.

Filings are categorised via report_nm keyword matching.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import OpenDartReader
import pandas as pd

from src.common.config import get_secrets
from src.research.ingest.schemas import FRESHNESS_REPORT, FRESHNESS_T0

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

_dart_client: Any = None  # OpenDartReader has no type stubs


def _get_dart() -> Any:
    global _dart_client
    if _dart_client is None:
        _dart_client = OpenDartReader(get_secrets().dart_api_key.get_secret_value())
    return _dart_client


# --- disclosures_market ----------------------------------------------------

# Order matters: first match wins.
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("사업보고서", "annual_report"),
    ("반기보고서", "semi_annual_report"),
    ("분기보고서", "quarterly_report"),
    ("유상증자", "rights_issue"),
    ("전환사채", "convertible_bond"),
    ("신주인수권부사채", "warrant_bond"),
    ("교환사채", "exchangeable_bond"),
    ("감사", "audit"),
    ("주요사항보고서", "material_event"),
    ("자기주식", "treasury_stock"),
    ("배당", "dividend"),
    ("실적", "earnings_announcement"),
]


def _classify_filing(report_nm: str) -> str | None:
    for kw, cat in _CATEGORY_KEYWORDS:
        if kw in report_nm:
            return cat
    return None


def load_disclosures(stock_code: str, date_range: tuple[date, date]) -> list[dict[str, Any]]:
    """Recent DART filings for one stock × range → ``disclosures_market`` rows."""
    start, end = date_range
    dart = _get_dart()
    df = dart.list(stock_code, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return []

    now_iso = datetime.now(KST).isoformat()
    rows: list[dict[str, Any]] = []
    for _, item in df.iterrows():
        rcept_dt_str = str(item["rcept_dt"])
        try:
            rcept_dt = datetime.strptime(rcept_dt_str, "%Y%m%d").date()
        except ValueError:
            log.warning(
                "skipping disclosure with bad rcept_dt",
                extra={"stock_code": stock_code, "rcept_dt": rcept_dt_str},
            )
            continue
        report_nm = str(item["report_nm"])
        rows.append(
            {
                "stock_code": stock_code,
                "rcept_no": str(item["rcept_no"]),
                "rcept_dt": rcept_dt.isoformat(),
                "report_type": report_nm,
                "title": report_nm,
                "category": _classify_filing(report_nm),
                "as_of": now_iso,
                "data_freshness": FRESHNESS_T0,
            }
        )
    return rows


# --- financials ------------------------------------------------------------

# DART account_nm → BQ metric. Some reports use "당기순이익", some "당기순이익(손실)".
_ACCOUNT_TO_METRIC: dict[str, str] = {
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "total_equity",
    "매출액": "revenue",
    "영업이익": "operating_income",
    "당기순이익(손실)": "net_income",
    "당기순이익": "net_income",
}

# DART reprt_code → fiscal_quarter (1=Q1, 2=H1 cumulative, 3=Q3, 4=annual)
_REPRT_CODE_TO_QUARTER: dict[str, int] = {
    "11013": 1,  # 1분기보고서
    "11012": 2,  # 반기보고서
    "11014": 3,  # 3분기보고서
    "11011": 4,  # 사업보고서 (annual)
}


def _to_float_won(val: Any) -> float | None:
    """DART amounts are commadelimited strings ('300,870,903,000,000')."""
    if val is None or pd.isna(val):
        return None
    s = str(val).replace(",", "").strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _yoy(this_amount: float | None, prev_amount: float | None) -> float | None:
    if this_amount is None or prev_amount is None or prev_amount == 0:
        return None
    return (this_amount - prev_amount) / prev_amount


def _report_date_from_rcept_no(rcept_no: str) -> date | None:
    """``rcept_no`` of DART filings is ``YYYYMMDDNNNNNN`` — first 8 chars are
    the official filing date."""
    if len(rcept_no) >= 8 and rcept_no[:8].isdigit():
        try:
            return datetime.strptime(rcept_no[:8], "%Y%m%d").date()
        except ValueError:
            return None
    return None


def load_financials(
    stock_code: str, year: int, reprt_code: str
) -> list[dict[str, Any]]:
    """One quarter's financials for one stock. Returns long-format rows.

    Each row is one ``metric`` (8 metrics: 6 raw + 2 derived). YoY is computed
    from DART's ``frmtrm_amount`` (which is the same fiscal period a year prior).
    """
    if reprt_code not in _REPRT_CODE_TO_QUARTER:
        raise ValueError(f"Unknown reprt_code: {reprt_code}")
    quarter = _REPRT_CODE_TO_QUARTER[reprt_code]

    dart = _get_dart()
    df = dart.finstate(stock_code, year, reprt_code=reprt_code)
    if df is None or df.empty:
        return []

    # Recover the actual filing date from rcept_no.
    rcept_no_first = str(df.iloc[0]["rcept_no"]) if "rcept_no" in df.columns else ""
    report_dt = _report_date_from_rcept_no(rcept_no_first) or date.today()

    raw: dict[str, float | None] = {}
    prev: dict[str, float | None] = {}
    for _, row in df.iterrows():
        metric = _ACCOUNT_TO_METRIC.get(str(row["account_nm"]).strip())
        if metric is None:
            continue
        # When a metric appears more than once (e.g. duplicated 당기순이익 rows
        # in IS), the first non-null wins.
        if raw.get(metric) is None:
            raw[metric] = _to_float_won(row.get("thstrm_amount"))
        if prev.get(metric) is None:
            prev[metric] = _to_float_won(row.get("frmtrm_amount"))

    if not raw:
        return []

    # Derived metrics (no YoY for ratios — they're already comparative)
    if raw.get("total_liabilities") is not None and raw.get("total_equity"):
        raw["debt_ratio"] = raw["total_liabilities"] / raw["total_equity"]  # type: ignore[operator]
    if raw.get("net_income") is not None and raw.get("total_equity"):
        raw["roe"] = raw["net_income"] / raw["total_equity"]  # type: ignore[operator]

    base_metrics = [
        "total_assets",
        "total_liabilities",
        "total_equity",
        "revenue",
        "operating_income",
        "net_income",
        "debt_ratio",
        "roe",
    ]
    now_iso = datetime.now(KST).isoformat()
    rows: list[dict[str, Any]] = []
    for metric in base_metrics:
        value = raw.get(metric)
        if value is None:
            continue
        yoy = None if metric in {"debt_ratio", "roe"} else _yoy(value, prev.get(metric))
        rows.append(
            {
                "stock_code": stock_code,
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "metric": metric,
                "value": value,
                "yoy_change": yoy,
                "report_date": report_dt.isoformat(),
                "as_of": now_iso,
                "data_freshness": FRESHNESS_REPORT,
            }
        )
    return rows
