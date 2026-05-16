"""disclosure_and_financial_events — recent filings + latest financials.

plan.md M3 contract:
  - Disclosures: bucket recent filings by ``category`` (already classified at
    M1 ingest time) within the last ``window_days``. Surface counts.
  - Financials: latest available quarter per metric + YoY change. Quarter
    deltas (QoQ) by comparing latest two quarters.

Output ``data_freshness`` is "T-0" for the disclosures portion; financial
sentences carry "{year}년 {quarter}분기 보고서" inline so the LLM never
mistakes a stale quarterly figure for live data.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from google.cloud import bigquery
from pydantic import ConfigDict

from src.research.ingest.bq import get_bq_client, table_id
from src.research.signals._types import SignalOutput
from src.research.signals._urls import dart_viewer_url

log = logging.getLogger(__name__)

DEFAULT_DISCLOSURE_WINDOW_DAYS = 90

# Categories that are "noteworthy" for surfacing as a sentence even at low count.
# Routine filings (annual report submission, dividend disclosure) are still
# bucketed but worded neutrally.
_CATEGORY_KOR: dict[str, str] = {
    "annual_report": "사업보고서 제출",
    "semi_annual_report": "반기보고서 제출",
    "quarterly_report": "분기보고서 제출",
    "rights_issue": "유상증자 공시",
    "convertible_bond": "전환사채 공시",
    "warrant_bond": "신주인수권부사채 공시",
    "exchangeable_bond": "교환사채 공시",
    "audit": "감사 관련 공시",
    "material_event": "주요사항보고서",
    "treasury_stock": "자기주식 공시",
    "dividend": "배당 공시",
    "earnings_announcement": "실적 발표 공시",
}

_METRIC_KOR: dict[str, str] = {
    "total_assets": "자산총계",
    "total_liabilities": "부채총계",
    "total_equity": "자본총계",
    "revenue": "매출액",
    "operating_income": "영업이익",
    "net_income": "당기순이익",
    "debt_ratio": "부채비율",
    "roe": "ROE",
}

# Metrics that are ratios (printed as %) vs absolute KRW
_RATIO_METRICS = {"debt_ratio", "roe"}

# Periodic-report categories — fin sentences point to the latest of these.
_PERIODIC_REPORT_CATEGORIES = ("annual_report", "semi_annual_report", "quarterly_report")


class DisclosureAndFinancialSignals(SignalOutput):
    """Recent disclosures by category + latest-quarter financials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_days: int
    disclosure_counts: dict[str, int]  # category → count (only categorised filings)
    uncategorised_count: int
    latest_quarter: str | None  # e.g. "2025Q3"
    financial_metrics: dict[str, float]  # metric → value (latest quarter)
    yoy_changes: dict[str, float]  # metric → YoY % change


def _fetch_disclosures(
    client: bigquery.Client, stock_code: str, as_of: date, window_days: int
) -> pd.DataFrame:
    tid = table_id("disclosures_market")
    start = as_of - timedelta(days=window_days)
    q = (
        f"SELECT rcept_no, rcept_dt, category, title FROM `{tid}` "
        f"WHERE stock_code=@sc AND rcept_dt BETWEEN @start AND @as_of "
        f"ORDER BY rcept_dt DESC"
    )
    return client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("start", "DATE", start),
                bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
            ],
        ),
    ).to_dataframe()


def _fetch_latest_periodic_report_rcept_no(
    client: bigquery.Client, stock_code: str, as_of: date
) -> str | None:
    """Most recent annual/semi/quarterly report rcept_no ≤ as_of, regardless
    of the disclosure window. Used to attach a DART viewer URL to fin
    sentences — the report whose XBRL produced the latest financials."""
    tid = table_id("disclosures_market")
    q = f"""
    SELECT rcept_no FROM `{tid}`
    WHERE stock_code=@sc AND rcept_dt <= @as_of
      AND category IN UNNEST(@cats)
    ORDER BY rcept_dt DESC
    LIMIT 1
    """
    rows = list(
        client.query(
            q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                    bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
                    bigquery.ArrayQueryParameter(
                        "cats", "STRING", list(_PERIODIC_REPORT_CATEGORIES)
                    ),
                ],
            ),
        ).result()
    )
    return str(rows[0]["rcept_no"]) if rows else None


def _fetch_latest_financials(client: bigquery.Client, stock_code: str, as_of: date) -> pd.DataFrame:
    """Latest fiscal quarter ≤ as_of for the stock. Returns rows for that
    single (year, quarter) tuple — all 8 metrics."""
    tid = table_id("financials")
    q = f"""
    WITH latest AS (
      SELECT fiscal_year, fiscal_quarter
      FROM `{tid}`
      WHERE stock_code=@sc AND report_date <= @as_of
      ORDER BY report_date DESC
      LIMIT 1
    )
    SELECT f.metric, f.value, f.yoy_change, f.fiscal_year, f.fiscal_quarter, f.report_date
    FROM `{tid}` f
    JOIN latest l USING (fiscal_year, fiscal_quarter)
    WHERE f.stock_code=@sc
    """
    return client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
            ],
        ),
    ).to_dataframe()


def _format_eok(value: float) -> str:
    eok = value / 1e8
    if abs(eok) >= 1e4:
        return f"{eok / 1e4:,.1f}조원"
    return f"{eok:,.0f}억원"


def _format_value(metric: str, value: float) -> str:
    if metric in _RATIO_METRICS:
        return f"{value * 100:.2f}%"
    return _format_eok(value)


def compute(
    stock_code: str,
    as_of: date,
    *,
    window_days: int = DEFAULT_DISCLOSURE_WINDOW_DAYS,
    client: bigquery.Client | None = None,
) -> DisclosureAndFinancialSignals:
    """Run disclosure_and_financial_events for one stock as of ``as_of``."""
    client = client or get_bq_client()
    disc = _fetch_disclosures(client, stock_code, as_of, window_days)
    fin = _fetch_latest_financials(client, stock_code, as_of)

    counts: dict[str, int] = {}
    rcept_by_category: dict[str, list[str]] = {}
    uncat = 0
    if not disc.empty:
        for _, row in disc.iterrows():
            cat = row["category"]
            rcept_no = str(row["rcept_no"])
            if cat is None or (isinstance(cat, float) and pd.isna(cat)):
                uncat += 1
            else:
                key = str(cat)
                counts[key] = counts.get(key, 0) + 1
                rcept_by_category.setdefault(key, []).append(rcept_no)

    latest_q: str | None = None
    metrics: dict[str, float] = {}
    yoy: dict[str, float] = {}
    if not fin.empty:
        first = fin.iloc[0]
        latest_q = f"{int(first['fiscal_year'])}Q{int(first['fiscal_quarter'])}"
        for _, row in fin.iterrows():
            metric = str(row["metric"])
            v = row.get("value")
            if v is not None and not pd.isna(v):
                metrics[metric] = float(v)
            y = row.get("yoy_change")
            if y is not None and not pd.isna(y):
                yoy[metric] = float(y)

    sentences: list[str] = []
    sentence_urls: list[list[str]] = []
    # Disclosure summary — list each present category once with its count.
    # URLs: all rcept_no in that category (most-recent first per fetch ORDER BY).
    if counts:
        for cat, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            label = _CATEGORY_KOR.get(cat, cat)
            sentences.append(f"직전 {window_days}일 동안 {label}이(가) {n}건 접수됨")
            sentence_urls.append([dart_viewer_url(r) for r in rcept_by_category.get(cat, [])])
    # Latest financials — emphasise the noteworthy lines (revenue / op_income /
    # net_income / debt_ratio / roe). yoy when available.
    # URLs: the latest periodic-report rcept_no (annual/semi/quarterly) — user
    # lands on the DART viewer root and clicks "Ⅲ. 재무에 관한 사항".
    fin_url: list[str] = []
    if latest_q and metrics:
        rcept_no = _fetch_latest_periodic_report_rcept_no(client, stock_code, as_of)
        if rcept_no:
            fin_url = [dart_viewer_url(rcept_no)]
        emphasis = ["revenue", "operating_income", "net_income", "debt_ratio", "roe"]
        for metric in emphasis:
            if metric not in metrics:
                continue
            kor = _METRIC_KOR[metric]
            val_str = _format_value(metric, metrics[metric])
            if metric in yoy:
                sign = "+" if yoy[metric] >= 0 else ""
                sentences.append(
                    f"{latest_q} 기준 {kor}은 {val_str} 으로 보고됨 "
                    f"(전년 동기 대비 {sign}{yoy[metric] * 100:.1f}%)"
                )
            else:
                sentences.append(f"{latest_q} 기준 {kor}은 {val_str} 으로 보고됨")
            sentence_urls.append(list(fin_url))

    return DisclosureAndFinancialSignals(
        stock_code=stock_code,
        as_of=as_of,
        data_freshness="T-0",
        sentences=sentences,
        sentence_urls=sentence_urls,
        window_days=window_days,
        disclosure_counts=counts,
        uncategorised_count=uncat,
        latest_quarter=latest_q,
        financial_metrics=metrics,
        yoy_changes=yoy,
    )


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    stock_code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    as_of_str = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    out = compute(stock_code, date.fromisoformat(as_of_str))
    print(out.model_dump_json(indent=2, exclude_none=False))
    print()
    for s in out.sentences:
        print(f"  • {s}")


if __name__ == "__main__":
    main()
