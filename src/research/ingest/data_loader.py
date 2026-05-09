"""End-to-end M1 backfill / daily-incremental orchestrator.

Runs all 5 loaders for the 8 covered stocks and the 4 macro series. Idempotent
— re-running is safe (each loader's DELETE-then-APPEND scope is the loader's
natural batch key).

Run as a module::

    uv run python -m src.research.ingest.data_loader
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

from google.cloud import bigquery

from src.research.ingest.bq import (
    delete_by_series_and_date_range,
    delete_by_stock_and_date_range,
    delete_by_values,
    delete_financials_quarter,
    ensure_all_tables,
    get_bq_client,
    load_rows,
    row_counts,
)
from src.research.ingest.dart_loader import load_disclosures, load_financials
from src.research.ingest.fred_loader import SERIES_FRESHNESS, load_macro_series
from src.research.ingest.naver_rss_loader import load_news
from src.research.ingest.pykrx_loader import load_flows, load_prices, load_shorts
from src.research.ingest.schemas import TABLES

log = logging.getLogger(__name__)

STOCKS: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "네이버",
    "035720": "카카오",
    "068270": "셀트리온",
    "105560": "KB금융",
    "012450": "한화에어로스페이스",
}

DEFAULT_LOOKBACK_DAYS = 150  # MA60 needs ≥ 60 trading days; 150 calendar ≈ 100 trading
NEWS_PAGE_SIZE = 50  # Naver API rejects pageSize > 50 with HTTP 400
MACRO_LOOKBACK_DAYS = 365

# (year, reprt_code) — last 4 quarters covered by DART. Keep small at first run
# to control runtime; widen later if the M3 disclosure_and_financial_events
# node needs deeper history.
DEFAULT_FINANCIAL_QUARTERS: list[tuple[int, str]] = [
    (2024, "11011"),  # 2024 annual
    (2025, "11013"),  # 2025 Q1
    (2025, "11012"),  # 2025 H1 (cumulative)
    (2025, "11014"),  # 2025 Q3
]


@dataclass(frozen=True)
class BackfillStats:
    table_counts: dict[str, int]
    elapsed_s: float


def _backfill_one_stock(
    client: bigquery.Client,
    stock_code: str,
    start_date: date,
    end_date: date,
    financial_quarters: list[tuple[int, str]],
) -> None:
    log.info(
        "backfilling stock",
        extra={"stock_code": stock_code, "stock_name": STOCKS[stock_code]},
    )

    # prices
    prices = load_prices(stock_code, (start_date, end_date))
    delete_by_stock_and_date_range(
        client, TABLES["prices"], stock_code=stock_code, start=start_date, end=end_date
    )
    load_rows(client, TABLES["prices"], prices)

    # flows + shorts → same flows table
    flows = load_flows(stock_code, (start_date, end_date))
    shorts = load_shorts(stock_code, (start_date, end_date))
    delete_by_stock_and_date_range(
        client, TABLES["flows"], stock_code=stock_code, start=start_date, end=end_date
    )
    load_rows(client, TABLES["flows"], flows + shorts)

    # news (single shot — page_size=100; OK for 90-day backfill on 8 stocks)
    news = load_news(stock_code, (start_date, end_date), page_size=NEWS_PAGE_SIZE)
    if news:
        delete_by_values(
            client, TABLES["news"], column="url_hash",
            values=[r["url_hash"] for r in news],
        )
        load_rows(client, TABLES["news"], news)

    # disclosures
    disc = load_disclosures(stock_code, (start_date, end_date))
    if disc:
        delete_by_values(
            client, TABLES["disclosures_market"], column="rcept_no",
            values=[r["rcept_no"] for r in disc],
        )
        load_rows(client, TABLES["disclosures_market"], disc)

    # financials — multi-quarter
    for year, reprt_code in financial_quarters:
        fin = load_financials(stock_code, year, reprt_code)
        if not fin:
            continue
        # Idempotent at (stock, year, quarter) granularity
        first = fin[0]
        delete_financials_quarter(
            client,
            stock_code=stock_code,
            fiscal_year=int(first["fiscal_year"]),
            fiscal_quarter=int(first["fiscal_quarter"]),
        )
        load_rows(client, TABLES["financials"], fin)


def _backfill_macro(client: bigquery.Client, end_date: date) -> None:
    macro_start = end_date - timedelta(days=MACRO_LOOKBACK_DAYS)
    for series_id in SERIES_FRESHNESS:
        rows = load_macro_series(series_id, (macro_start, end_date))
        delete_by_series_and_date_range(
            client, TABLES["macro"],
            series_id=series_id, start=macro_start, end=end_date,
        )
        load_rows(client, TABLES["macro"], rows)


def run_backfill(
    *,
    end_date: date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    financial_quarters: list[tuple[int, str]] | None = None,
) -> BackfillStats:
    """Backfill all 8 stocks + macro for the past ``lookback_days``."""
    end_date = end_date or date.today()
    start_date = end_date - timedelta(days=lookback_days)
    quarters = financial_quarters if financial_quarters is not None else DEFAULT_FINANCIAL_QUARTERS

    t0 = time.perf_counter()
    client = get_bq_client()
    ensure_all_tables(client)

    for stock_code in STOCKS:
        _backfill_one_stock(client, stock_code, start_date, end_date, quarters)

    _backfill_macro(client, end_date)

    counts = row_counts(client)
    return BackfillStats(table_counts=counts, elapsed_s=time.perf_counter() - t0)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    stats = run_backfill()
    print()
    print("=== Backfill complete ===")
    for table, n in stats.table_counts.items():
        print(f"  {table:<22} {n:>6} rows")
    print(f"  elapsed: {stats.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
