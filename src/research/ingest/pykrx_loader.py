"""PyKRX-based loaders for prices / flows / short balance.

PyKRX scrapes KRX info-data web pages:
  - OHLCV is on a public endpoint (no login required).
  - Trading value (외인/기관/개인 순매수) and short balance require KRX
    member login via ``KRX_ID`` / ``KRX_PW`` env vars. We hydrate those
    lazily from Secret Manager (`krx-id` / `krx-pw`) on first call.

Freshness contract (per plan.md M1):
  - prices         : T-0 after market close (16:00 KST), intraday before.
  - flows (long)   : T-1 (foreign / institution / individual net buys).
  - short_balance  : T-2 (official 2-day publication delay).

PyKRX caveats observed (2026-05-09):
  - ``get_market_ohlcv(from, to, ticker)`` returns only OHLCV+volume+등락률 —
    거래대금 (trading value, KRW) is NOT in single-stock time-series mode.
    We leave the ``value`` column NULL; technical_signals only uses 거래량.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pykrx import stock

from src.common.config import get_krx_credentials
from src.research.ingest.schemas import (
    FRESHNESS_INTRADAY,
    FRESHNESS_T0,
    FRESHNESS_T1,
    FRESHNESS_T2,
)

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
MARKET_CLOSE = time(16, 0)

# PyKRX column name → BQ flows.metric
_FLOWS_COLUMN_TO_METRIC: dict[str, str] = {
    "외국인합계": "foreign_net",
    "기관합계": "institution_net",
    "개인": "individual_net",
}

_krx_login_done = False


def ensure_krx_login() -> None:
    """Hydrate KRX_ID / KRX_PW env vars from Secret Manager. Idempotent."""
    global _krx_login_done
    if _krx_login_done:
        return
    creds = get_krx_credentials()
    os.environ["KRX_ID"] = creds.krx_id.get_secret_value()
    os.environ["KRX_PW"] = creds.krx_pw.get_secret_value()
    _krx_login_done = True
    log.info("KRX login hydrated from Secret Manager")


def _today_kst() -> date:
    return datetime.now(KST).date()


def _market_closed_kst() -> bool:
    return datetime.now(KST).time() >= MARKET_CLOSE


def _drop_intraday_today(df: pd.DataFrame, *, today: date) -> pd.DataFrame:
    """Drop today's row when called before market close — avoids storing a
    partial intraday close that contradicts the T-0 contract."""
    if df.empty or _market_closed_kst():
        return df
    today_ts = pd.Timestamp(today)
    if today_ts in df.index:
        log.info("dropping intraday today row", extra={"date": today.isoformat()})
        return df.drop(today_ts)
    return df


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _f(val: Any) -> float | None:
    if val is None or pd.isna(val):
        return None
    return float(val)


def _i(val: Any) -> int | None:
    if val is None or pd.isna(val):
        return None
    return int(val)


def load_prices(stock_code: str, date_range: tuple[date, date]) -> list[dict[str, Any]]:
    """Daily OHLCV for one stock × range. Returns rows for the ``prices`` table.

    Public endpoint — no KRX login needed.
    """
    start, end = date_range
    df = stock.get_market_ohlcv(_yyyymmdd(start), _yyyymmdd(end), stock_code)
    df = _drop_intraday_today(df, today=_today_kst())
    if df.empty:
        return []

    now_iso = datetime.now(KST).isoformat()
    today = _today_kst()
    market_closed = _market_closed_kst()
    rows: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        trade_date = ts.date()
        # today row was already dropped pre-close; if it survives here it's
        # either past (T-0) or today after close (T-0).
        freshness = (
            FRESHNESS_INTRADAY
            if trade_date == today and not market_closed
            else FRESHNESS_T0
        )
        rows.append(
            {
                "stock_code": stock_code,
                "trade_date": trade_date.isoformat(),
                "open": _f(row.get("시가")),
                "high": _f(row.get("고가")),
                "low": _f(row.get("저가")),
                "close": _f(row.get("종가")),
                "volume": _i(row.get("거래량")),
                "value": None,  # 거래대금 — PyKRX 단일종목 모드 미제공
                "as_of": now_iso,
                "data_freshness": freshness,
            }
        )
    return rows


def load_flows(stock_code: str, date_range: tuple[date, date]) -> list[dict[str, Any]]:
    """Daily foreign/institution/individual net buy values (KRW).

    Returns long-format rows for the ``flows`` table with ``metric`` ∈
    {foreign_net, institution_net, individual_net}. Freshness = T-1.
    Requires KRX member login.
    """
    ensure_krx_login()
    start, end = date_range
    df = stock.get_market_trading_value_by_date(_yyyymmdd(start), _yyyymmdd(end), stock_code)
    if df.empty:
        return []

    now_iso = datetime.now(KST).isoformat()
    rows: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        for col, metric in _FLOWS_COLUMN_TO_METRIC.items():
            if col not in row.index:
                continue
            rows.append(
                {
                    "stock_code": stock_code,
                    "trade_date": ts.date().isoformat(),
                    "metric": metric,
                    "value": _f(row[col]),
                    "as_of": now_iso,
                    "data_freshness": FRESHNESS_T1,
                }
            )
    return rows


def load_shorts(stock_code: str, date_range: tuple[date, date]) -> list[dict[str, Any]]:
    """Daily short balance (잔고 수량 또는 금액 — varies by PyKRX version).

    Returns long-format rows for ``flows`` with ``metric`` = short_balance.
    Freshness = T-2. Requires KRX member login.
    """
    ensure_krx_login()
    start, end = date_range
    df = stock.get_shorting_balance_by_date(_yyyymmdd(start), _yyyymmdd(end), stock_code)
    if df.empty:
        return []

    short_col = "공매도잔고" if "공매도잔고" in df.columns else df.columns[0]
    now_iso = datetime.now(KST).isoformat()
    rows: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        rows.append(
            {
                "stock_code": stock_code,
                "trade_date": ts.date().isoformat(),
                "metric": "short_balance",
                "value": _f(row.get(short_col)),
                "as_of": now_iso,
                "data_freshness": FRESHNESS_T2,
            }
        )
    return rows
