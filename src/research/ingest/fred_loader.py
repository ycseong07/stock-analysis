"""FRED-based loader for macro context (4 series).

Series tracked (per plan.md M1):
  - IRSTCI01KRM156N : KR central bank rate            (monthly)
  - DFF             : US federal funds rate           (daily)
  - DEXKOUS         : KRW/USD spot rate               (daily, business days)
  - VIXCLS          : VIX                             (daily, business days)

KR CPI was originally specified (``KORCPIALLMINMEI``) but FRED-hosted Korean
CPI series all lag 12+ months — see decision 2026-05-09 in plan.md.

`change_1m` / `change_3m` are absolute diffs ``value[t] - value[t-Nd]`` where
N is 30 / 90 calendar days. Signal nodes can re-derive % change as needed.

Note: macro is *not* per-stock; signature differs from PyKRX loaders. Symptom-
relief signal interpretation is delegated to the LLM (see plan.md M3
``macro_context``).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd
from fredapi import Fred

from src.common.config import get_fred_api_key
from src.research.ingest.schemas import FRESHNESS_DAILY, FRESHNESS_MONTHLY

log = logging.getLogger(__name__)

SERIES_FRESHNESS: dict[str, str] = {
    "IRSTCI01KRM156N": FRESHNESS_MONTHLY,
    "DFF": FRESHNESS_DAILY,
    "DEXKOUS": FRESHNESS_DAILY,
    "VIXCLS": FRESHNESS_DAILY,
}

_fred_client: Fred | None = None


def _get_fred() -> Fred:
    global _fred_client
    if _fred_client is None:
        _fred_client = Fred(api_key=get_fred_api_key().get_secret_value())
    return _fred_client


def _diff_back(series: pd.Series, ts: pd.Timestamp, days: int) -> float | None:
    """``value[ts] - last value at-or-before (ts - days)``. ``None`` if no prior."""
    cutoff = ts - pd.Timedelta(days=days)
    prior = series.loc[:cutoff]
    if prior.empty:
        return None
    return float(series.loc[ts] - prior.iloc[-1])


def load_macro_series(series_id: str, date_range: tuple[date, date]) -> list[dict[str, Any]]:
    """Fetch one FRED series for the date range. Returns rows for ``macro``."""
    if series_id not in SERIES_FRESHNESS:
        raise ValueError(
            f"Unknown series_id: {series_id}; expected one of {list(SERIES_FRESHNESS)}"
        )
    freshness = SERIES_FRESHNESS[series_id]
    start, end = date_range

    fred = _get_fred()
    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    series = series.dropna()
    if series.empty:
        return []

    now_iso = datetime.now().astimezone().isoformat()
    rows: list[dict[str, Any]] = []
    for ts, value in series.items():
        rows.append(
            {
                "series_id": series_id,
                "observation_date": ts.date().isoformat(),
                "value": float(value),
                "change_1m": _diff_back(series, ts, days=30),
                "change_3m": _diff_back(series, ts, days=90),
                "as_of": now_iso,
                "data_freshness": freshness,
            }
        )
    return rows


def load_all_macro(date_range: tuple[date, date]) -> dict[str, list[dict[str, Any]]]:
    """Fetch all 5 series. Returns ``{series_id: rows}``."""
    return {sid: load_macro_series(sid, date_range) for sid in SERIES_FRESHNESS}
