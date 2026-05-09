"""macro_context — global macro fact pack (stock-agnostic).

plan.md M3 contract: 4 series, each with (a) latest value, (b) 1m/3m
change, (c) observation date. Stock-impact interpretation is delegated to
the LLM downstream — this node only emits past-fact sentences.

``stock_code`` is None on the output (signature differs from per-stock
signal nodes). Per-series freshness is surfaced in sentences themselves
("daily" / "monthly") so the LLM never confuses cadences.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from google.cloud import bigquery
from pydantic import ConfigDict

from src.research.ingest.bq import get_bq_client, table_id
from src.research.signals._types import SignalOutput

log = logging.getLogger(__name__)

_SERIES_KOR: dict[str, tuple[str, str]] = {
    "IRSTCI01KRM156N": ("한국 기준금리", "%"),
    "DFF": ("미국 연방기금금리", "%"),
    "DEXKOUS": ("원/달러 환율", "원"),
    "VIXCLS": ("VIX", ""),
}


class MacroSignals(SignalOutput):
    """Output of ``macro_context``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_snapshot: dict[str, dict[str, float | str | None]]


def _fetch_macro_latest(client: bigquery.Client, as_of: date) -> pd.DataFrame:
    """One row per series_id — the latest observation ≤ as_of."""
    tid = table_id("macro")
    q = f"""
    WITH ranked AS (
      SELECT
        series_id, observation_date, value, change_1m, change_3m,
        data_freshness,
        ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY observation_date DESC) AS rn
      FROM `{tid}`
      WHERE observation_date <= @as_of
    )
    SELECT * FROM ranked WHERE rn = 1
    """
    return client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
            ],
        ),
    ).to_dataframe()


def _format_value(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "원":
        return f"{value:,.2f}원"
    return f"{value:.2f}"


def _format_change(change: float, unit: str) -> str:
    # Suppress "-0.00" for near-zero values
    if abs(change) < 0.005:
        change = 0.0
    sign = "+" if change >= 0 else ""
    if unit == "%":
        return f"{sign}{change:.2f}%p"  # rate change in percentage points
    if unit == "원":
        return f"{sign}{change:,.2f}원"
    return f"{sign}{change:.2f}"


def compute(
    as_of: date,
    *,
    client: bigquery.Client | None = None,
) -> MacroSignals:
    """Run macro_context as of ``as_of``."""
    client = client or get_bq_client()
    df = _fetch_macro_latest(client, as_of)

    snapshot: dict[str, dict[str, float | str | None]] = {}
    sentences: list[str] = []

    for sid, (kor, unit) in _SERIES_KOR.items():
        rows = df[df["series_id"] == sid]
        if rows.empty:
            continue
        row = rows.iloc[0]
        value = row["value"]
        change_1m = row["change_1m"]
        change_3m = row["change_3m"]
        obs_date = row["observation_date"]
        cadence = str(row["data_freshness"])

        snapshot[sid] = {
            "korean_name": kor,
            "value": float(value) if pd.notna(value) else None,
            "change_1m": float(change_1m) if pd.notna(change_1m) else None,
            "change_3m": float(change_3m) if pd.notna(change_3m) else None,
            "observation_date": str(obs_date),
            "cadence": cadence,
        }

        if pd.isna(value):
            continue
        val_str = _format_value(float(value), unit)
        parts = [f"{kor} ({cadence}, {obs_date} 발표) 최근 값은 {val_str} 으로 측정됨"]
        if pd.notna(change_1m):
            parts.append(f"1개월 변동 {_format_change(float(change_1m), unit)}")
        if pd.notna(change_3m):
            parts.append(f"3개월 변동 {_format_change(float(change_3m), unit)}")
        sentences.append(", ".join(parts))

    return MacroSignals(
        stock_code=None,  # macro is stock-agnostic
        as_of=as_of,
        data_freshness="mixed",  # per-series cadence is in each sentence
        sentences=sentences,
        series_snapshot=snapshot,
    )


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    as_of_str = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    out = compute(date.fromisoformat(as_of_str))
    print(out.model_dump_json(indent=2))
    print()
    for s in out.sentences:
        print(f"  • {s}")


if __name__ == "__main__":
    main()
