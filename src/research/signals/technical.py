"""technical_signals — past-trade interpretation only.

plan.md M3 contract: every output sentence describes *what trades have
already occurred*. Forward-projection language is forbidden (enforced by
``_lint`` via the ``SignalOutput`` validator).

Signals computed from ``dart_rag.prices``:
  - **MA20/MA60 crossover** in the last 5 trading days (boolean event).
  - **Volume z-score** of the most recent day vs. the prior 60-day distribution.
  - **20-day volatility** = std-dev of daily returns, last 20 days.
  - **5-day price change** = (close[t] - close[t-5]) / close[t-5].

If insufficient history (< 20 rows) is available for the stock, we return
an output with empty ``sentences`` rather than raising — keeps the agent
graph robust to corpus gaps.
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

LOOKBACK_DAYS = 90  # ~60 trading days + buffer; matches M1 backfill window


class TechnicalSignals(SignalOutput):
    """Output of ``technical_signals`` — past-trade fact pack."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ma_cross_recent_5d: bool | None
    volume_ratio_to_60d_mean: float | None
    volume_z_score_60d: float | None
    volatility_20d: float | None
    price_change_5d: float | None
    n_observations: int


def _fetch_prices(
    client: bigquery.Client, stock_code: str, as_of: date
) -> pd.DataFrame:
    tid = table_id("prices")
    query = (
        f"SELECT trade_date, close, volume, data_freshness "
        f"FROM `{tid}` "
        f"WHERE stock_code = @sc AND trade_date <= @as_of "
        f"ORDER BY trade_date ASC "
        f"LIMIT 200"  # plenty of headroom; 90-day window holds ~60 rows
    )
    df = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
            ],
        ),
    ).to_dataframe()
    return df


def _compute_from_df(
    df: pd.DataFrame, stock_code: str, as_of: date
) -> TechnicalSignals:
    n = len(df)
    if n < 20:
        return TechnicalSignals(
            stock_code=stock_code,
            as_of=as_of,
            data_freshness=str(df["data_freshness"].iloc[-1]) if n else "unknown",
            sentences=[],
            ma_cross_recent_5d=None,
            volume_ratio_to_60d_mean=None,
            volume_z_score_60d=None,
            volatility_20d=None,
            price_change_5d=None,
            n_observations=n,
        )

    # MA cross in last 5 trading days (only meaningful with >= 60 rows)
    ma_cross_recent_5d: bool | None = None
    if n >= 60:
        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()
        diff = (ma20 - ma60).iloc[-5:]
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in diff if pd.notna(v)]
        ma_cross_recent_5d = len(set(signs)) > 1

    # Volume z-score: today vs prior 60-day window (excluding today)
    volume_ratio: float | None = None
    volume_z: float | None = None
    if n >= 60:
        prior = df["volume"].iloc[-61:-1]
        mean_v, std_v = prior.mean(), prior.std()
        today_v = float(df["volume"].iloc[-1])
        if mean_v > 0:
            volume_ratio = today_v / float(mean_v)
        if std_v and std_v > 0:
            volume_z = (today_v - float(mean_v)) / float(std_v)

    # 20-day volatility (std of daily returns)
    returns = df["close"].pct_change().dropna()
    volatility_20d: float | None = None
    if len(returns) >= 20:
        v = returns.iloc[-20:].std()
        if pd.notna(v):
            volatility_20d = float(v)

    # 5-day price change
    price_chg_5d: float | None = None
    if n > 5:
        close_now = float(df["close"].iloc[-1])
        close_5 = float(df["close"].iloc[-6])
        if close_5 != 0:
            price_chg_5d = (close_now - close_5) / close_5

    # Build past-tense sentences
    sentences: list[str] = []
    if ma_cross_recent_5d is True:
        sentences.append("최근 5거래일 내 단기-장기 이평선이 교차한 거래가 있었음")
    elif ma_cross_recent_5d is False:
        sentences.append("최근 5거래일 내 단기-장기 이평선 교차는 없었음")
    if volume_ratio is not None:
        sentences.append(f"최근 거래일 거래량은 60일 평균 대비 {volume_ratio:.1f}배 체결됨")
    if volatility_20d is not None:
        sentences.append(f"20일 일별 수익률 표준편차는 {volatility_20d * 100:.2f}% 로 측정됨")
    if price_chg_5d is not None:
        sign = "+" if price_chg_5d >= 0 else ""
        sentences.append(f"직전 5거래일간 종가 변동폭은 {sign}{price_chg_5d * 100:.2f}% 로 기록됨")

    return TechnicalSignals(
        stock_code=stock_code,
        as_of=as_of,
        data_freshness=str(df["data_freshness"].iloc[-1]),
        sentences=sentences,
        ma_cross_recent_5d=ma_cross_recent_5d,
        volume_ratio_to_60d_mean=volume_ratio,
        volume_z_score_60d=volume_z,
        volatility_20d=volatility_20d,
        price_change_5d=price_chg_5d,
        n_observations=n,
    )


def compute(
    stock_code: str,
    as_of: date,
    *,
    client: bigquery.Client | None = None,
) -> TechnicalSignals:
    """Run technical_signals for one stock as of ``as_of``."""
    client = client or get_bq_client()
    df = _fetch_prices(client, stock_code, as_of)
    return _compute_from_df(df, stock_code, as_of)


def main() -> None:
    """Eyeball: print signal for one stock × one date."""
    import sys

    logging.basicConfig(level=logging.INFO)
    stock_code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    as_of_str = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    as_of = date.fromisoformat(as_of_str)
    out = compute(stock_code, as_of)
    print(out.model_dump_json(indent=2))
    print()
    print("--- sentences ---")
    for s in out.sentences:
        print(f"  • {s}")


if __name__ == "__main__":
    main()
