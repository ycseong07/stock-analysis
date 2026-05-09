"""flow_signals — investor flows + short balance, with strict T-1 / T-2 metadata.

plan.md M3 contract: foreign / institution / individual net buys are T-1
(post-close), short balance is T-2 (official 2-day delay). The signal output
tags ``data_freshness`` as ``T-2`` (worst case) and surfaces the per-metric
freshness in sentences themselves (e.g. "T-1 기준", "T-2 기준").

Signals computed from ``dart_rag.flows`` (long format):
  - **외인 연속 순매수 일수** : streak of foreign_net > 0 from latest day.
  - **5일 누적 순매수**       : sum of last-5 foreign / institution / individual.
  - **공매도 잔고 추이**       : pct change of short_balance over last 5 days.
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


class FlowSignals(SignalOutput):
    """Output of ``flow_signals``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    foreign_streak_days: int | None
    foreign_net_sum_5d: float | None
    institution_net_sum_5d: float | None
    individual_net_sum_5d: float | None
    short_balance_pct_change_5d: float | None
    n_observations: int


def _fetch_flows(
    client: bigquery.Client, stock_code: str, as_of: date
) -> pd.DataFrame:
    tid = table_id("flows")
    query = (
        f"SELECT trade_date, metric, value "
        f"FROM `{tid}` "
        f"WHERE stock_code = @sc AND trade_date <= @as_of "
        f"ORDER BY trade_date ASC "
        f"LIMIT 400"  # 4 metrics × ~100 days headroom
    )
    return client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
            ],
        ),
    ).to_dataframe()


def _format_billion_won(value: float) -> str:
    """KRW → '+1,234억원' / '-567억원' (1억원 = 1e8)."""
    eok = value / 1e8
    sign = "+" if eok >= 0 else ""
    return f"{sign}{eok:,.0f}억원"


def _foreign_buy_streak(foreign_series: pd.Series) -> int:
    """Trailing streak of positive foreign_net values from the latest day."""
    streak = 0
    for v in foreign_series.iloc[::-1]:
        if pd.isna(v) or v <= 0:
            break
        streak += 1
    return streak


def _compute_from_df(
    df: pd.DataFrame, stock_code: str, as_of: date
) -> FlowSignals:
    n = len(df)
    if df.empty:
        return FlowSignals(
            stock_code=stock_code,
            as_of=as_of,
            data_freshness="T-2",
            sentences=[],
            foreign_streak_days=None,
            foreign_net_sum_5d=None,
            institution_net_sum_5d=None,
            individual_net_sum_5d=None,
            short_balance_pct_change_5d=None,
            n_observations=n,
        )

    wide = df.pivot(index="trade_date", columns="metric", values="value").sort_index()

    foreign = wide.get("foreign_net")
    institution = wide.get("institution_net")
    individual = wide.get("individual_net")
    short_bal = wide.get("short_balance")

    foreign_streak = _foreign_buy_streak(foreign) if foreign is not None else None

    def _sum_last_5(s: pd.Series | None) -> float | None:
        if s is None:
            return None
        tail = s.dropna().iloc[-5:]
        return float(tail.sum()) if not tail.empty else None

    f_sum = _sum_last_5(foreign)
    i_sum = _sum_last_5(institution)
    p_sum = _sum_last_5(individual)

    short_chg: float | None = None
    if short_bal is not None:
        sb = short_bal.dropna()
        if len(sb) >= 6 and sb.iloc[-6] != 0:
            short_chg = (float(sb.iloc[-1]) - float(sb.iloc[-6])) / float(sb.iloc[-6])

    sentences: list[str] = []
    if foreign_streak is not None and foreign_streak > 0:
        sentences.append(f"외국인은 T-1 기준 직전 {foreign_streak}거래일 연속 순매수가 체결됨")
    elif foreign_streak == 0:
        sentences.append("외국인은 T-1 기준 직전 거래일에 연속 순매수가 없었음")
    if f_sum is not None:
        sentences.append(
            f"T-1 기준 직전 5거래일 외국인 누적 순매수는 {_format_billion_won(f_sum)} 으로 집계됨"
        )
    if i_sum is not None:
        sentences.append(
            f"T-1 기준 직전 5거래일 기관 누적 순매수는 {_format_billion_won(i_sum)} 으로 집계됨"
        )
    if p_sum is not None:
        sentences.append(
            f"T-1 기준 직전 5거래일 개인 누적 순매수는 {_format_billion_won(p_sum)} 으로 집계됨"
        )
    if short_chg is not None:
        sign = "+" if short_chg >= 0 else ""
        sentences.append(
            f"T-2 기준 공매도 잔고는 직전 5거래일 {sign}{short_chg * 100:.1f}% 변동했음"
        )

    return FlowSignals(
        stock_code=stock_code,
        as_of=as_of,
        data_freshness="T-2",  # worst-case among row-level freshness
        sentences=sentences,
        foreign_streak_days=foreign_streak,
        foreign_net_sum_5d=f_sum,
        institution_net_sum_5d=i_sum,
        individual_net_sum_5d=p_sum,
        short_balance_pct_change_5d=short_chg,
        n_observations=n,
    )


def compute(
    stock_code: str,
    as_of: date,
    *,
    client: bigquery.Client | None = None,
) -> FlowSignals:
    """Run flow_signals for one stock as of ``as_of``."""
    client = client or get_bq_client()
    df = _fetch_flows(client, stock_code, as_of)
    return _compute_from_df(df, stock_code, as_of)


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    stock_code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    as_of_str = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    out = compute(stock_code, date.fromisoformat(as_of_str))
    print(out.model_dump_json(indent=2))
    print()
    for s in out.sentences:
        print(f"  • {s}")


if __name__ == "__main__":
    main()
