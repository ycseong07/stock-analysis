"""BigQuery helpers for the research card pipeline.

Mirrors the patterns in ``src/index/load.py`` (1번):
  - ``CREATE TABLE IF NOT EXISTS`` via DDL.
  - ``load_table_from_json`` via load job (free; streaming insert is metered).
  - Idempotent DELETE-then-APPEND, scoped by the loader's natural batch key
    (e.g. one stock × one date range).

Run as a module to create / refresh the 6 research tables::

    uv run python -m src.research.ingest.bq
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from google.cloud import bigquery

from src.common.config import get_infra
from src.research.ingest.schemas import TABLES, TableSpec

log = logging.getLogger(__name__)


def get_bq_client() -> bigquery.Client:
    """Project-bound BQ client. One per process is fine."""
    return bigquery.Client(project=get_infra().project)


def table_id(spec_or_name: TableSpec | str) -> str:
    infra = get_infra()
    name = spec_or_name.name if isinstance(spec_or_name, TableSpec) else spec_or_name
    return f"{infra.project}.{infra.bq_dataset}.{name}"


def ensure_table(client: bigquery.Client, spec: TableSpec) -> str:
    """Create the table if absent, return its full id."""
    tid = table_id(spec)
    client.query(spec.ddl.format(table=tid)).result()
    return tid


def ensure_all_tables(client: bigquery.Client | None = None) -> dict[str, str]:
    """Idempotently create all 6 research tables. Returns ``{name: full_id}``."""
    client = client or get_bq_client()
    return {name: ensure_table(client, spec) for name, spec in TABLES.items()}


def load_rows(
    client: bigquery.Client,
    spec: TableSpec,
    rows: list[dict[str, Any]],
) -> int:
    """Append rows via load job. Caller handles idempotent DELETE first.

    No-op if ``rows`` is empty. Returns the row count loaded.
    """
    if not rows:
        return 0
    tid = table_id(spec)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=spec.schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    client.load_table_from_json(rows, tid, job_config=job_config).result()
    return len(rows)


def delete_by_values(
    client: bigquery.Client,
    spec: TableSpec,
    *,
    column: str,
    values: Iterable[Any],
    sql_type: str = "STRING",
) -> None:
    """``DELETE WHERE column IN UNNEST(@values)`` — for single-column unique keys
    (e.g. ``rcept_no``, ``url_hash``)."""
    vals = list(values)
    if not vals:
        return
    tid = table_id(spec)
    client.query(
        f"DELETE FROM `{tid}` WHERE {column} IN UNNEST(@values)",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("values", sql_type, vals)],
        ),
    ).result()


def delete_by_stock_and_date_range(
    client: bigquery.Client,
    spec: TableSpec,
    *,
    stock_code: str,
    start: date,
    end: date,
    date_column: str = "trade_date",
) -> None:
    """``DELETE WHERE stock_code = @sc AND date_column BETWEEN @start AND @end``.

    Used by ``prices`` / ``flows`` loaders to refresh one stock's window.
    """
    tid = table_id(spec)
    client.query(
        f"DELETE FROM `{tid}` "
        f"WHERE stock_code = @sc AND {date_column} BETWEEN @start AND @end",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("start", "DATE", start),
                bigquery.ScalarQueryParameter("end", "DATE", end),
            ],
        ),
    ).result()


def delete_by_series_and_date_range(
    client: bigquery.Client,
    spec: TableSpec,
    *,
    series_id: str,
    start: date,
    end: date,
) -> None:
    """``DELETE WHERE series_id = @sid AND observation_date BETWEEN @start AND @end``."""
    tid = table_id(spec)
    client.query(
        f"DELETE FROM `{tid}` "
        f"WHERE series_id = @sid AND observation_date BETWEEN @start AND @end",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sid", "STRING", series_id),
                bigquery.ScalarQueryParameter("start", "DATE", start),
                bigquery.ScalarQueryParameter("end", "DATE", end),
            ],
        ),
    ).result()


def delete_financials_quarter(
    client: bigquery.Client,
    *,
    stock_code: str,
    fiscal_year: int,
    fiscal_quarter: int,
) -> None:
    """``DELETE WHERE stock=... AND fiscal_year=... AND fiscal_quarter=...``."""
    tid = table_id("financials")
    client.query(
        f"DELETE FROM `{tid}` "
        f"WHERE stock_code=@sc AND fiscal_year=@y AND fiscal_quarter=@q",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("y", "INT64", fiscal_year),
                bigquery.ScalarQueryParameter("q", "INT64", fiscal_quarter),
            ],
        ),
    ).result()


def delete_card(
    client: bigquery.Client, *, stock_code: str, as_of: date
) -> None:
    """``DELETE WHERE stock_code=... AND as_of=...`` — used before re-saving a
    card to keep (stock, as_of) unique."""
    tid = table_id("cards")
    client.query(
        f"DELETE FROM `{tid}` WHERE stock_code=@sc AND as_of=@as_of",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
            ],
        ),
    ).result()


def row_counts(client: bigquery.Client, table_names: Sequence[str] | None = None) -> dict[str, int]:
    """Diagnostic — ``COUNT(*)`` for each research table. Used by M1 verification."""
    names = list(table_names) if table_names is not None else list(TABLES.keys())
    out: dict[str, int] = {}
    for name in names:
        tid = table_id(name)
        result = list(client.query(f"SELECT COUNT(*) AS n FROM `{tid}`").result())
        out[name] = int(result[0]["n"])
    return out


def main() -> None:
    """Idempotently create / refresh all 6 research tables."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = get_bq_client()
    created = ensure_all_tables(client)
    for name, tid in created.items():
        log.info("table ready", extra={"table": name, "id": tid})


if __name__ == "__main__":
    main()
