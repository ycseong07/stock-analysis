"""BigQuery schemas + DDL for the research card pipeline (M1).

Six tables under `dart_rag.*`:
  - prices             — wide OHLCV+volume+value, partition by trade_date, GC 90d.
  - flows              — long format (foreign/inst/individual/short_balance), GC 90d.
  - disclosures_market — 1 row per filing, GC 90d.
  - financials         — long format (eps/per/pbr/roe/debt_ratio/op_income), no GC.
  - news               — 1 row per article (url_hash unique), GC 90d.
  - macro              — 1 row per (series_id, observation_date), no GC.

Shared contract on every table:
  - ``as_of TIMESTAMP NOT NULL`` — when the upstream considers this datum effective.
  - ``data_freshness STRING NOT NULL`` — one of the FRESHNESS_* constants below.
  - ``ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()`` — load time.

GC policy (decided 2026-05-09): rolling 90-day partition expiration via DDL
``OPTIONS(partition_expiration_days = 90)`` for the high-volume daily tables.
``financials`` and ``macro`` are unbounded — quarterly/monthly cadences are
small and the time series is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.cloud import bigquery

# --- data_freshness constants (kept here so loaders / signal nodes share the same vocabulary) ---

FRESHNESS_T0 = "T-0"            # confirmed end-of-day (prices after market close, news at fetch)
FRESHNESS_T1 = "T-1"            # one business day lag (flows: foreign/inst/individual net buys)
FRESHNESS_T2 = "T-2"            # two business day lag (short balance)
FRESHNESS_INTRADAY = "intraday"  # fetched during market hours; not yet confirmed
FRESHNESS_MONTHLY = "monthly"    # macro series with monthly cadence
FRESHNESS_WEEKLY = "weekly"      # macro series with weekly cadence
FRESHNESS_DAILY = "daily"        # macro series with daily cadence (e.g. DFF)
FRESHNESS_REPORT = "report"      # financials / disclosures tied to a specific filing date


@dataclass(frozen=True)
class TableSpec:
    name: str
    ddl: str
    schema: list[bigquery.SchemaField]


# --- prices: wide OHLCV (T-0 after market close, intraday otherwise) ---

PRICES_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  stock_code      STRING NOT NULL,
  trade_date      DATE NOT NULL,
  open            FLOAT64,
  high            FLOAT64,
  low             FLOAT64,
  close           FLOAT64,
  volume          INT64,
  value           INT64,
  as_of           TIMESTAMP NOT NULL,
  data_freshness  STRING NOT NULL,
  ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY trade_date
CLUSTER BY stock_code
OPTIONS(partition_expiration_days = 90)
"""

PRICES_SCHEMA = [
    bigquery.SchemaField("stock_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("trade_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("open", "FLOAT64"),
    bigquery.SchemaField("high", "FLOAT64"),
    bigquery.SchemaField("low", "FLOAT64"),
    bigquery.SchemaField("close", "FLOAT64"),
    bigquery.SchemaField("volume", "INT64"),
    bigquery.SchemaField("value", "INT64"),
    bigquery.SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_freshness", "STRING", mode="REQUIRED"),
]


# --- flows: long format. metric ∈ {foreign_net, institution_net, individual_net, short_balance} ---

FLOWS_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  stock_code      STRING NOT NULL,
  trade_date      DATE NOT NULL,
  metric          STRING NOT NULL,
  value           FLOAT64,
  as_of           TIMESTAMP NOT NULL,
  data_freshness  STRING NOT NULL,
  ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY trade_date
CLUSTER BY stock_code, metric
OPTIONS(partition_expiration_days = 90)
"""

FLOWS_SCHEMA = [
    bigquery.SchemaField("stock_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("trade_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("metric", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_freshness", "STRING", mode="REQUIRED"),
]


# --- disclosures_market: 1 row per filing (rcept_no unique) ---

DISCLOSURES_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  stock_code      STRING NOT NULL,
  rcept_no        STRING NOT NULL,
  rcept_dt        DATE NOT NULL,
  report_type     STRING NOT NULL,
  title           STRING NOT NULL,
  category        STRING,
  as_of           TIMESTAMP NOT NULL,
  data_freshness  STRING NOT NULL,
  ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY rcept_dt
CLUSTER BY stock_code, report_type
OPTIONS(partition_expiration_days = 90)
"""

DISCLOSURES_SCHEMA = [
    bigquery.SchemaField("stock_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("rcept_no", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("rcept_dt", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("report_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("category", "STRING"),
    bigquery.SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_freshness", "STRING", mode="REQUIRED"),
]


# --- financials: long format, quarterly time series, no expiration ---

FINANCIALS_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  stock_code      STRING NOT NULL,
  fiscal_year     INT64 NOT NULL,
  fiscal_quarter  INT64 NOT NULL,
  metric          STRING NOT NULL,
  value           FLOAT64,
  yoy_change      FLOAT64,
  report_date     DATE NOT NULL,
  as_of           TIMESTAMP NOT NULL,
  data_freshness  STRING NOT NULL,
  ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY report_date
CLUSTER BY stock_code, metric
"""

FINANCIALS_SCHEMA = [
    bigquery.SchemaField("stock_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("fiscal_year", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("fiscal_quarter", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("metric", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("yoy_change", "FLOAT64"),
    bigquery.SchemaField("report_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_freshness", "STRING", mode="REQUIRED"),
]


# --- news: 1 row per article, dedup by url_hash, GC 90d ---

NEWS_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  stock_code      STRING NOT NULL,
  url_hash        STRING NOT NULL,
  title           STRING NOT NULL,
  link            STRING NOT NULL,
  published_at    TIMESTAMP NOT NULL,
  source          STRING,
  summary         STRING,
  as_of           TIMESTAMP NOT NULL,
  data_freshness  STRING NOT NULL,
  ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(published_at)
CLUSTER BY stock_code
OPTIONS(partition_expiration_days = 90)
"""

NEWS_SCHEMA = [
    bigquery.SchemaField("stock_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("url_hash", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("link", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("published_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("summary", "STRING"),
    bigquery.SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_freshness", "STRING", mode="REQUIRED"),
]


# --- macro: 1 row per (series_id, observation_date), unbounded ---

MACRO_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  series_id        STRING NOT NULL,
  observation_date DATE NOT NULL,
  value            FLOAT64,
  change_1m        FLOAT64,
  change_3m        FLOAT64,
  as_of            TIMESTAMP NOT NULL,
  data_freshness   STRING NOT NULL,
  ingested_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY observation_date
CLUSTER BY series_id
"""

MACRO_SCHEMA = [
    bigquery.SchemaField("series_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("observation_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("change_1m", "FLOAT64"),
    bigquery.SchemaField("change_3m", "FLOAT64"),
    bigquery.SchemaField("as_of", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_freshness", "STRING", mode="REQUIRED"),
]


TABLES: dict[str, TableSpec] = {
    "prices": TableSpec(name="prices", ddl=PRICES_DDL, schema=PRICES_SCHEMA),
    "flows": TableSpec(name="flows", ddl=FLOWS_DDL, schema=FLOWS_SCHEMA),
    "disclosures_market": TableSpec(
        name="disclosures_market", ddl=DISCLOSURES_DDL, schema=DISCLOSURES_SCHEMA
    ),
    "financials": TableSpec(name="financials", ddl=FINANCIALS_DDL, schema=FINANCIALS_SCHEMA),
    "news": TableSpec(name="news", ddl=NEWS_DDL, schema=NEWS_SCHEMA),
    "macro": TableSpec(name="macro", ddl=MACRO_DDL, schema=MACRO_SCHEMA),
}
