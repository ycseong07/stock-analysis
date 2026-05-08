"""BigQuery loader: chunks.jsonl on GCS → embeddings → `dart_rag.chunks` rows.

Per-report flow (idempotent):
  1. Read ``gs://<bucket>/_chunks/{rcept_no}.jsonl``.
  2. Embed each chunk's ``content`` via `GeminiEmbedder` (cache hits skip API).
  3. ``DELETE FROM dart_rag.chunks WHERE rcept_no = <r>`` — clears prior rows.
  4. Load new rows via a BigQuery **load job** (free; streaming insert is not).

Re-running with the same data is safe and ~free in Gemini cost (cache hits)
and BigQuery cost (load jobs are not metered).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from google.cloud import bigquery

from src.common.config import get_infra, get_secrets
from src.index.embed import EmbeddingCache, GeminiEmbedder, content_sha
from src.ingest.store import download_zip, open_meta_db
from src.parse.schemas import Chunk

log = logging.getLogger(__name__)

CHUNKS_TABLE = "chunks"
VECTOR_INDEX = "dart_rag_chunks_emb_idx"

_CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  chunk_id    STRING NOT NULL,
  rcept_no    STRING NOT NULL,
  corp_code   STRING NOT NULL,
  corp_name   STRING NOT NULL,
  fiscal_year INT64 NOT NULL,
  report_type STRING NOT NULL,
  section     STRING,
  is_table    BOOL,
  content     STRING NOT NULL,
  table_uri   STRING,
  content_sha STRING NOT NULL,
  embedding   ARRAY<FLOAT64>,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY RANGE_BUCKET(fiscal_year, GENERATE_ARRAY(2018, 2030, 1))
CLUSTER BY corp_code, report_type
"""

_CREATE_VECTOR_INDEX_DDL = """
CREATE VECTOR INDEX IF NOT EXISTS {index}
ON `{table}`(embedding)
OPTIONS(index_type='IVF', distance_type='COSINE')
"""


def _bq_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("rcept_no", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("corp_code", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("corp_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("fiscal_year", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("report_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("section", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("is_table", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("table_uri", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("content_sha", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]


@dataclass(frozen=True)
class LoadStats:
    rcept_no: str
    n_chunks: int
    n_embedded: int  # chunks that hit the Gemini API (cache misses)
    n_cached: int
    elapsed_ms: int


def ensure_table(client: bigquery.Client, *, project: str, dataset: str) -> str:
    """Create the chunks table if absent. Returns its full ID."""
    table_id = f"{project}.{dataset}.{CHUNKS_TABLE}"
    client.query(_CREATE_TABLE_DDL.format(table=table_id)).result()
    return table_id


def ensure_vector_index(client: bigquery.Client, *, table_id: str) -> None:
    """Create the IVF/COSINE vector index if absent."""
    client.query(_CREATE_VECTOR_INDEX_DDL.format(index=VECTOR_INDEX, table=table_id)).result()


def _read_chunks_jsonl(uri: str) -> list[Chunk]:
    raw = download_zip(uri).decode("utf-8")
    return [Chunk.model_validate_json(line) for line in raw.strip().split("\n") if line]


def _chunk_to_row(chunk: Chunk, sha: str, embedding: list[float]) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "rcept_no": chunk.rcept_no,
        "corp_code": chunk.corp_code,
        "corp_name": chunk.corp_name,
        "fiscal_year": chunk.fiscal_year,
        "report_type": chunk.report_type,
        "section": chunk.section,
        "is_table": chunk.is_table,
        "content": chunk.content,
        "table_uri": chunk.table_uri,
        "content_sha": sha,
        "embedding": embedding,
    }


def load_one(
    *,
    bq_client: bigquery.Client,
    table_id: str,
    bucket: str,
    rcept_no: str,
    embedder: GeminiEmbedder,
) -> LoadStats:
    """Load one report's chunks into BQ. Idempotent.

    Defensively filters out chunks with empty content (Gemini's embed_content
    rejects empty strings; defense in depth — chunker also drops these).
    """
    t0 = time.perf_counter()
    chunks = _read_chunks_jsonl(f"gs://{bucket}/_chunks/{rcept_no}.jsonl")
    chunks = [c for c in chunks if c.content.strip()]

    cache = embedder._cache  # noqa: SLF001  (intentional metric collection)
    pre_size = len(cache) if cache is not None else 0
    texts = [c.content for c in chunks]
    vectors = embedder.embed_documents(texts)
    post_size = len(cache) if cache is not None else 0
    n_embedded = post_size - pre_size
    n_cached = len(chunks) - n_embedded

    rows = [
        _chunk_to_row(chunk, content_sha(chunk.content), vec)
        for chunk, vec in zip(chunks, vectors, strict=True)
    ]

    bq_client.query(
        f"DELETE FROM `{table_id}` WHERE rcept_no = @rcept_no",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("rcept_no", "STRING", rcept_no),
            ]
        ),
    ).result()

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=_bq_schema(),
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    bq_client.load_table_from_json(rows, table_id, job_config=job_config).result()

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return LoadStats(
        rcept_no=rcept_no,
        n_chunks=len(chunks),
        n_embedded=n_embedded,
        n_cached=n_cached,
        elapsed_ms=elapsed_ms,
    )


def load_all(*, db_path: Path, cache_path: Path) -> list[LoadStats]:
    """Iterate every row in `filings`, load chunks into BQ.

    Builds BQ client + GeminiEmbedder from project config + Secret Manager.
    """
    infra = get_infra()
    bq_client = bigquery.Client(project=infra.project)
    table_id = ensure_table(bq_client, project=infra.project, dataset=infra.bq_dataset)

    cache = EmbeddingCache(cache_path)
    embedder = GeminiEmbedder(
        api_key=get_secrets().gemini_api_key.get_secret_value(),
        cache=cache,
    )

    all_stats: list[LoadStats] = []
    with open_meta_db(db_path) as conn:
        rows = list(conn.execute("SELECT rcept_no FROM filings ORDER BY rcept_no"))
        for row in rows:
            stats = load_one(
                bq_client=bq_client,
                table_id=table_id,
                bucket=infra.raw_bucket,
                rcept_no=row["rcept_no"],
                embedder=embedder,
            )
            log.info(
                "loaded",
                extra={
                    "rcept_no": stats.rcept_no,
                    "chunks": stats.n_chunks,
                    "embedded": stats.n_embedded,
                    "cached": stats.n_cached,
                    "elapsed_ms": stats.elapsed_ms,
                },
            )
            all_stats.append(stats)
    return all_stats
