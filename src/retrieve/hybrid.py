"""Hybrid retrieval over the BigQuery `dart_rag.chunks` table.

Two ranked lists merged via Reciprocal Rank Fusion:
  1. Dense  — `VECTOR_SEARCH` (cosine on the 768-dim embedding).
  2. Sparse — `SEARCH(content, query)` boolean lexical match.

Optional metadata filters (corp_code, fiscal_year, report_type) are applied
**before** VECTOR_SEARCH via a CTE so the requested `k_each` always falls
inside the requested scope.

The final `top_n` is fetched back from BQ with full chunk metadata and
returned as `RetrievedChunk` objects ordered by RRF score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.cloud import bigquery

from src.index.embed import GeminiEmbedder

log = logging.getLogger(__name__)

RRF_K = 60  # standard Cormack-Clarke-Buettcher RRF constant


@dataclass(frozen=True)
class RetrievedChunk:
    """One result from hybrid retrieval, ranked by RRF."""

    chunk_id: str
    rcept_no: str
    corp_code: str
    corp_name: str
    fiscal_year: int
    report_type: str
    section: str
    is_table: bool
    content: str
    table_uri: str | None
    vector_distance: float | None  # cosine distance from query (None if BM25-only)
    vector_rank: int | None  # 1-based rank in vector list
    bm25_rank: int | None  # 1-based rank in lexical list
    rrf_score: float


def reciprocal_rank_fusion(
    *ranked_lists: list[str],
    k: int = RRF_K,
) -> dict[str, float]:
    """Per-id RRF score: sum over lists of `1 / (k + rank)`."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _build_filters(
    corp_code: str | None,
    fiscal_year: int | None,
    report_type: str | None,
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Return ``(where_body, params)``. Empty body means no filter."""
    clauses: list[str] = []
    params: list[bigquery.ScalarQueryParameter] = []
    if corp_code:
        clauses.append("corp_code = @corp_code")
        params.append(bigquery.ScalarQueryParameter("corp_code", "STRING", corp_code))
    if fiscal_year:
        clauses.append("fiscal_year = @fiscal_year")
        params.append(bigquery.ScalarQueryParameter("fiscal_year", "INT64", fiscal_year))
    if report_type:
        clauses.append("report_type = @report_type")
        params.append(bigquery.ScalarQueryParameter("report_type", "STRING", report_type))
    return " AND ".join(clauses), params


def _vector_source(table_id: str, where_body: str) -> str:
    """Return the 1st-argument expression for `VECTOR_SEARCH`.

    No filter → ``TABLE \`...\``` (uses any vector index on the base table).
    With filter → ``(SELECT * FROM \`...\` WHERE ...)`` subquery.
    """
    if where_body:
        return f"(SELECT * FROM `{table_id}` WHERE {where_body})"
    return f"TABLE `{table_id}`"


def _vector_search(
    bq_client: bigquery.Client,
    *,
    table_id: str,
    qvec: list[float],
    k: int,
    where_body: str,
    filter_params: list[bigquery.ScalarQueryParameter],
) -> list[tuple[str, float]]:
    sql = f"""
    SELECT base.chunk_id AS chunk_id, distance
    FROM VECTOR_SEARCH(
      {_vector_source(table_id, where_body)},
      'embedding',
      (SELECT @qv AS embedding),
      top_k => @k,
      distance_type => 'COSINE'
    )
    ORDER BY distance ASC
    """
    params = [
        *filter_params,
        bigquery.ArrayQueryParameter("qv", "FLOAT64", qvec),
        bigquery.ScalarQueryParameter("k", "INT64", k),
    ]
    job = bq_client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    return [(row.chunk_id, float(row.distance)) for row in job.result()]


def _bm25_query_expr(query: str) -> str:
    """Convert a natural-language query into a `SEARCH()` expression.

    BigQuery `SEARCH(text, query)` defaults to AND of all query tokens —
    too strict for natural-language Korean. We split on whitespace, drop
    1-character tokens (조사 noise), and OR-join the rest.
    """
    tokens = [t for t in query.split() if len(t) >= 2]
    if not tokens:
        return query
    return " OR ".join(f'"{t}"' for t in tokens)


def _bm25_search(
    bq_client: bigquery.Client,
    *,
    table_id: str,
    query: str,
    k: int,
    where_body: str,
    filter_params: list[bigquery.ScalarQueryParameter],
) -> list[str]:
    where = "SEARCH(content, @q)"
    if where_body:
        where += f" AND ({where_body})"
    sql = f"""
    SELECT chunk_id
    FROM `{table_id}`
    WHERE {where}
    LIMIT @k
    """
    params = [
        *filter_params,
        bigquery.ScalarQueryParameter("q", "STRING", _bm25_query_expr(query)),
        bigquery.ScalarQueryParameter("k", "INT64", k),
    ]
    job = bq_client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    return [row.chunk_id for row in job.result()]


def _fetch_chunks_by_id(
    bq_client: bigquery.Client,
    *,
    table_id: str,
    chunk_ids: list[str],
) -> dict[str, dict[str, object]]:
    if not chunk_ids:
        return {}
    sql = f"""
    SELECT chunk_id, rcept_no, corp_code, corp_name, fiscal_year, report_type,
           section, is_table, content, table_uri
    FROM `{table_id}`
    WHERE chunk_id IN UNNEST(@ids)
    """
    job = bq_client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ids", "STRING", chunk_ids),
            ]
        ),
    )
    return {row.chunk_id: dict(row.items()) for row in job.result()}


def hybrid_retrieve(
    query: str,
    *,
    bq_client: bigquery.Client,
    embedder: GeminiEmbedder,
    table_id: str,
    top_n: int = 10,
    k_each: int = 20,
    corp_code: str | None = None,
    fiscal_year: int | None = None,
    report_type: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid (vector + BM25) retrieval with RRF merge.

    Filters are applied as a CTE pre-filter; pass `None` to skip.
    """
    qvec = embedder.embed_query(query)
    where_body, filter_params = _build_filters(corp_code, fiscal_year, report_type)

    vec_hits = _vector_search(
        bq_client,
        table_id=table_id,
        qvec=qvec,
        k=k_each,
        where_body=where_body,
        filter_params=filter_params,
    )
    bm25_hits = _bm25_search(
        bq_client,
        table_id=table_id,
        query=query,
        k=k_each,
        where_body=where_body,
        filter_params=filter_params,
    )

    vec_ids = [cid for cid, _ in vec_hits]
    rrf = reciprocal_rank_fusion(vec_ids, bm25_hits)
    top_ids = sorted(rrf, key=lambda c: -rrf[c])[:top_n]

    rows = _fetch_chunks_by_id(bq_client, table_id=table_id, chunk_ids=top_ids)
    vec_dist = dict(vec_hits)
    vec_rank = {cid: i + 1 for i, cid in enumerate(vec_ids)}
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_hits)}

    out: list[RetrievedChunk] = []
    for cid in top_ids:
        row = rows.get(cid)
        if row is None:
            log.warning("hybrid_missing_chunk", extra={"chunk_id": cid})
            continue
        out.append(
            RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                rcept_no=str(row["rcept_no"]),
                corp_code=str(row["corp_code"]),
                corp_name=str(row["corp_name"]),
                fiscal_year=int(str(row["fiscal_year"])),
                report_type=str(row["report_type"]),
                section=str(row["section"]) if row["section"] else "",
                is_table=bool(row["is_table"]),
                content=str(row["content"]),
                table_uri=str(row["table_uri"]) if row["table_uri"] else None,
                vector_distance=vec_dist.get(cid),
                vector_rank=vec_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
                rrf_score=rrf[cid],
            )
        )

    log.info(
        "hybrid_retrieve",
        extra={
            "query_len": len(query),
            "vec_hits": len(vec_hits),
            "bm25_hits": len(bm25_hits),
            "merged": len(rrf),
            "returned": len(out),
        },
    )
    return out
