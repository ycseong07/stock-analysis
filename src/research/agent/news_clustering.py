"""News clustering — embed Naver headlines, group via DBSCAN, pick a
representative article per cluster.

plan.md M4 contract: "클러스터별 대표 기사 1 건만 다음 단계로 — 같은 사건
5번 반복 방지." Articles in the same cluster cover the same event; we pass
exactly one of them to the LLM analyser to avoid wasted Flash calls and
narrative repetition.

Pipeline:
  1. Fetch ``dart_rag.news`` rows for one stock × date range.
  2. Embed each ``title + summary`` via ``GeminiEmbedder`` (1번 wrapper).
     The shared sqlite cache (``data/emb_cache.db``) means re-runs hit cache
     and unique URL count == new API calls.
  3. DBSCAN with cosine distance (``eps=0.3``, ``min_samples=2``).
  4. Per cluster pick the earliest ``published_at`` as the representative.
     Noise points (``cluster_id == -1``) are kept as their own one-article
     "clusters" — we don't drop them, since a unique article can carry the
     only signal for that day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.cluster import DBSCAN

from src.common.config import get_secrets
from src.index.embed import EmbeddingCache, GeminiEmbedder
from src.research.ingest.bq import get_bq_client, table_id

log = logging.getLogger(__name__)

DEFAULT_EPS = 0.3
DEFAULT_MIN_SAMPLES = 2
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_CACHE_PATH = Path("data/emb_cache.db")


@dataclass(frozen=True)
class ClusteredArticle:
    """One article + its cluster assignment."""

    url_hash: str
    title: str
    link: str
    summary: str | None
    source: str | None
    published_at: pd.Timestamp
    cluster_id: int  # -1 = DBSCAN noise (treated as singleton cluster)


@dataclass(frozen=True)
class ClusterResult:
    stock_code: str
    as_of: date
    n_articles: int
    n_clusters: int
    n_noise: int
    representatives: list[ClusteredArticle]
    all_articles: list[ClusteredArticle]
    n_embedding_api_calls: int  # cache misses only


def _fetch_news(
    client: bigquery.Client, stock_code: str, start: date, end: date
) -> pd.DataFrame:
    tid = table_id("news")
    q = (
        f"SELECT url_hash, title, link, summary, source, published_at "
        f"FROM `{tid}` "
        f"WHERE stock_code=@sc AND DATE(published_at) BETWEEN @start AND @end "
        f"ORDER BY published_at ASC"
    )
    return client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("start", "DATE", start),
                bigquery.ScalarQueryParameter("end", "DATE", end),
            ],
        ),
    ).to_dataframe()


def _build_embedding_text(title: str, summary: str | None) -> str:
    """Concatenate title + summary for embedding. Title alone is too short
    for reliable cosine clustering on Korean."""
    body = (summary or "").strip()
    return f"{title.strip()}\n\n{body}" if body else title.strip()


def _pick_representative(group: list[ClusteredArticle]) -> ClusteredArticle:
    """Earliest published_at wins — first reporter usually framed the story."""
    return min(group, key=lambda a: a.published_at)


def cluster(
    stock_code: str,
    as_of: date,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    bq_client: bigquery.Client | None = None,
    embedder: GeminiEmbedder | None = None,
) -> ClusterResult:
    """Cluster recent news for one stock and return per-cluster representatives.

    Pass an existing ``embedder`` to share the sqlite cache across calls
    (recommended for batch backfill); otherwise one is constructed.
    """
    bq_client = bq_client or get_bq_client()
    df = _fetch_news(bq_client, stock_code, as_of - timedelta(days=lookback_days), as_of)
    if df.empty:
        return ClusterResult(
            stock_code=stock_code,
            as_of=as_of,
            n_articles=0,
            n_clusters=0,
            n_noise=0,
            representatives=[],
            all_articles=[],
            n_embedding_api_calls=0,
        )

    if embedder is None:
        cache = EmbeddingCache(cache_path) if cache_path else None
        embedder = GeminiEmbedder(
            api_key=get_secrets().gemini_api_key.get_secret_value(),
            cache=cache,
        )

    # Track cache size to compute API call count for verification
    cache_size_before = len(embedder._cache) if embedder._cache is not None else 0  # noqa: SLF001

    texts = [_build_embedding_text(t, s) for t, s in zip(df["title"], df["summary"], strict=True)]
    vectors = embedder.embed_documents(texts)
    cache_size_after = len(embedder._cache) if embedder._cache is not None else 0  # noqa: SLF001
    n_api_calls = cache_size_after - cache_size_before

    arr = np.array(vectors, dtype=np.float32)
    if len(arr) >= min_samples:
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(arr)
    else:
        # Too few articles for DBSCAN — every article is its own singleton.
        labels = np.full(len(arr), -1, dtype=int)

    articles = [
        ClusteredArticle(
            url_hash=str(row["url_hash"]),
            title=str(row["title"]),
            link=str(row["link"]),
            summary=(str(row["summary"]) if pd.notna(row["summary"]) else None),
            source=(str(row["source"]) if pd.notna(row["source"]) else None),
            published_at=row["published_at"],
            cluster_id=int(label),
        )
        for (_, row), label in zip(df.iterrows(), labels, strict=True)
    ]

    # Group by cluster_id; noise (-1) → each is its own singleton group
    groups: dict[int, list[ClusteredArticle]] = {}
    next_singleton_id = -1
    for art in articles:
        if art.cluster_id == -1:
            groups[next_singleton_id] = [art]
            next_singleton_id -= 1
        else:
            groups.setdefault(art.cluster_id, []).append(art)

    representatives = [_pick_representative(g) for g in groups.values()]
    representatives.sort(key=lambda a: a.published_at)

    n_noise = sum(1 for a in articles if a.cluster_id == -1)
    n_real_clusters = len({a.cluster_id for a in articles if a.cluster_id != -1})
    n_total_clusters = n_real_clusters + n_noise  # singletons count

    return ClusterResult(
        stock_code=stock_code,
        as_of=as_of,
        n_articles=len(articles),
        n_clusters=n_total_clusters,
        n_noise=n_noise,
        representatives=representatives,
        all_articles=articles,
        n_embedding_api_calls=n_api_calls,
    )


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    stock_code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    as_of_str = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    out = cluster(stock_code, date.fromisoformat(as_of_str))
    print(f"\n=== {stock_code} as_of={out.as_of} ===")
    print(f"  articles: {out.n_articles}")
    print(f"  clusters (incl. singletons): {out.n_clusters}")
    print(f"  DBSCAN noise (singletons): {out.n_noise}")
    print(f"  embedding API calls (cache misses): {out.n_embedding_api_calls}")
    print()
    print("--- representatives (earliest per cluster) ---")
    for art in out.representatives:
        cid = art.cluster_id if art.cluster_id != -1 else "noise"
        ts = f"{art.published_at:%Y-%m-%d %H:%M}"
        src = art.source or "-"
        print(f"  [{cid}] {ts} {src:<10} {art.title[:60]}")


if __name__ == "__main__":
    main()
