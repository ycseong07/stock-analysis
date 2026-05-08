"""Gemini `embedding-001` client with content-hash cache.

Two responsibilities:
  1. Call Gemini's embedding endpoint in batches with retry/backoff.
  2. Cache vectors locally by SHA256 of the input text — re-embedding
     is the most likely free-tier blower, so this is mandatory.

Auth: pass `api_key` (loaded from Secret Manager via src.common.config)
*or* an already-constructed `genai.Client` (for testing).

Retry policy:
  - 5xx server errors: retry up to 5 times with exponential backoff (4–60s).
  - 429 rate limit: same retry policy.
  - Other 4xx: fail immediately (auth / bad request — retry won't help).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 100
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Conservative RPM ceiling for the free tier on gemini-embedding-001.
# Tune via `rpm_limit` if the actual quota is higher. 0 disables throttling.
DEFAULT_RPM_LIMIT = 60


def content_sha(text: str) -> str:
    """SHA-256 hex digest of a UTF-8-encoded string. Cache key for embeddings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 5xx and 429; pass through everything else."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    content_sha TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    embedded_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class EmbeddingCache:
    """SQLite-backed cache from `sha256(content)` to embedding vector."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_many(self, shas: list[str]) -> dict[str, list[float]]:
        if not shas:
            return {}
        placeholders = ",".join("?" * len(shas))
        rows = self._conn.execute(
            f"SELECT content_sha, vector_json FROM embeddings "
            f"WHERE content_sha IN ({placeholders})",
            shas,
        ).fetchall()
        return {sha: json.loads(vec_json) for sha, vec_json in rows}

    def put(self, sha: str, vector: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (content_sha, dim, vector_json) " "VALUES (?, ?, ?)",
            (sha, len(vector), json.dumps(vector)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(n)


class GeminiEmbedder:
    """Synchronous Gemini embedder with batching, retry, and SHA-keyed cache."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        cache: EmbeddingCache | None = None,
        dim: int = EMBED_DIM,
        batch_size: int = BATCH_SIZE,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
    ):
        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            raise ValueError("Either `api_key` or `client` is required.")
        self._cache = cache
        self._dim = dim
        self._batch_size = batch_size
        self._min_interval_s = 60.0 / rpm_limit if rpm_limit > 0 else 0.0
        self._last_call_t: float | None = None

    def _throttle(self) -> None:
        if self._min_interval_s <= 0 or self._last_call_t is None:
            return
        wait = self._min_interval_s - (time.perf_counter() - self._last_call_t)
        if wait > 0:
            time.sleep(wait)

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        self._throttle()
        try:
            return self._embed_batch_call(texts, task_type)
        finally:
            self._last_call_t = time.perf_counter()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=4, max=120),
        stop=stop_after_attempt(7),
        reraise=True,
    )
    def _embed_batch_call(self, texts: list[str], task_type: str) -> list[list[float]]:
        t0 = time.perf_counter()
        resp = self._client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,  # type: ignore[arg-type]
            config={
                "output_dimensionality": self._dim,
                "task_type": task_type,
            },
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        embeddings = resp.embeddings or []
        vectors = [list(e.values or []) for e in embeddings]
        log.info(
            "gemini_embed",
            extra={
                "model": EMBED_MODEL,
                "n": len(texts),
                "dim": self._dim,
                "task": task_type,
                "latency_ms": latency_ms,
            },
        )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (uses cache + RETRIEVAL_DOCUMENT task type)."""
        return self._embed_with_cache(texts, task_type=TASK_DOCUMENT)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (no cache; RETRIEVAL_QUERY task type)."""
        return self._embed_batch([query], task_type=TASK_QUERY)[0]

    def _embed_with_cache(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        shas = [content_sha(t) for t in texts]
        cached = self._cache.get_many(shas) if self._cache is not None else {}
        out: list[list[float] | None] = [cached.get(sha) for sha in shas]

        misses = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(misses), self._batch_size):
            batch_idx = misses[start : start + self._batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            vecs = self._embed_batch(batch_texts, task_type=task_type)
            for i, v in zip(batch_idx, vecs, strict=True):
                out[i] = v
                if self._cache is not None:
                    self._cache.put(shas[i], v)

        # All slots filled by this point.
        return [v for v in out if v is not None]
