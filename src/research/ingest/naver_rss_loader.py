"""Naver Finance per-stock news loader.

Endpoint: ``https://m.stock.naver.com/api/news/stock/{code}?pageSize=N``

Returns a list of article-cluster groups; each cluster's ``items[]`` holds
the article record. The legacy ``rss.naver.com`` host is deprecated (DNS no
longer resolves), so we hit the JSON API directly with ``requests`` —
``feedparser`` is unused for now but kept available in case Naver ever
restores an RSS feed.

Output rows match the ``news`` BQ schema. ``url_hash`` is a stable 16-char
sha256 prefix of ``mobileNewsUrl`` for idempotent dedup.
"""

from __future__ import annotations

import hashlib
import html
import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from src.research.ingest.schemas import FRESHNESS_T0

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
NEWS_API = "https://m.stock.naver.com/api/news/stock/{code}"
USER_AGENT = "Mozilla/5.0 (compatible; dart-rag-research/0.1)"
DEFAULT_PAGE_SIZE = 30


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _parse_kst_datetime(s: str) -> datetime:
    """``YYYYMMDDHHMM`` (KST) → tz-aware datetime."""
    return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=KST)


def load_news(
    stock_code: str,
    date_range: tuple[date, date],
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Fetch recent news articles for one stock and filter to ``date_range``.

    The Naver API does not document a date filter, so we always pull the
    latest ``page_size`` articles and discard those outside the range
    client-side. For 30-minute polling this is fine; for first backfill,
    raise ``page_size`` (e.g. 100).
    """
    response = requests.get(
        NEWS_API.format(code=stock_code),
        params={"pageSize": page_size},
        timeout=10,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    clusters = response.json()
    if not isinstance(clusters, list):
        log.warning("unexpected naver news shape", extra={"stock_code": stock_code})
        return []

    start, end = date_range
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=KST)
    end_dt = datetime.combine(end, datetime.max.time(), tzinfo=KST)
    now_iso = datetime.now(KST).isoformat()

    rows: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    for cluster in clusters:
        for item in cluster.get("items", []):
            link = item.get("mobileNewsUrl")
            if not link or link in seen_links:
                continue
            try:
                published_at = _parse_kst_datetime(item["datetime"])
            except (KeyError, ValueError):
                log.warning(
                    "skipping article with bad datetime",
                    extra={"stock_code": stock_code, "id": item.get("id")},
                )
                continue
            if not (start_dt <= published_at <= end_dt):
                continue
            seen_links.add(link)
            title = html.unescape(item.get("titleFull") or item.get("title") or "")
            body = html.unescape(item.get("body") or "")
            rows.append(
                {
                    "stock_code": stock_code,
                    "url_hash": _hash(link),
                    "title": title,
                    "link": link,
                    "published_at": published_at.isoformat(),
                    "source": item.get("officeName"),
                    "summary": (body[:500] or None),
                    "as_of": now_iso,
                    "data_freshness": FRESHNESS_T0,
                }
            )
    return rows
