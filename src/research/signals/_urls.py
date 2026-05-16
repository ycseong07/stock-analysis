"""URL builders for fact_pack source attribution.

Display-only metadata attached to each ``FactItem.source_urls`` so the
frontend can render citations as clickable links pointing back at the
original document / chart. LLM nodes ignore these.
"""

from __future__ import annotations


def dart_viewer_url(rcept_no: str) -> str:
    """DART filing viewer (HTML, with the report's own TOC).

    Users land on the report root and click section headings (e.g.
    "Ⅲ. 재무에 관한 사항") themselves — page-precise anchors are not
    available without per-report parsing.
    """
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


def naver_finance_url(stock_code: str) -> str:
    """Naver Finance per-stock main page (chart + key metrics).

    Used for technical signals — there's no canonical "source document"
    for derived stats (MA crossover, volume ratio, etc.), so we point to
    the chart the user can use to reconstruct the calculation.
    """
    return f"https://finance.naver.com/item/main.naver?code={stock_code}"


def fred_series_url(series_id: str) -> str:
    """FRED series page (St. Louis Fed)."""
    return f"https://fred.stlouisfed.org/series/{series_id}"
