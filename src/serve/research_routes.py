"""FastAPI router for the research-card endpoints.

Mounted onto the main ``src/serve/app.py`` app — same Cloud Run service,
same SA. Endpoints:

  GET  /                               — static HTML page (research workspace)
  GET  /stocks                         — 8 covered tickers
  POST /research/{stock_code}          — generate or fetch cached card
  GET  /research/{stock_code}/history  — past cards from `dart_rag.cards`

Caching: ``cards`` table holds one row per (stock_code, as_of). POST checks
the cache first; ``?force_refresh=true`` re-runs the graph and overwrites.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from google.cloud import bigquery
from pydantic import BaseModel, ConfigDict

from src.research.ingest.bq import delete_card, get_bq_client, load_rows, table_id
from src.research.ingest.data_loader import STOCKS
from src.research.ingest.schemas import TABLES

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML_PATH = _STATIC_DIR / "index.html"


# --- response models -------------------------------------------------------


class StockOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    name: str


class FactItemOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    source: str
    sentence: str


class CardResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    stock_code: str
    stock_name: str
    as_of: date
    card_markdown: str
    fact_pack: list[FactItemOut]
    bullish_count: int
    bearish_count: int
    faithful: bool
    language_violations_count: int
    cached: bool
    generated_at: str | None
    latency_ms: int


class CardSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    as_of: date
    bullish_count: int
    bearish_count: int
    faithful: bool
    generated_at: str | None


class HistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    stock_code: str
    cards: list[CardSummary]


# --- BQ helpers (cards-specific) -------------------------------------------


def _fetch_cached_card(
    client: bigquery.Client, stock_code: str, as_of: date
) -> dict[str, Any] | None:
    tid = table_id("cards")
    rows = list(
        client.query(
            f"SELECT * FROM `{tid}` "
            f"WHERE stock_code=@sc AND as_of=@as_of "
            f"ORDER BY generated_at DESC LIMIT 1",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                    bigquery.ScalarQueryParameter("as_of", "DATE", as_of),
                ],
            ),
        ).result()
    )
    return dict(rows[0]) if rows else None


def _save_card(
    client: bigquery.Client, *, stock_code: str, as_of: date, state: dict[str, Any]
) -> None:
    """Idempotent — DELETE existing (stock, as_of) then APPEND."""
    fact_pack = state.get("fact_pack") or []
    fact_pack_payload = [
        {"id": f.id, "source": f.source, "sentence": f.sentence} for f in fact_pack
    ]
    delete_card(client, stock_code=stock_code, as_of=as_of)
    load_rows(
        client,
        TABLES["cards"],
        [
            {
                "stock_code": stock_code,
                "as_of": as_of.isoformat(),
                "card_markdown": state.get("card_markdown") or "",
                "bullish_count": len(state.get("bullish_points") or []),
                "bearish_count": len(state.get("bearish_points") or []),
                "faithful": bool(state.get("faithful", False)),
                "language_violations_count": len(state.get("language_violations") or []),
                "fact_pack_json": json.dumps(fact_pack_payload, ensure_ascii=False),
                "signal_dumps_json": json.dumps(
                    state.get("signal_dumps") or {}, ensure_ascii=False, default=str
                ),
            }
        ],
    )


def _row_to_card_response(
    row: dict[str, Any], stock_code: str, *, cached: bool, latency_ms: int
) -> CardResponse:
    fact_pack_raw = row.get("fact_pack_json") or "[]"
    fact_pack = [FactItemOut(**item) for item in json.loads(fact_pack_raw)]
    generated_at = row.get("generated_at")
    return CardResponse(
        stock_code=stock_code,
        stock_name=STOCKS.get(stock_code, stock_code),
        as_of=row["as_of"],
        card_markdown=row.get("card_markdown") or "",
        fact_pack=fact_pack,
        bullish_count=int(row.get("bullish_count") or 0),
        bearish_count=int(row.get("bearish_count") or 0),
        faithful=bool(row.get("faithful", False)),
        language_violations_count=int(row.get("language_violations_count") or 0),
        cached=cached,
        generated_at=str(generated_at) if generated_at else None,
        latency_ms=latency_ms,
    )


# --- router ----------------------------------------------------------------


GraphFactory = Callable[[], Any]


def create_research_router(
    *,
    bq_client: bigquery.Client | None = None,
    research_graph_attr: str = "research_graph",
) -> APIRouter:
    """Build the research router. ``research_graph_attr`` is the FastAPI
    app.state attribute the POST handler reads to invoke the graph — letting
    tests inject a stub graph."""
    bq_client = bq_client or get_bq_client()
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def index_page() -> HTMLResponse:
        if not _INDEX_HTML_PATH.exists():
            raise HTTPException(status_code=500, detail="index.html missing")
        return HTMLResponse(_INDEX_HTML_PATH.read_text(encoding="utf-8"))

    @router.get("/stocks", response_model=list[StockOut])
    def list_stocks() -> list[StockOut]:
        return [StockOut(code=c, name=n) for c, n in STOCKS.items()]

    @router.post("/research/{stock_code}", response_model=CardResponse)
    def generate_card(
        request: Request,
        stock_code: str,
        as_of: Annotated[date | None, Query()] = None,
        force_refresh: Annotated[bool, Query()] = False,
    ) -> CardResponse:
        if stock_code not in STOCKS:
            raise HTTPException(
                status_code=404,
                detail=f"Unsupported stock_code: {stock_code}; "
                f"see GET /stocks for the 8 covered tickers.",
            )
        as_of = as_of or date.today()

        # Cache hit
        if not force_refresh:
            row = _fetch_cached_card(bq_client, stock_code, as_of)
            if row is not None:
                return _row_to_card_response(row, stock_code, cached=True, latency_ms=0)

        # Cache miss → run graph
        graph = getattr(request.app.state, research_graph_attr, None)
        if graph is None:
            raise HTTPException(status_code=503, detail="research_graph not initialised")
        t0 = time.perf_counter()
        try:
            final_state = graph.invoke({"stock_code": stock_code, "as_of": as_of})
        except Exception as exc:
            log.exception("graph_invoke_failed", extra={"stock_code": stock_code})
            raise HTTPException(status_code=500, detail=f"graph error: {exc}") from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)

        _save_card(bq_client, stock_code=stock_code, as_of=as_of, state=final_state)
        row = _fetch_cached_card(bq_client, stock_code, as_of)
        if row is None:
            raise HTTPException(status_code=500, detail="cache write failed")

        log.info(
            "research_card_generated",
            extra={
                "stock_code": stock_code,
                "as_of": str(as_of),
                "latency_ms": latency_ms,
                "faithful": bool(final_state.get("faithful")),
                "bullish": len(final_state.get("bullish_points") or []),
                "bearish": len(final_state.get("bearish_points") or []),
            },
        )
        return _row_to_card_response(row, stock_code, cached=False, latency_ms=latency_ms)

    @router.get("/research/{stock_code}/history", response_model=HistoryResponse)
    def card_history(
        stock_code: str,
        limit: Annotated[int, Query(ge=1, le=60)] = 10,
    ) -> HistoryResponse:
        if stock_code not in STOCKS:
            raise HTTPException(status_code=404, detail=f"Unsupported stock_code: {stock_code}")
        tid = table_id("cards")
        rows = client_query_history(bq_client, tid, stock_code, limit)
        return HistoryResponse(stock_code=stock_code, cards=rows)

    return router


def client_query_history(
    bq_client: bigquery.Client, tid: str, stock_code: str, limit: int
) -> list[CardSummary]:
    rows = bq_client.query(
        f"SELECT as_of, bullish_count, bearish_count, faithful, generated_at "
        f"FROM `{tid}` WHERE stock_code=@sc "
        f"ORDER BY as_of DESC LIMIT @lim",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sc", "STRING", stock_code),
                bigquery.ScalarQueryParameter("lim", "INT64", limit),
            ],
        ),
    ).result()
    return [
        CardSummary(
            as_of=r["as_of"],
            bullish_count=int(r["bullish_count"] or 0),
            bearish_count=int(r["bearish_count"] or 0),
            faithful=bool(r["faithful"]),
            generated_at=str(r["generated_at"]) if r["generated_at"] else None,
        )
        for r in rows
    ]
