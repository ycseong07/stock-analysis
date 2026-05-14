"""FastAPI app for the AI 투자 리서치 에이전트.

Endpoints:
  GET  /                              — static HTML UI (research card workspace)
  GET  /health                        — readiness probe (always 200 once lifespan ran;
                                        /healthz is reserved by Google's frontend).
  GET  /stocks                        — 8 covered tickers
  POST /research/{stock_code}         — generate or fetch a cached research card
  GET  /research/{stock_code}/history — recent cards from ``dart_rag.cards``

The LangGraph research graph is built once in the lifespan handler and stored
on ``app.state.research_graph`` so handlers reuse it (BQ + Gemini clients are
expensive to instantiate per request).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.research.agent.graph import build_default_graph as build_research_graph
from src.serve.research_routes import create_research_router
from src.serve.types import HealthResponse

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("serve_startup")
        app.state.research_graph = build_research_graph()
        yield
        log.info("serve_shutdown")

    app = FastAPI(title="AI 투자 리서치 에이전트", lifespan=lifespan)
    app.include_router(create_research_router())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # /healthz is reserved by Google's frontend on Cloud Run; use /health.
        return HealthResponse(status="ok")

    return app


# Module-level app for `uvicorn src.serve.app:app` (production entry).
app = create_app()
