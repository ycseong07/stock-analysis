"""FastAPI app for the DART RAG agent.

Endpoints:
  GET  /          — static HTML UI (centered search box + file attach).
  GET  /health    — readiness probe (always 200; lifespan must have run).
                    Note: /healthz is reserved by Google's frontend on Cloud Run.
  POST /ask       — multipart form: ``query`` (str) + optional ``file``
                    (PDF or HTML). Response: AskResponse.

The LangGraph graph is built once in the lifespan handler and stored on
`app.state.graph` so request handlers reuse it. BQ client + Gemini client
are expensive to instantiate per request.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.agent.citation import parse_citations
from src.agent.graph import build_default_graph
from src.serve.types import AskResponse, CitationOut, HealthResponse
from src.serve.upload import (
    MAX_UPLOAD_CHARS,
    SUPPORTED_TYPES,
    extract_text,
    make_upload_chunk,
)

log = logging.getLogger(__name__)


GraphFactory = Callable[[], Any]

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


def create_app(
    *,
    graph_factory: GraphFactory | None = None,
    cache_path: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app. `graph_factory` is injectable for tests."""
    factory = graph_factory or (lambda: build_default_graph(cache_path=cache_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("serve_startup")
        app.state.graph = factory()
        yield
        log.info("serve_shutdown")

    app = FastAPI(title="DART RAG", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # /healthz is reserved by Google's frontend on Cloud Run; use /health.
        return HealthResponse(status="ok")

    @app.post("/ask", response_model=AskResponse)
    async def ask(
        request: Request,
        query: Annotated[str, Form(min_length=1, max_length=2000)],
        file: Annotated[UploadFile | None, File()] = None,
    ) -> AskResponse:
        t0 = time.perf_counter()

        initial_state: dict[str, Any] = {"query": query}
        if file is not None and file.filename:
            raw = await file.read()
            ct = file.content_type or ""
            ext = file.filename.lower()
            if ct not in SUPPORTED_TYPES and not ext.endswith((".pdf", ".html", ".htm")):
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type: {ct or ext}",
                )
            text = extract_text(file.filename, ct, raw)
            if text.strip():
                upload_chunk = make_upload_chunk(
                    filename=file.filename,
                    text=text,
                    year=datetime.now().year,
                )
                initial_state["hits"] = [upload_chunk]
                log.info(
                    "upload_attached",
                    extra={
                        "filename": file.filename,
                        "content_type": ct,
                        "raw_bytes": len(raw),
                        "extracted_chars": len(text),
                        "truncated": len(text) >= MAX_UPLOAD_CHARS,
                    },
                )

        result = request.app.state.graph.invoke(initial_state)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        draft = result.get("draft", "")
        citations = [
            CitationOut(
                corp_name=c.corp_name,
                fiscal_year=c.fiscal_year,
                report_type=c.report_type,
                section=c.section,
                chunk_id=c.chunk_id,
            )
            for c in parse_citations(draft)
        ]
        log.info(
            "ask",
            extra={
                "query_len": len(query),
                "answer_len": len(draft),
                "citations": len(citations),
                "faithful": bool(result.get("faithful")),
                "hits": len(result.get("hits") or []),
                "latency_ms": latency_ms,
            },
        )
        return AskResponse(
            answer=draft,
            citations=citations,
            faithful=bool(result.get("faithful")),
            hits=len(result.get("hits") or []),
            retry_count=int(result.get("retry_count", 0)),
            latency_ms=latency_ms,
        )

    return app


# Module-level app for `uvicorn src.serve.app:app` (production entry).
app = create_app()
