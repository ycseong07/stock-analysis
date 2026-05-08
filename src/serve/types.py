"""Pydantic models for the HTTP API.

Kept separate from the agent's internal `AgentState` so we can change the wire
format independently of the graph state."""

from __future__ import annotations

from pydantic import BaseModel


class CitationOut(BaseModel):
    """One citation tag as a structured object (parsed from the draft text)."""

    corp_name: str
    fiscal_year: int
    report_type: str
    section: str
    chunk_id: str


class AskResponse(BaseModel):
    """POST /ask response."""

    answer: str
    citations: list[CitationOut]
    faithful: bool
    hits: int
    retry_count: int
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
