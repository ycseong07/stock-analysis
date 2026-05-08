"""Agent graph state.

Kept as a TypedDict so LangGraph's StateGraph reducer treats updates as
shallow merges. Pydantic-validated views are produced on demand.
"""

from __future__ import annotations

from typing import TypedDict

from src.retrieve.hybrid import RetrievedChunk


class FilterSpec(TypedDict, total=False):
    """Optional retrieval filters parsed from the query."""

    corp_code: str
    fiscal_year: int
    report_type: str  # "사업" | "반기" | "분기"


class AgentState(TypedDict, total=False):
    """The full state passed between LangGraph nodes."""

    # Inputs
    query: str

    # classify_query output
    route: str  # "text" (only branch wired in MVP; "table"/"compare" later)
    filters: FilterSpec

    # retrieve output
    hits: list[RetrievedChunk]

    # synthesize output
    draft: str  # the model's answer with inline citation tags

    # faithfulness_check output
    faithful: bool
    faith_violations: list[str]  # citation chunk_ids whose claims couldn't be verified
    retry_count: int
