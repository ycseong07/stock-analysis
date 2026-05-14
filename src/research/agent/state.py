"""LangGraph state for the research-card agent.

TypedDict (so LangGraph's StateGraph reducer treats updates as shallow
merges) with Pydantic-validated sub-objects when structure matters.

Lifecycle:
  inputs (stock_code, as_of)
    → gather_signals      (5 raw signal outputs + flat ``fact_pack``)
    → bullish_synthesizer (bullish_points)
    → bearish_synthesizer (bearish_points)
    → balance_check       (retry one synthesizer if < 2 points)
    → final_card_writer   (card_markdown)
    → faithfulness_check  (faithful + faith_violations)
    → language_check      (language_violations)
    → END
"""

from __future__ import annotations

from datetime import date
from typing import TypedDict

from pydantic import BaseModel, ConfigDict


class FactItem(BaseModel):
    """One factual sentence the synthesizers may cite.

    The ``id`` is a stable per-card identifier (e.g. ``tech-1``, ``flow-3``,
    ``news-2``) that downstream nodes embed in their output as the citation
    contract.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    source: str  # technical / flow / disclosure_financial / macro / news
    sentence: str


class BulletPoint(BaseModel):
    """One bullish or bearish point with its source citations."""

    model_config = ConfigDict(frozen=True)

    text: str
    source_ids: list[str]


class SynthesisOutput(BaseModel):
    """Gemini Pro structured output schema for bullish/bearish synthesis."""

    # Cannot use extra="forbid" — Gemini schema endpoint rejects
    # additionalProperties: false (M4 finding 2026-05-09).
    model_config = ConfigDict(frozen=True)

    points: list[BulletPoint]


class CardState(TypedDict, total=False):
    """Full LangGraph state."""

    # Inputs
    stock_code: str
    as_of: date

    # gather_signals output
    fact_pack: list[FactItem]
    # raw signal dumps (for downstream metadata access — not handed to LLMs directly)
    signal_dumps: dict[str, object]

    # synthesizers
    bullish_points: list[BulletPoint]
    bearish_points: list[BulletPoint]
    bullish_retried: bool
    bearish_retried: bool

    # final card
    card_markdown: str

    # checks
    faithful: bool
    faith_violations: list[str]
    language_violations: list[str]
