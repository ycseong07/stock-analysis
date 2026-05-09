"""Pydantic base for deterministic signal outputs.

Every signal node returns a model derived from ``SignalOutput`` so that:
  - ``as_of`` and ``data_freshness`` are always present (M1 contract).
  - ``sentences`` (LLM-input natural language) are validated past-tense via
    ``_lint`` — any future-prediction phrasing fails fast.
  - signal-specific raw fields are added by subclasses.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator

from src.research.signals._lint import find_violations


class SignalOutput(BaseModel):
    """Base for all deterministic signal nodes.

    Derived classes add their own raw fields. ``stock_code`` is None for
    stock-agnostic signals (e.g. ``macro_context``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stock_code: str | None
    as_of: date
    data_freshness: str
    sentences: list[str]

    @field_validator("sentences")
    @classmethod
    def _no_future_words(cls, v: list[str]) -> list[str]:
        all_violations: list[tuple[str, list[str]]] = []
        for s in v:
            found = find_violations(s)
            if found:
                all_violations.append((s, found))
        if all_violations:
            raise ValueError(
                "future-prediction phrasing in signal sentences: " + "; ".join(
                    f"{s!r} → {labels}" for s, labels in all_violations
                )
            )
        return v
