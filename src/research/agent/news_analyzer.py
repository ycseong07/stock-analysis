"""News analyser — sentiment + event extraction via Gemini Flash structured output.

plan.md M4 contract: one prompt per article emits both:
  1. **Sentiment**  — positive / neutral / negative + confidence + evidence quote.
  2. **Event**      — fixed enum (실적 / 계약 / 규제 / 소송 / 인사 / 제품출시 /
                      M&A / 기타) + entities + per-event sentiment polarity +
                      source quote.

The 8 covered tickers' Korean names are baked in so the prompt always names
the company in Korean (helps Gemini disambiguate in news copy).

Output is a Pydantic ``NewsAnalysis`` validated by ``response_schema``;
schema-violating Gemini responses raise ``ValidationError`` (caller decides
whether to retry).
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.common.config import get_secrets
from src.research.agent.news_clustering import ClusteredArticle

log = logging.getLogger(__name__)

FLASH_MODEL = "gemini-2.5-flash"

# Mirrors src/research/ingest/data_loader.STOCKS — kept in sync manually for now.
# Used only for prompt naming, not for filtering.
_STOCK_NAMES: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "네이버",
    "035720": "카카오",
    "068270": "셀트리온",
    "105560": "KB금융",
    "012450": "한화에어로스페이스",
}

Sentiment = Literal["positive", "neutral", "negative"]
EventType = Literal[
    "실적", "계약", "규제", "소송", "인사", "제품출시", "M&A", "기타"
]


class NewsAnalysis(BaseModel):
    """Gemini-flash structured output schema (per article).

    Note: cannot use ``extra="forbid"`` — Gemini's schema endpoint rejects
    ``additionalProperties: false`` (2026-05-09).
    """

    model_config = ConfigDict(frozen=True)

    sentiment: Sentiment
    sentiment_confidence: float = Field(ge=0.0, le=1.0)
    sentiment_evidence_quote: str
    event_type: EventType
    event_entities: list[str]
    event_sentiment_polarity: Sentiment
    event_source_quote: str


_SYSTEM_PROMPT = (
    "당신은 한국 주식 시장 뉴스를 분석하는 분석가다. "
    "주어진 기사를 읽고 정확히 명시된 JSON 스키마에 따라 응답하라. "
    "근거 인용은 반드시 기사 본문(제목 또는 요약)에 등장한 문구를 그대로 사용하라. "
    "추측하지 말고 기사가 실제로 말한 사실만 분류하라. "
    "event_type 은 8개 enum 중에서 가장 적합한 하나만 고른다."
)


def _build_prompt(stock_code: str, article: ClusteredArticle) -> str:
    name = _STOCK_NAMES.get(stock_code, stock_code)
    summary = (article.summary or "").strip() or "(요약 없음)"
    return (
        f"종목: {name} ({stock_code})\n"
        f"발행일: {article.published_at:%Y-%m-%d %H:%M}\n"
        f"매체: {article.source or '-'}\n"
        f"제목: {article.title}\n"
        f"요약:\n{summary}\n"
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False


class NewsAnalyzer:
    """Gemini Flash + structured output. One client per process, batchable
    over articles via ``analyze_many``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        model: str = FLASH_MODEL,
    ) -> None:
        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client(api_key=get_secrets().gemini_api_key.get_secret_value())
        self._model = model

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=4, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def analyze(self, stock_code: str, article: ClusteredArticle) -> NewsAnalysis:
        prompt = _build_prompt(stock_code, article)
        t0 = time.perf_counter()
        config = genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=NewsAnalysis,
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = resp.text or ""
        log.info(
            "news_analyze",
            extra={
                "stock_code": stock_code,
                "url_hash": article.url_hash,
                "in_chars": len(prompt),
                "out_chars": len(text),
                "latency_ms": latency_ms,
            },
        )
        # Gemini returns JSON text; ``resp.parsed`` would also work but we
        # prefer the explicit Pydantic round-trip for clearer errors.
        return NewsAnalysis.model_validate_json(text)

    def analyze_many(
        self, stock_code: str, articles: list[ClusteredArticle]
    ) -> list[tuple[ClusteredArticle, NewsAnalysis]]:
        """Sequential analyse — keeps within free-tier RPM and surfaces per-
        article failures cleanly."""
        out: list[tuple[ClusteredArticle, NewsAnalysis]] = []
        for art in articles:
            try:
                out.append((art, self.analyze(stock_code, art)))
            except Exception:
                log.exception(
                    "news_analyze_failed",
                    extra={"stock_code": stock_code, "url_hash": art.url_hash},
                )
        return out
