"""LangGraph nodes for the research-card agent.

8 nodes total (per plan.md M5):
  gather_signals → bullish_synthesizer → bearish_synthesizer → balance_check
                 → final_card_writer → faithfulness_check → language_check

Each node receives the full ``CardState`` and returns a partial ``CardState``
update (LangGraph merges shallowly).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.cloud import bigquery
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.research.agent.news_analyzer import NewsAnalysis, NewsAnalyzer
from src.research.agent.news_clustering import ClusteredArticle
from src.research.agent.news_clustering import cluster as cluster_news
from src.research.agent.state import (
    BulletPoint,
    CardState,
    FactItem,
    SynthesisOutput,
)
from src.research.signals import disclosure_financial, flow, macro, technical
from src.research.signals._lint import find_violations

log = logging.getLogger(__name__)

PRO_MODEL = "gemini-2.5-pro"
FLASH_MODEL = "gemini-2.5-flash"

NEWS_LOOKBACK_DAYS = 7
MAX_NEWS_REPRESENTATIVES = 5
MIN_BULLISH_POINTS = 2
MIN_BEARISH_POINTS = 2
MAX_POINTS_PER_SIDE = 5

_SYSTEM_CARD = (
    "당신은 한국 주식 시장의 결정론적 사실 인풋을 분석하는 분석가다. "
    "절대 규칙: "
    "(1) 기술적 지표는 과거 거래량 변화 결과로만 해석. 향후 가격 방향 추정 금지. "
    "(2) 수급은 T-1, 공매도는 T-2 기준. '실시간' / '현재' 표현 금지. "
    "(3) 재무 지표는 가장 최근 보고서 기준이며 발표일 명시. "
    "(4) 거시 데이터는 컨텍스트로만. 종목 영향 단정 금지. "
    "모든 주장은 반드시 source_id 한 개 이상을 인용해야 한다."
)


def _is_retryable_genai(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False


# ---------------------------------------------------------------------------
# 1. gather_signals
# ---------------------------------------------------------------------------


def _news_to_sentence(art: ClusteredArticle, analysis: NewsAnalysis) -> str:
    src = art.source or "?"
    return (
        f"[{art.published_at:%Y-%m-%d} {src}] {art.title} — "
        f"event={analysis.event_type}, sentiment={analysis.sentiment} "
        f"(conf {analysis.sentiment_confidence:.2f}). "
        f'근거 인용: "{analysis.sentiment_evidence_quote}"'
    )


def gather_signals(
    state: CardState,
    *,
    bq_client: bigquery.Client,
    news_analyzer: NewsAnalyzer,
) -> dict[str, Any]:
    """Run all 5 signal nodes sequentially and build a flat ``fact_pack`` of
    ID-tagged sentences. Each ``id`` is the citation key downstream nodes
    must reference."""
    stock_code = state["stock_code"]
    as_of = state["as_of"]

    tech = technical.compute(stock_code, as_of, client=bq_client)
    flow_out = flow.compute(stock_code, as_of, client=bq_client)
    disc = disclosure_financial.compute(stock_code, as_of, client=bq_client)
    mac = macro.compute(as_of, client=bq_client)

    # News: cluster + analyze top representatives
    cluster_result = cluster_news(stock_code, as_of, lookback_days=NEWS_LOOKBACK_DAYS)
    news_pairs = news_analyzer.analyze_many(
        stock_code, cluster_result.representatives[:MAX_NEWS_REPRESENTATIVES]
    )

    def _urls_for(sig: Any, idx: int) -> list[str]:
        """sentence_urls[idx] when populated, else empty list."""
        urls = getattr(sig, "sentence_urls", None) or []
        return list(urls[idx]) if idx < len(urls) else []

    fact_pack: list[FactItem] = []
    for i, s in enumerate(tech.sentences, 1):
        fact_pack.append(
            FactItem(
                id=f"tech-{i}",
                source="technical",
                sentence=s,
                source_urls=_urls_for(tech, i - 1),
            )
        )
    for i, s in enumerate(flow_out.sentences, 1):
        fact_pack.append(
            FactItem(
                id=f"flow-{i}",
                source="flow",
                sentence=s,
                source_urls=_urls_for(flow_out, i - 1),
            )
        )
    for i, s in enumerate(disc.sentences, 1):
        fact_pack.append(
            FactItem(
                id=f"disc-{i}",
                source="disclosure_financial",
                sentence=s,
                source_urls=_urls_for(disc, i - 1),
            )
        )
    for i, s in enumerate(mac.sentences, 1):
        fact_pack.append(
            FactItem(
                id=f"macro-{i}",
                source="macro",
                sentence=s,
                source_urls=_urls_for(mac, i - 1),
            )
        )
    for i, (art, ana) in enumerate(news_pairs, 1):
        fact_pack.append(
            FactItem(
                id=f"news-{i}",
                source="news",
                sentence=_news_to_sentence(art, ana),
                source_urls=[art.link] if art.link else [],
            )
        )

    log.info(
        "gather_signals",
        extra={
            "stock_code": stock_code,
            "n_tech": len(tech.sentences),
            "n_flow": len(flow_out.sentences),
            "n_disc": len(disc.sentences),
            "n_macro": len(mac.sentences),
            "n_news": len(news_pairs),
            "fact_pack_total": len(fact_pack),
        },
    )

    return {
        "fact_pack": fact_pack,
        "signal_dumps": {
            "technical": tech.model_dump(),
            "flow": flow_out.model_dump(),
            "disclosure_financial": disc.model_dump(),
            "macro": mac.model_dump(),
            "news_cluster": {
                "n_articles": cluster_result.n_articles,
                "n_clusters": cluster_result.n_clusters,
                "n_analyses": len(news_pairs),
            },
        },
    }


# ---------------------------------------------------------------------------
# 2. bullish / bearish synthesizers
# ---------------------------------------------------------------------------


def _format_fact_pack(fact_pack: list[FactItem]) -> str:
    return "\n".join(f"- [{item.id}] {item.sentence}" for item in fact_pack)


def _build_synth_prompt(
    *, side: str, stock_code: str, fact_pack: list[FactItem], min_n: int
) -> str:
    side_kor = "긍정 / 매수" if side == "bullish" else "부정 / 리스크 / 매도"
    return (
        f"종목: {stock_code}\n"
        f"\n"
        f"다음은 결정론적 노드가 추출한 사실 인풋이다. 각 줄에 [id] 가 붙어 있다.\n"
        f"\n"
        f"## 사실 인풋\n"
        f"{_format_fact_pack(fact_pack)}\n"
        f"\n"
        f"## 작업\n"
        f"위 사실 인풋만 보고 종목의 **{side_kor}** 관점 근거를 최소 {min_n}개, "
        f"최대 {MAX_POINTS_PER_SIDE}개 추출하라. "
        f"각 근거는 반드시 사실 인풋의 source_id 한 개 이상을 인용해야 한다 "
        f"(없는 id 는 사용 금지). "
        f"새로운 사실을 추측하지 마라. 사실 인풋이 부족해 {min_n}개를 못 채우면 "
        f"채울 수 있는 만큼만 반환하라.\n"
    )


@retry(
    retry=retry_if_exception(_is_retryable_genai),
    wait=wait_exponential(multiplier=4, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _synthesize_side(
    *,
    client: genai.Client,
    side: str,
    stock_code: str,
    fact_pack: list[FactItem],
    min_n: int,
    model: str,
) -> SynthesisOutput:
    prompt = _build_synth_prompt(side=side, stock_code=stock_code, fact_pack=fact_pack, min_n=min_n)
    config = genai_types.GenerateContentConfig(
        system_instruction=_SYSTEM_CARD,
        response_mime_type="application/json",
        response_schema=SynthesisOutput,
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    return SynthesisOutput.model_validate_json(resp.text or "{}")


def _filter_invalid_citations(points: list[BulletPoint], valid_ids: set[str]) -> list[BulletPoint]:
    """Drop any source_id the model hallucinated; if a point has zero valid
    ids left, drop the point. We don't *fix* citations — we just refuse to
    propagate ones the fact_pack can't ground."""
    out: list[BulletPoint] = []
    for p in points:
        kept = [sid for sid in p.source_ids if sid in valid_ids]
        if kept:
            out.append(BulletPoint(text=p.text, source_ids=kept))
    return out


def bullish_synthesizer(
    state: CardState, *, client: genai.Client, model: str = PRO_MODEL
) -> dict[str, Any]:
    fact_pack = state["fact_pack"]
    valid_ids = {item.id for item in fact_pack}
    out = _synthesize_side(
        client=client,
        side="bullish",
        stock_code=state["stock_code"],
        fact_pack=fact_pack,
        min_n=MIN_BULLISH_POINTS,
        model=model,
    )
    points = _filter_invalid_citations(out.points, valid_ids)
    log.info(
        "bullish_synthesizer", extra={"stock_code": state["stock_code"], "n_points": len(points)}
    )
    return {"bullish_points": points}


def bearish_synthesizer(
    state: CardState, *, client: genai.Client, model: str = PRO_MODEL
) -> dict[str, Any]:
    fact_pack = state["fact_pack"]
    valid_ids = {item.id for item in fact_pack}
    out = _synthesize_side(
        client=client,
        side="bearish",
        stock_code=state["stock_code"],
        fact_pack=fact_pack,
        min_n=MIN_BEARISH_POINTS,
        model=model,
    )
    points = _filter_invalid_citations(out.points, valid_ids)
    log.info(
        "bearish_synthesizer", extra={"stock_code": state["stock_code"], "n_points": len(points)}
    )
    return {"bearish_points": points}


# ---------------------------------------------------------------------------
# 3. balance_check (conditional edge)
# ---------------------------------------------------------------------------


def balance_check(state: CardState) -> dict[str, Any]:
    """No-op state-wise; routing is decided by ``balance_router`` below."""
    return {}


def balance_router(state: CardState) -> str:
    """Return the next node name. Retries one side at most once."""
    bullish_n = len(state.get("bullish_points") or [])
    bearish_n = len(state.get("bearish_points") or [])
    bullish_retried = bool(state.get("bullish_retried"))
    bearish_retried = bool(state.get("bearish_retried"))

    if bullish_n < MIN_BULLISH_POINTS and not bullish_retried:
        return "retry_bullish"
    if bearish_n < MIN_BEARISH_POINTS and not bearish_retried:
        return "retry_bearish"
    return "write_card"


def retry_bullish(
    state: CardState, *, client: genai.Client, model: str = PRO_MODEL
) -> dict[str, Any]:
    """Re-run bullish_synthesizer with a stricter prompt, then mark retried."""
    out = bullish_synthesizer(state, client=client, model=model)
    out["bullish_retried"] = True
    return out


def retry_bearish(
    state: CardState, *, client: genai.Client, model: str = PRO_MODEL
) -> dict[str, Any]:
    out = bearish_synthesizer(state, client=client, model=model)
    out["bearish_retried"] = True
    return out


# ---------------------------------------------------------------------------
# 4. final_card_writer
# ---------------------------------------------------------------------------


def _format_points(points: list[BulletPoint]) -> str:
    return "\n".join(f"- {p.text} [{', '.join(p.source_ids)}]" for p in points)


def _build_card_prompt(state: CardState) -> str:
    bullish = state.get("bullish_points") or []
    bearish = state.get("bearish_points") or []
    fact_pack = state["fact_pack"]
    return (
        f"종목: {state['stock_code']}\n"
        f"분석 기준일: {state['as_of']}\n\n"
        f"## 매수 관점 (synthesizer 출력)\n{_format_points(bullish)}\n\n"
        f"## 매도/리스크 관점 (synthesizer 출력)\n{_format_points(bearish)}\n\n"
        f"## 사실 인풋 (참조용)\n{_format_fact_pack(fact_pack)}\n\n"
        f"## 작업\n"
        f"위 정보를 한국어 마크다운 리서치 카드로 합성하라.\n"
        f"섹션:\n"
        f"### 요약 (3 줄 이내, **각 줄 끝에 [source_ids] 인용 필수**)\n"
        f"### 매수 관점\n  - bullish points (각 줄 끝에 [source_ids] 인용 유지)\n"
        f"### 매도/리스크 관점\n  - bearish points (각 줄 끝에 [source_ids] 인용 유지)\n"
        f"### 거시 컨텍스트\n  - macro source_id 인용\n\n"
        f"규칙: 모든 사실 주장에 [source_ids] 부착 (요약 포함). "
        f"새 사실 추가 금지, source_ids 변경 금지, 미래 예측 단어 금지."
    )


@retry(
    retry=retry_if_exception(_is_retryable_genai),
    wait=wait_exponential(multiplier=4, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def final_card_writer(
    state: CardState, *, client: genai.Client, model: str = PRO_MODEL
) -> dict[str, Any]:
    prompt = _build_card_prompt(state)
    config = genai_types.GenerateContentConfig(system_instruction=_SYSTEM_CARD)
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    text = resp.text or ""
    log.info(
        "final_card_writer",
        extra={"stock_code": state["stock_code"], "out_chars": len(text)},
    )
    return {"card_markdown": text}


# ---------------------------------------------------------------------------
# 5. faithfulness_check
# ---------------------------------------------------------------------------


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _extract_cited_ids(card: str) -> list[str]:
    """Extract [id1, id2] tokens — split on ',' since the writer may group ids."""
    out: list[str] = []
    for match in _CITATION_RE.findall(card):
        for raw in match.split(","):
            tok = raw.strip()
            if tok:
                out.append(tok)
    return out


class _FaithJudgement(BaseModel):
    model_config = ConfigDict(frozen=True)
    faithful: bool
    violations: list[str]


def _build_faith_prompt(card: str, fact_pack: list[FactItem]) -> str:
    return (
        f"## 사실 인풋\n{_format_fact_pack(fact_pack)}\n\n"
        f"## 평가 대상 카드\n{card}\n\n"
        f"## 작업\n"
        f"카드의 각 줄에 부착된 [source_id] 인용이 사실 인풋과 의미상 일치하는지 검증하라.\n"
        f"  - source_id 가 실제 사실 인풋에 존재하지 않으면 violation.\n"
        f"  - source_id 는 존재하나 카드 줄의 주장이 그 sentence 와 의미상 어긋나면 violation.\n"
        f"  - 카드의 어떤 사실 주장에 source_id 가 아예 없으면 violation.\n"
        f"  - 거시 컨텍스트 / 요약은 macro source_id 가 있으면 OK.\n"
        f"violations 리스트에 위반한 source_id 또는 줄 요약을 적어라."
    )


@retry(
    retry=retry_if_exception(_is_retryable_genai),
    wait=wait_exponential(multiplier=4, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def faithfulness_check(
    state: CardState, *, client: genai.Client, model: str = FLASH_MODEL
) -> dict[str, Any]:
    card = state.get("card_markdown") or ""
    fact_pack = state["fact_pack"]

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_FaithJudgement,
    )
    resp = client.models.generate_content(
        model=model, contents=_build_faith_prompt(card, fact_pack), config=config
    )
    try:
        judge = _FaithJudgement.model_validate_json(resp.text or "{}")
    except ValidationError:
        log.warning("faithfulness_check parse failed", extra={"raw": resp.text})
        return {"faithful": False, "faith_violations": ["parse_error"]}

    log.info(
        "faithfulness_check",
        extra={
            "stock_code": state["stock_code"],
            "faithful": judge.faithful,
            "n_violations": len(judge.violations),
        },
    )
    return {"faithful": judge.faithful, "faith_violations": list(judge.violations)}


# ---------------------------------------------------------------------------
# 6. language_check
# ---------------------------------------------------------------------------


def language_check(state: CardState) -> dict[str, Any]:
    """Regex-only — re-uses the M3 _lint future-prediction word list."""
    card = state.get("card_markdown") or ""
    bullish_texts = [p.text for p in (state.get("bullish_points") or [])]
    bearish_texts = [p.text for p in (state.get("bearish_points") or [])]

    violations: list[str] = []
    for label, text in [
        ("card", card),
        *((f"bullish:{t[:40]}", t) for t in bullish_texts),
        *((f"bearish:{t[:40]}", t) for t in bearish_texts),
    ]:
        for line in text.splitlines() if isinstance(text, str) else [text]:
            found = find_violations(line)
            if found:
                violations.append(f"{label} → {found}: {line[:80]}")

    log.info(
        "language_check",
        extra={"stock_code": state["stock_code"], "n_violations": len(violations)},
    )
    return {"language_violations": violations}


# ---------------------------------------------------------------------------
# Diagnostics — used by tests / eyeball
# ---------------------------------------------------------------------------


def state_to_json(state: CardState) -> str:
    """Serialize CardState for logging / debugging."""

    def default(o: object) -> object:
        if isinstance(o, BaseModel):
            return o.model_dump()
        return str(o)

    return json.dumps(state, default=default, ensure_ascii=False, indent=2)
