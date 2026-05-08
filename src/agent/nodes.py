"""LangGraph nodes — pure functions of `(state, deps) → state-update`.

Deps (BQ client, Gemini chat, embedder, table_id) are injected via
`functools.partial` in `graph.py` so nodes themselves stay easy to unit test.
"""

from __future__ import annotations

import json
import logging
import re

from google.cloud import bigquery

from src.agent.citation import format_citation, parse_citations
from src.agent.llm import ChatLLM
from src.agent.prompts import CLASSIFY_SYSTEM, FAITHFUL_SYSTEM, SYNTH_SYSTEM
from src.agent.state import AgentState, FilterSpec
from src.index.embed import GeminiEmbedder
from src.retrieve.hybrid import RetrievedChunk, hybrid_retrieve

log = logging.getLogger(__name__)

MAX_RETRIES = 1  # one retry is enough; faithfulness rarely flips on a 2nd try
TOP_N = 8


def _strip_json(text: str) -> str:
    """Best-effort: peel ```json fences and grab the first {...} block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def classify_query(state: AgentState, *, llm: ChatLLM) -> AgentState:
    """Use Flash to extract optional corp/year/report_type filters from the query."""
    raw = llm.flash(state["query"], system=CLASSIFY_SYSTEM)
    filters: FilterSpec = {}
    try:
        parsed = json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        log.warning("classify_unparseable", extra={"raw": raw[:200]})
        parsed = {}
    if isinstance(parsed.get("corp_code"), str):
        filters["corp_code"] = parsed["corp_code"]
    if isinstance(parsed.get("fiscal_year"), int):
        filters["fiscal_year"] = parsed["fiscal_year"]
    if isinstance(parsed.get("report_type"), str) and parsed["report_type"] in (
        "사업",
        "반기",
        "분기",
    ):
        filters["report_type"] = parsed["report_type"]
    return {"route": "text", "filters": filters}


def retrieve(
    state: AgentState,
    *,
    bq_client: bigquery.Client,
    embedder: GeminiEmbedder,
    table_id: str,
) -> AgentState:
    """Hybrid (vector + BM25) retrieval, filtered by classifier output.

    Preserves any existing hits (e.g. user-uploaded synthetic chunks) and
    appends BQ retrieval results after them.
    """
    f = state.get("filters") or {}
    bq_hits = hybrid_retrieve(
        state["query"],
        bq_client=bq_client,
        embedder=embedder,
        table_id=table_id,
        top_n=TOP_N,
        corp_code=f.get("corp_code"),
        fiscal_year=f.get("fiscal_year"),
        report_type=f.get("report_type"),
    )
    existing = list(state.get("hits") or [])
    return {"hits": existing + bq_hits}


def _format_hits_block(hits: list[RetrievedChunk]) -> str:
    """Render hits as a numbered context block for the synthesizer."""
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        tag = format_citation(h)
        blocks.append(f"[{i}] {tag}\n{h.content}")
    return "\n\n".join(blocks)


def synthesize(state: AgentState, *, llm: ChatLLM) -> AgentState:
    """Pro generates the answer with inline citation tags."""
    hits = state.get("hits") or []
    if not hits:
        return {"draft": "주어진 자료에서 확인되지 않습니다."}
    prompt = (
        f"질문: {state['query']}\n\n"
        f"검색된 청크:\n{_format_hits_block(hits)}\n\n"
        f"위 청크만 근거로 답하고, 모든 사실 진술 끝에 인용 태그를 붙이세요."
    )
    draft = llm.pro(prompt, system=SYNTH_SYSTEM)
    return {"draft": draft}


def _split_claims(draft: str) -> list[tuple[str, list[str]]]:
    """Split draft into ``(sentence, [chunk_id, ...])`` pairs.

    A "sentence" here is a chunk of text up to its trailing citation tags.
    Sentences without a citation tag are skipped (nothing to verify).
    """
    pairs: list[tuple[str, list[str]]] = []
    cursor = 0
    pattern = re.compile(r"((?:\[corp:[^\]]+\])+)")
    for match in pattern.finditer(draft):
        sentence = draft[cursor : match.start()].strip()
        if sentence:
            cited = [c.chunk_id for c in parse_citations(match.group(1))]
            if cited:
                pairs.append((sentence, cited))
        cursor = match.end()
    return pairs


def _claim_supported_by_chunk(*, llm: ChatLLM, claim: str, chunk_content: str) -> tuple[bool, str]:
    raw = llm.flash(
        f"주장:\n{claim}\n\n청크 본문:\n{chunk_content}",
        system=FAITHFUL_SYSTEM,
    )
    try:
        parsed = json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        log.warning("faithful_unparseable", extra={"raw": raw[:200]})
        return False, "judge_unparseable"
    return bool(parsed.get("supported")), str(parsed.get("reason", ""))


def faithfulness_check(state: AgentState, *, llm: ChatLLM) -> AgentState:
    """Verify each cited claim against its chunk; loop back if any unsupported."""
    draft = state.get("draft") or ""
    hits_by_id = {h.chunk_id: h for h in (state.get("hits") or [])}
    pairs = _split_claims(draft)
    if not pairs:
        return {
            "faithful": True,
            "faith_violations": [],
            "retry_count": state.get("retry_count", 0),
        }

    violations: list[str] = []
    for claim, cited_ids in pairs:
        if not any(
            _claim_supported_by_chunk(
                llm=llm,
                claim=claim,
                chunk_content=hits_by_id[cid].content,
            )[0]
            for cid in cited_ids
            if cid in hits_by_id
        ):
            violations.extend(cid for cid in cited_ids if cid in hits_by_id)

    return {
        "faithful": not violations,
        "faith_violations": violations,
        "retry_count": state.get("retry_count", 0) + (0 if not violations else 1),
    }


def should_retry(state: AgentState) -> str:
    """LangGraph conditional edge: retry synthesis if unfaithful and under cap."""
    if state.get("faithful", True):
        return "end"
    if state.get("retry_count", 0) > MAX_RETRIES:
        return "end"
    return "synthesize"
