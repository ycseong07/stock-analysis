"""LangGraph wiring: classify → retrieve → synthesize → faithfulness_check (loop).

Conditional edge from `faithfulness_check`:
  - faithful=True  → END
  - faithful=False AND retry_count <= MAX_RETRIES → back to `synthesize`
  - otherwise → END (return draft as-is, log violations)
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from langgraph.graph import END, StateGraph

from src.agent import nodes
from src.agent.llm import ChatLLM, GeminiChat
from src.agent.state import AgentState
from src.common.config import get_infra, get_secrets
from src.index.embed import EmbeddingCache, GeminiEmbedder


def build_graph(
    *,
    llm: ChatLLM,
    bq_client: bigquery.Client,
    embedder: GeminiEmbedder,
    table_id: str,
) -> Any:
    """Compile the agent graph with the given dependencies bound."""
    g: StateGraph[AgentState] = StateGraph(AgentState)
    g.add_node("classify", partial(nodes.classify_query, llm=llm))
    g.add_node(
        "retrieve",
        partial(nodes.retrieve, bq_client=bq_client, embedder=embedder, table_id=table_id),
    )
    g.add_node("synthesize", partial(nodes.synthesize, llm=llm))
    g.add_node("faithfulness", partial(nodes.faithfulness_check, llm=llm))

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "faithfulness")
    g.add_conditional_edges(
        "faithfulness",
        nodes.should_retry,
        {"synthesize": "synthesize", "end": END},
    )
    return g.compile()


def build_default_graph(*, cache_path: Path | None = None) -> Any:
    """Build with project defaults (Secret Manager + meta.db cache)."""
    infra = get_infra()
    secrets = get_secrets()
    api_key = secrets.gemini_api_key.get_secret_value()
    bq = bigquery.Client(project=infra.project)
    embedder = GeminiEmbedder(
        api_key=api_key,
        cache=EmbeddingCache(cache_path) if cache_path else None,
    )
    llm = GeminiChat(api_key=api_key)
    table_id = f"{infra.project}.{infra.bq_dataset}.chunks"
    return build_graph(llm=llm, bq_client=bq, embedder=embedder, table_id=table_id)
