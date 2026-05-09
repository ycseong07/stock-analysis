"""LangGraph wiring for the research-card agent (M5).

Topology (per plan.md M5):

    gather → bullish → bearish → balance ──→ write_card → faith → language → END
                                  │              ▲
                                  ├─ retry_bullish ─┤
                                  └─ retry_bearish ─┘

Conditional ``balance_router``:
  - bullish_points < 2  AND  not bullish_retried → retry_bullish
  - bearish_points < 2  AND  not bearish_retried → retry_bearish
  - else → write_card

faithfulness / language are linear (no retry) — failures land in state for
caller-side handling, mirroring 1번's MVP pattern.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from google import genai
from google.cloud import bigquery
from langgraph.graph import END, StateGraph

from src.common.config import get_secrets
from src.research.agent import nodes
from src.research.agent.news_analyzer import NewsAnalyzer
from src.research.agent.state import CardState
from src.research.ingest.bq import get_bq_client


def build_graph(
    *,
    bq_client: bigquery.Client,
    genai_client: genai.Client,
    news_analyzer: NewsAnalyzer,
) -> Any:
    """Compile the research-card graph with the given dependencies bound."""
    g: StateGraph[CardState] = StateGraph(CardState)

    g.add_node(
        "gather",
        partial(nodes.gather_signals, bq_client=bq_client, news_analyzer=news_analyzer),
    )
    g.add_node("bullish", partial(nodes.bullish_synthesizer, client=genai_client))
    g.add_node("bearish", partial(nodes.bearish_synthesizer, client=genai_client))
    g.add_node("balance", nodes.balance_check)
    g.add_node("retry_bullish", partial(nodes.retry_bullish, client=genai_client))
    g.add_node("retry_bearish", partial(nodes.retry_bearish, client=genai_client))
    g.add_node("write_card", partial(nodes.final_card_writer, client=genai_client))
    g.add_node("faith", partial(nodes.faithfulness_check, client=genai_client))
    g.add_node("language", nodes.language_check)

    g.set_entry_point("gather")
    g.add_edge("gather", "bullish")
    g.add_edge("bullish", "bearish")
    g.add_edge("bearish", "balance")
    g.add_conditional_edges(
        "balance",
        nodes.balance_router,
        {
            "retry_bullish": "retry_bullish",
            "retry_bearish": "retry_bearish",
            "write_card": "write_card",
        },
    )
    g.add_edge("retry_bullish", "balance")
    g.add_edge("retry_bearish", "balance")
    g.add_edge("write_card", "faith")
    g.add_edge("faith", "language")
    g.add_edge("language", END)

    return g.compile()


def build_default_graph() -> Any:
    """Build with project defaults (Secret Manager auth + project BQ)."""
    api_key = get_secrets().gemini_api_key.get_secret_value()
    bq_client = get_bq_client()
    genai_client = genai.Client(api_key=api_key)
    news_analyzer = NewsAnalyzer(client=genai_client)
    return build_graph(
        bq_client=bq_client,
        genai_client=genai_client,
        news_analyzer=news_analyzer,
    )


def main() -> None:
    """Smoke run — print the resulting card for one stock × one date."""
    import logging
    import sys
    from datetime import date

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stock_code = sys.argv[1] if len(sys.argv) > 1 else "005930"
    as_of_str = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    as_of = date.fromisoformat(as_of_str)

    graph = build_default_graph()
    final_state = graph.invoke({"stock_code": stock_code, "as_of": as_of})

    print("\n" + "=" * 60)
    print(f"Card for {stock_code} as_of={as_of}")
    print("=" * 60)
    print()
    print(final_state.get("card_markdown", "<no card>"))
    print()
    print("--- meta ---")
    print(f"  fact_pack    : {len(final_state.get('fact_pack', []))} items")
    print(f"  bullish      : {len(final_state.get('bullish_points', []))} points "
          f"(retried={final_state.get('bullish_retried', False)})")
    print(f"  bearish      : {len(final_state.get('bearish_points', []))} points "
          f"(retried={final_state.get('bearish_retried', False)})")
    print(f"  faithful     : {final_state.get('faithful')} "
          f"violations={final_state.get('faith_violations', [])}")
    print(f"  lang violations: {final_state.get('language_violations', [])}")


if __name__ == "__main__":
    main()
