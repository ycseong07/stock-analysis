"""HTTP server: FastAPI wrapping the LangGraph research-card agent.

`create_app()` builds the FastAPI instance; the LangGraph graph is built once
at lifespan startup and reused across requests (BQ client + Gemini client are
expensive to construct).
"""
