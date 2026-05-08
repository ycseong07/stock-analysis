# syntax=docker/dockerfile:1.7
# Multi-stage build using uv for fast, deterministic installs.
# Final image is python:3.11-slim with the project + deps only.

FROM python:3.11-slim AS builder

# uv from the official image (small, statically linked).
COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first (cache-friendly: changes to src/ don't bust this layer).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/

# Now install the project itself (entry into the venv).
RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml /app/pyproject.toml

# Cloud Run sends SIGTERM and waits up to 10s for graceful shutdown.
# uvicorn handles SIGTERM by default.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT}"]
