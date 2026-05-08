"""Gemini chat client for the agent (Flash for cheap calls, Pro for synthesis).

Wraps `google-genai` with the same retry/backoff policy as `GeminiEmbedder`.
Two model tiers per the routing strategy in plan.md:

  - Flash (`gemini-2.5-flash`)   — query classification, light judgements.
  - Pro   (`gemini-2.5-pro`)     — final synthesis + faithfulness check.

Both methods share one `genai.Client` (HTTPS connection pool reuse).
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-pro"


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 5xx and 429 (rate limit); pass other 4xx through."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False


class ChatLLM(Protocol):
    """Surface area the agent nodes depend on. Lets tests inject stubs."""

    def flash(self, prompt: str, *, system: str | None = None) -> str: ...

    def pro(self, prompt: str, *, system: str | None = None) -> str: ...


class GeminiChat:
    """Synchronous Gemini chat client with retry."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        flash_model: str = FLASH_MODEL,
        pro_model: str = PRO_MODEL,
    ):
        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            raise ValueError("Either `api_key` or `client` is required.")
        self._flash_model = flash_model
        self._pro_model = pro_model

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=4, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _generate(self, *, model: str, prompt: str, system: str | None) -> str:
        t0 = time.perf_counter()
        config = genai_types.GenerateContentConfig(system_instruction=system) if system else None
        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = resp.text or ""
        log.info(
            "gemini_chat",
            extra={
                "model": model,
                "in_chars": len(prompt),
                "out_chars": len(text),
                "latency_ms": latency_ms,
            },
        )
        return text

    def flash(self, prompt: str, *, system: str | None = None) -> str:
        return self._generate(model=self._flash_model, prompt=prompt, system=system)

    def pro(self, prompt: str, *, system: str | None = None) -> str:
        return self._generate(model=self._pro_model, prompt=prompt, system=system)
