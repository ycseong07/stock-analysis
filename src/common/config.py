"""Project configuration: static GCP infrastructure IDs + Secret Manager loading.

Hard rules (see CLAUDE.md):
- Secrets come from Secret Manager only — no .env fallback for secret values.
- This is the only module that reads from Secret Manager or hardcodes
  infrastructure identifiers. Other modules must call `get_infra()` and
  `get_secrets()` rather than reading env vars directly.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from google.cloud import secretmanager
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class GcpInfra(BaseSettings):
    """Static GCP infrastructure identifiers.

    Defaults are not sensitive and are committed. Override via env vars
    prefixed `DART_RAG_` or via a local `.env` (gitignored).
    """

    model_config = SettingsConfigDict(
        env_prefix="DART_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    project: str = "project-68bd3bce-668f-4ddd-af0"
    raw_bucket: str = "dart-rag-raw-project-68bd3bce-668f-4ddd-af0"
    bq_dataset: str = "dart_rag"
    bq_location: str = "US"
    region: str = "us-central1"


class Secrets(BaseModel):
    """Runtime secrets resolved once from Secret Manager."""

    model_config = ConfigDict(frozen=True)

    gemini_api_key: SecretStr
    dart_api_key: SecretStr


def _access_secret(secret_id: str, project: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8")


@lru_cache(maxsize=1)
def get_infra() -> GcpInfra:
    return GcpInfra()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    infra = get_infra()
    log.info("loading secrets from Secret Manager", extra={"project": infra.project})
    return Secrets(
        gemini_api_key=SecretStr(_access_secret("gemini-api-key", infra.project)),
        dart_api_key=SecretStr(_access_secret("dart-api-key", infra.project)),
    )


@lru_cache(maxsize=1)
def get_fred_api_key() -> SecretStr:
    """Lazy-loaded FRED API key.

    Kept out of `Secrets` so unrelated code paths don't fail when the
    `fred-api-key` secret is absent. Only the macro signal loader calls this.
    """
    infra = get_infra()
    return SecretStr(_access_secret("fred-api-key", infra.project))


class KrxCredentials(BaseModel):
    """KRX 정보데이터시스템 회원 로그인 (PyKRX uses KRX_ID / KRX_PW env vars)."""

    model_config = ConfigDict(frozen=True)

    krx_id: SecretStr
    krx_pw: SecretStr


@lru_cache(maxsize=1)
def get_krx_credentials() -> KrxCredentials:
    """Lazy-loaded KRX login. Required by PyKRX functions that hit member-only
    endpoints (trading value, short balance). Public OHLCV does not need this.

    Stored as a single Secret Manager entry ``krx-credentials`` whose payload
    is JSON ``{"id": "...", "pw": "..."}`` — keeps id+pw together (always
    rotated as a pair) and saves a slot under the 6-secret free-tier cap.
    """
    infra = get_infra()
    payload = json.loads(_access_secret("krx-credentials", infra.project))
    return KrxCredentials(
        krx_id=SecretStr(payload["id"]),
        krx_pw=SecretStr(payload["pw"]),
    )
