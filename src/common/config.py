"""Project configuration: static GCP infrastructure IDs + Secret Manager loading.

Hard rules (see CLAUDE.md):
- Secrets come from Secret Manager only — no .env fallback for secret values.
- This is the only module that reads from Secret Manager or hardcodes
  infrastructure identifiers. Other modules must call `get_infra()` and
  `get_secrets()` rather than reading env vars directly.
"""

from __future__ import annotations

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
