"""DART corp-code master loader.

The DART corpCode endpoint returns a zip containing CORPCODE.xml — a flat
list of every entity DART tracks (~100k rows, ~4 MB). This module:

  1. Fetches the zip via DartClient (or reads a local cache if fresh).
  2. Parses CORPCODE.xml into `CorpInfo` objects.
  3. Provides exact-match lookup by corp name, scoped to listed corps.

The cache lives at `~/.cache/dart-rag/corp_code.zip` by default and
refreshes weekly. Override `cache_dir` for tests.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import BaseModel

from src.ingest.dart_client import DartClient

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dart-rag"
_CACHE_FILENAME = "corp_code.zip"
_CACHE_TTL_SECONDS = 7 * 24 * 3600


class CorpInfo(BaseModel):
    corp_code: str
    corp_name: str
    stock_code: str | None = None
    modify_date: str


class CorpNotFoundError(LookupError):
    """No DART entity matches the requested name."""

    def __init__(self, name: str):
        super().__init__(f"No DART corp matches name={name!r}")
        self.name = name


class AmbiguousCorpNameError(LookupError):
    """Multiple listed DART entities share the requested name."""

    def __init__(self, name: str, candidates: list[str]):
        super().__init__(f"Multiple DART corps match name={name!r}: corp_codes={candidates}")
        self.name = name
        self.candidates = candidates


def load_corp_codes(
    client: DartClient,
    *,
    cache_dir: Path | None = None,
    ttl_seconds: int = _CACHE_TTL_SECONDS,
) -> list[CorpInfo]:
    """Return all DART corps. Uses local cache; refetches when stale or missing."""
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_file = cache_dir / _CACHE_FILENAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    fresh = cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < ttl_seconds
    if fresh:
        log.info("corp_codes_cache_hit", extra={"path": str(cache_file)})
        zip_bytes = cache_file.read_bytes()
    else:
        log.info("corp_codes_fetch", extra={"path": str(cache_file)})
        zip_bytes = client.fetch_corp_code_zip()
        cache_file.write_bytes(zip_bytes)

    return _parse_corp_zip(zip_bytes)


def _parse_corp_zip(zip_bytes: bytes) -> list[CorpInfo]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf, zf.open("CORPCODE.xml") as f:
        tree = ET.parse(f)
    out: list[CorpInfo] = []
    for el in tree.iter("list"):
        stock_code = (el.findtext("stock_code") or "").strip()
        out.append(
            CorpInfo(
                corp_code=(el.findtext("corp_code") or "").strip(),
                corp_name=(el.findtext("corp_name") or "").strip(),
                stock_code=stock_code or None,
                modify_date=(el.findtext("modify_date") or "").strip(),
            )
        )
    return out


def find_listed_corp(corps: list[CorpInfo], name: str) -> CorpInfo:
    """Return the *listed* corp (has stock_code) whose name exactly matches.

    Raises `CorpNotFoundError` when nothing matches, or
    `AmbiguousCorpNameError` when multiple listed entities share the name.
    """
    matches = [c for c in corps if c.corp_name == name and c.stock_code]
    if not matches:
        raise CorpNotFoundError(name)
    if len(matches) > 1:
        raise AmbiguousCorpNameError(name, [m.corp_code for m in matches])
    return matches[0]
