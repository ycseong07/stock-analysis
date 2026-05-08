"""Extract text from a user-uploaded PDF or HTML and wrap as a `RetrievedChunk`.

Design choice (MVP): the uploaded file becomes a single synthetic chunk that
the agent's synthesizer sees alongside BQ-retrieved chunks. The synthesizer
can cite either source. Faithfulness check verifies upload citations against
the synthetic chunk's content.

Limitations:
  - Truncated to MAX_UPLOAD_CHARS (≈ 8 K chars). Larger files lose the tail.
  - No embedding / similarity ranking — single chunk always passed in.
  - Phase 2: chunk + embed + retrieve over the upload (per `25-serving.md`).
"""

from __future__ import annotations

import io
import logging
from dataclasses import replace

import pdfplumber
from lxml import html as lxml_html

from src.retrieve.hybrid import RetrievedChunk

log = logging.getLogger(__name__)

MAX_UPLOAD_CHARS = 8000
SUPPORTED_TYPES = {"application/pdf", "text/html"}


def extract_text(filename: str, content_type: str | None, raw: bytes) -> str:
    """Best-effort text extraction from PDF or HTML bytes.

    Returns truncated text (≤ MAX_UPLOAD_CHARS). Falls back to UTF-8 decode if
    the content_type / extension is unrecognized.
    """
    name = filename.lower()
    ct = (content_type or "").lower()

    if ct == "application/pdf" or name.endswith(".pdf"):
        text = _extract_pdf(raw)
    elif ct in ("text/html", "application/xhtml+xml") or name.endswith((".html", ".htm")):
        text = _extract_html(raw)
    else:
        # Last resort — treat as plain text. Don't crash on binary; just truncate.
        text = raw.decode("utf-8", errors="replace")

    return text[:MAX_UPLOAD_CHARS]


def _extract_pdf(raw: bytes) -> str:
    out: list[str] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                out.append(page_text)
    return "\n\n".join(out)


def _extract_html(raw: bytes) -> str:
    # Decode UTF-8 first; lxml's default fallback charset is latin-1 which
    # mangles Korean text when the HTML lacks an explicit <meta charset>.
    text_in = raw.decode("utf-8", errors="replace")
    root = lxml_html.fromstring(text_in)
    # Remove scripts / styles before grabbing text.
    for el in root.iter("script", "style"):
        el.getparent().remove(el)
    text = root.text_content() or ""
    # Normalize whitespace (collapse runs).
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def make_upload_chunk(*, filename: str, text: str, year: int) -> RetrievedChunk:
    """Wrap extracted text as a `RetrievedChunk` so the agent treats it like a hit.

    Synthetic metadata uses `corp_name="업로드"`, `report_type="첨부"`,
    `chunk_id="upload://{filename}"` so citations are visually distinct.
    `year` must be a 4-digit integer — required by the citation regex.
    """
    return RetrievedChunk(
        chunk_id=f"upload://{filename}",
        rcept_no="upload",
        corp_code="upload",
        corp_name="업로드",
        fiscal_year=year,
        report_type="첨부",
        section=filename,
        is_table=False,
        content=text,
        table_uri=None,
        vector_distance=None,
        vector_rank=None,
        bm25_rank=None,
        rrf_score=0.0,
    )


def with_filename(chunk: RetrievedChunk, filename: str) -> RetrievedChunk:
    """Return a copy with `section` updated to the supplied filename."""
    return replace(chunk, section=filename)
