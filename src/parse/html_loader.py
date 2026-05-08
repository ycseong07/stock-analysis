"""Load DART filing HTML (web-viewer format) from GCS into lxml trees."""

from __future__ import annotations

import logging

from lxml import html as lxml_html
from lxml.html import HtmlElement

from src.ingest.store import download_zip

log = logging.getLogger(__name__)


def load_html(html_text: str) -> HtmlElement:
    """Parse an HTML document into an lxml tree."""
    return lxml_html.fromstring(html_text)


def load_html_from_gcs(uri: str) -> HtmlElement:
    """Download an HTML document from GCS and parse it."""
    text = download_zip(uri).decode("utf-8")
    log.info("html_loaded", extra={"uri": uri, "bytes": len(text)})
    return load_html(text)
