"""Load DART report zips into parsed lxml elements.

A DART zip contains:
  - {rcept_no}.xml         — the main report body (HTML-like XML).
  - {rcept_no}_NNNNN.xml   — zero or more attachments (e.g. 감사보고서).

Both are HTML-hybrid (not strict XML), so we use lxml's lenient parser.
Empirically this always succeeds on DART output.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from io import BytesIO

from lxml import etree

from src.ingest.store import download_zip

log = logging.getLogger(__name__)

PARSER = etree.XMLParser(recover=True, encoding="utf-8")


@dataclass(frozen=True)
class LoadedReport:
    """A parsed DART report: one main XML root plus zero or more attachments."""

    rcept_no: str
    main: etree._Element
    attachments: dict[str, etree._Element] = field(default_factory=dict)


def split_main_and_attachments(names: list[str]) -> tuple[str, list[str]]:
    """Pick the main XML name (no underscore in stem) from a zip name list.

    Raises `ValueError` if there is not exactly one such file.
    """
    candidates = [n for n in names if n.endswith(".xml") and "_" not in n.removesuffix(".xml")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly 1 main XML (no underscore), found: {candidates}")
    main = candidates[0]
    attachments = [n for n in names if n.endswith(".xml") and n != main]
    return main, attachments


def load_report_zip(zip_bytes: bytes) -> LoadedReport:
    """Parse a DART report zip into a `LoadedReport`."""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        main_name, attachment_names = split_main_and_attachments(zf.namelist())
        rcept_no = main_name.removesuffix(".xml")

        main = etree.fromstring(zf.read(main_name), PARSER)
        attachments: dict[str, etree._Element] = {
            name: etree.fromstring(zf.read(name), PARSER) for name in attachment_names
        }

    log.info(
        "report_loaded",
        extra={"rcept_no": rcept_no, "n_attachments": len(attachments)},
    )
    return LoadedReport(rcept_no=rcept_no, main=main, attachments=attachments)


def load_report_from_gcs(uri: str) -> LoadedReport:
    """Download a DART report zip from GCS and parse it."""
    return load_report_zip(download_zip(uri))
