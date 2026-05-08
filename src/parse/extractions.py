"""Extract structured metadata from DART `<EXTRACTION ACODE='...'>` tags.

Each DART report embeds 20+ standardized values (e.g. TOT_ASSETS, TOT_SALES,
SUPV_OPIN, IFRS_YN) via `<EXTRACTION>` tags inside the main and/or
attachment XMLs. We collect them into a flat `dict[str, str]` for the
SQLite metadata layer, where they serve as a cache for "단순 사실" queries.
"""

from __future__ import annotations

from src.parse.xml_loader import LoadedReport


def extract_metadata(report: LoadedReport) -> dict[str, str]:
    """Return `ACODE → trimmed text` for every `<EXTRACTION>` tag.

    Searches main + all attachments. Empty ACODE values are dropped.
    First-seen value wins for duplicate ACODEs.
    """
    out: dict[str, str] = {}
    for root in (report.main, *report.attachments.values()):
        for el in root.iter("EXTRACTION"):
            acode = el.get("ACODE", "").strip()
            if not acode or acode in out:
                continue
            out[acode] = (el.text or "").strip()
    return out
