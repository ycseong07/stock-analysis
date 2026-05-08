"""Split a DART web-viewer HTML document into hierarchical sections.

DART's viewer pre-renders the body so section boundaries are CSS classes:

  - ``<p class='section-1'>``  → 부  (level 1, e.g. 'I. 회사의 개요')
  - ``<p class='section-2'>``  → 절  (level 2, e.g. '1. 회사의 연혁')
  - ``<p class='section-3'>``  → 항  (level 3, rare)

Layout-only paragraphs (``pgbrk``, ``cover-title``, ``img-caption``,
``table-group``) are skipped. All other ``<p>`` and ``<table>`` between
two headings belong to the latest section.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from lxml.html import HtmlElement

from src.parse.sections import Section

log = logging.getLogger(__name__)

_LEVEL_CLASSES: dict[str, int] = {"section-1": 1, "section-2": 2, "section-3": 3}
_IGNORED_P_CLASSES = {"pgbrk", "cover-title", "img-caption", "table-group"}


def _heading_level(el: HtmlElement) -> int:
    """Return level (1/2/3) if `el` is a `<p class='section-N'>` heading; else 0."""
    if el.tag != "p":
        return 0
    cls = el.get("class") or ""
    for marker, level in _LEVEL_CLASSES.items():
        if marker in cls:
            return level
    return 0


def _is_layout_p(el: HtmlElement) -> bool:
    """True if this `<p>` is a layout/page-break paragraph (no real content)."""
    if el.tag != "p":
        return False
    cls = el.get("class") or ""
    return cls in _IGNORED_P_CLASSES


def _body_children(root: HtmlElement) -> Iterator[HtmlElement]:
    """Yield body's direct element children (skip Comment / PI nodes)."""
    bodies = root.xpath("//body")
    if not bodies:
        return
    for el in bodies[0]:
        if isinstance(el.tag, str):
            yield el


def split_html_into_sections(root: HtmlElement) -> list[Section]:
    """Walk body's children in document order; emit one `Section` per heading.

    Content paragraphs and tables encountered between headings are appended
    to the latest section's `elements` list. Orphan content before the first
    heading is dropped.
    """
    sections: list[Section] = []
    current: Section | None = None

    for el in _body_children(root):
        level = _heading_level(el)
        if level > 0:
            title = (el.text_content() or "").strip()
            if title:
                current = Section(title=title, level=level)
                sections.append(current)
            continue
        if _is_layout_p(el):
            continue
        if current is None:
            continue
        if el.tag in ("p", "table"):
            current.elements.append(el)
    return sections
