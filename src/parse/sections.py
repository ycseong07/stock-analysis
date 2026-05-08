"""Split a DART report into hierarchical sections via `<TITLE>` tags.

DART periodic reports follow the standard Korean disclosure form:

  level 1: I. 회사의 개요         (roman numeral)
  level 2: 1. 회사의 연혁         (arabic numeral)
  level 3: 가. ...                (한글 자모)
  level 4: (1) ...                (parenthesized number)
  level 5: ① ...                  (circled number)

`detect_level` infers the level from the title's prefix. `split_into_sections`
walks the document linearly, treating each `<TITLE>` as a section heading
and assigning the following `<P>` and `<TABLE>` elements (encountered in
document order, table internals not descended into) to that section.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from lxml import etree

_ROMAN = re.compile(r"^[IVXLCDM]+\.\s")
_ARABIC = re.compile(r"^\d+\.\s")
_HANGUL = re.compile(r"^[가-힣]\.\s")
_PAREN_NUM = re.compile(r"^\(\d+\)\s")
_CIRCLED = re.compile(r"^[①-⑳]\s")


@dataclass
class Section:
    """One section of a DART report: a heading plus its content elements."""

    title: str
    level: int
    elements: list[etree._Element] = field(default_factory=list)


def detect_level(title: str) -> int:
    """Return the heading level (1–5) implied by the title's prefix.

    Returns 0 when no known prefix matches.
    """
    t = title.strip()
    if _ROMAN.match(t):
        return 1
    if _ARABIC.match(t):
        return 2
    if _HANGUL.match(t):
        return 3
    if _PAREN_NUM.match(t):
        return 4
    if _CIRCLED.match(t):
        return 5
    return 0


def iter_content(root: etree._Element) -> Iterator[etree._Element]:
    """Yield `<TITLE>`, `<P>`, `<TABLE>` in document order.

    Treats `<TITLE>` and `<TABLE>` as leaves: their internals are not
    descended into. This avoids yielding `<P>` tags that live inside table
    cells (which would otherwise duplicate or misattribute table text).
    """
    stack: list[etree._Element] = [root]
    while stack:
        el = stack.pop()
        if el.tag in ("TITLE", "TABLE"):
            yield el
            continue
        if el.tag == "P":
            yield el
            continue
        for child in reversed(list(el)):
            stack.append(child)


def split_into_sections(root: etree._Element) -> list[Section]:
    """Walk `root` linearly; emit one `Section` per `<TITLE>`."""
    sections: list[Section] = []
    current: Section | None = None
    for el in iter_content(root):
        if el.tag == "TITLE":
            title = "".join(el.itertext()).strip()
            if not title:
                continue
            current = Section(title=title, level=detect_level(title))
            sections.append(current)
        elif current is not None:
            current.elements.append(el)
    return sections
