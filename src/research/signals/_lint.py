"""Future-prediction word linter for signal sentences.

The research-card pipeline's design decision (plan.md M3): all signal text
is **past-result interpretation only** — never future projection. This module
detects banned future-prediction language in sentences before they leave
deterministic nodes.

Banned categories:
  - 추정/전망: 예상, 전망, 가능성, 추정, 보일, 예측, 기대
  - 방향성: 강세, 약세, 상승세, 하락세, 강세 흐름, 약세 흐름, 흐름
  - 신호 표현: 신호, 시그널
  - 예측 동사: 보일 것, 될 것, 갈 것, 예상된다

Rule of thumb: if a sentence reads "X 가 ~할 것이다" / "X 신호" / "X 흐름" —
it's making a forward claim. We allow "~했음" / "~되었음" / "~체결됨" /
"~확대됨" — past-tense factual reporting.

Edge case: financial terminology like "상승률" (rate of increase) is
legitimate when describing a past return — we anchor on whole words to
avoid false positives. The patterns target the *forward-looking* nuance.
"""

from __future__ import annotations

import re

# Banned phrases. Order matters only for human readability — all are checked.
# Each entry is (regex_pattern, human_label).
_BANNED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"예상(?!치|가)"), "예상"),
    (re.compile(r"전망"), "전망"),
    (re.compile(r"가능성"), "가능성"),
    (re.compile(r"추정"), "추정"),
    (re.compile(r"예측"), "예측"),
    (re.compile(r"기대(?!감|치)"), "기대"),
    (re.compile(r"강세 흐름"), "강세 흐름"),
    (re.compile(r"약세 흐름"), "약세 흐름"),
    (re.compile(r"상승세"), "상승세"),
    (re.compile(r"하락세"), "하락세"),
    (re.compile(r"신호(?!등)"), "신호"),
    (re.compile(r"시그널"), "시그널"),
    (re.compile(r"보일 것"), "보일 것"),
    (re.compile(r"될 것"), "될 것"),
    (re.compile(r"갈 것"), "갈 것"),
    (re.compile(r"예상된다"), "예상된다"),
    (re.compile(r"전망된다"), "전망된다"),
]


def find_violations(sentence: str) -> list[str]:
    """Return the human labels of all banned phrases found in ``sentence``.

    Empty list ⇒ sentence is past-tense factual.
    """
    return [label for pattern, label in _BANNED if pattern.search(sentence)]


def assert_past_only(sentence: str) -> None:
    """Raise ``ValueError`` if ``sentence`` contains any banned phrase."""
    violations = find_violations(sentence)
    if violations:
        raise ValueError(
            f"future-prediction word(s) {violations} in sentence: {sentence!r}"
        )
