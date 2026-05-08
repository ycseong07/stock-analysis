"""Fetch DART filings as rendered HTML from the dart.fss.or.kr web viewer.

The OPEN DART `document.xml` zip carries a non-standard XML body whose tags
(`<TU>`, `<TABLE-GROUP>`, `<PGBRK>`, etc.) lack official rendering rules.
The same filing fetched from `dart.fss.or.kr` (the public viewer) comes back
as well-formed HTML 4.01 with `section-1`/`section-2` CSS classes — the
parsing/rendering already done by DART itself.

Per-report flow:
  1. GET /dsaf001/main.do?rcpNo=...  → parse JS `treeData` (TOC) variables.
  2. Filter top-level nodes (var name == 'node1') = 부 (level 1 sections).
     Each top-level node's viewer body contains all of its child 절s.
  3. For each top-level node, GET /report/viewer.do with
     (rcpNo, dcmNo, eleId, offset, length).
  4. Strip per-section `<html>/<body>` wrappers and concatenate into a single
     HTML document that the caller can persist to GCS.

Auth: none (public web viewer). User-Agent + Referer required.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from types import TracebackType

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

DART_HOST = "https://dart.fss.or.kr"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": DART_HOST,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8",
}
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_SLEEP_S = 2.0  # polite throttle between calls

# Matches a JS block of the form `var nodeN = {}; nodeN['k'] = "v"; ...`.
_NODE_BLOCK = re.compile(
    r"var\s+(node\d+)\s*=\s*\{\};((?:\s*\1\[\'[A-Za-z_][\w]*\'\]\s*=\s*\"[^\"]*\";)+)"
)
_PROP = re.compile(r"\['([A-Za-z_][\w]*)'\]\s*=\s*\"([^\"]*)\"")


@dataclass(frozen=True)
class PageNode:
    """One node in the `treeData` TOC of a DART filing's main.do."""

    var_name: str  # 'node1' | 'node2' | ... — depth indicator
    text: str
    rcp_no: str
    dcm_no: str
    ele_id: str
    offset: str
    length: str


def parse_tree_data(main_html: str) -> list[PageNode]:
    """Extract `PageNode`s from a main.do response, in document order."""
    nodes: list[PageNode] = []
    for var_name, body in _NODE_BLOCK.findall(main_html):
        props = dict(_PROP.findall(body))
        required = ("text", "rcpNo", "dcmNo", "eleId", "offset", "length")
        if not all(k in props for k in required):
            continue
        nodes.append(
            PageNode(
                var_name=var_name,
                text=props["text"],
                rcp_no=props["rcpNo"],
                dcm_no=props["dcmNo"],
                ele_id=props["eleId"],
                offset=props["offset"],
                length=props["length"],
            )
        )
    return nodes


def top_level_nodes(nodes: list[PageNode]) -> list[PageNode]:
    """Filter to level-1 (부) nodes; their viewer body contains all 절 children."""
    return [n for n in nodes if n.var_name == "node1"]


def _extract_body(html: str) -> str:
    """Return the inner HTML of `<body>...</body>` (or the whole input if missing)."""
    m = re.search(r"<body[^>]*>(.*?)</body>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else html


def _wrap_full_html(rcept_no: str, section_bodies: list[str]) -> str:
    """Wrap concatenated section bodies into a single browseable HTML document."""
    inner = "\n\n".join(section_bodies)
    return (
        '<!DOCTYPE html>\n<html lang="ko"><head>'
        '<meta charset="utf-8">'
        f"<title>DART {rcept_no}</title>"
        '<link rel="stylesheet" type="text/css" '
        'href="https://dart.fss.or.kr/css/report_xml.css">'
        "<style>body{font-family:-apple-system,Segoe UI,sans-serif;"
        "max-width:1100px;margin:1.5em auto;padding:0 1em;line-height:1.5;}"
        "table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:4px 8px;}"
        "</style>"
        "</head><body>\n"
        f"{inner}\n"
        "</body></html>\n"
    )


class DartViewer:
    """Synchronous client for the dart.fss.or.kr web viewer.

    Usage:
        with DartViewer() as v:
            html = v.fetch_report("20240312000736")
    """

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        sleep_between_s: float = _DEFAULT_SLEEP_S,
    ):
        # DART closes long-lived connections aggressively; disable keep-alive.
        self._client = httpx.Client(
            timeout=timeout,
            headers=_HEADERS,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._sleep = sleep_between_s
        self._last_call_t: float | None = None

    def _throttle(self) -> None:
        if self._sleep <= 0 or self._last_call_t is None:
            return
        wait = self._sleep - (time.perf_counter() - self._last_call_t)
        if wait > 0:
            time.sleep(wait)

    def __enter__(self) -> DartViewer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=4, max=120),
        reraise=True,
    )
    def fetch_main(self, rcept_no: str) -> str:
        self._throttle()
        try:
            t0 = time.perf_counter()
            resp = self._client.get(f"{DART_HOST}/dsaf001/main.do", params={"rcpNo": rcept_no})
            resp.raise_for_status()
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "dart_viewer_main",
                extra={
                    "rcept_no": rcept_no,
                    "bytes": len(resp.text),
                    "latency_ms": latency_ms,
                },
            )
            return resp.text
        finally:
            self._last_call_t = time.perf_counter()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=4, max=120),
        reraise=True,
    )
    def fetch_section(self, node: PageNode) -> str:
        self._throttle()
        try:
            t0 = time.perf_counter()
            resp = self._client.get(
                f"{DART_HOST}/report/viewer.do",
                params={
                    "rcpNo": node.rcp_no,
                    "dcmNo": node.dcm_no,
                    "eleId": node.ele_id,
                    "offset": node.offset,
                    "length": node.length,
                    "dtd": "dart3.xsd",
                },
            )
            resp.raise_for_status()
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "dart_viewer_section",
                extra={
                    "rcept_no": node.rcp_no,
                    "ele_id": node.ele_id,
                    "bytes": len(resp.text),
                    "latency_ms": latency_ms,
                },
            )
            return resp.text
        finally:
            self._last_call_t = time.perf_counter()

    def fetch_report(self, rcept_no: str) -> str:
        """Return one HTML document concatenating every top-level section."""
        main_html = self.fetch_main(rcept_no)
        nodes = top_level_nodes(parse_tree_data(main_html))
        if not nodes:
            raise ValueError(f"No top-level nodes in main.do for {rcept_no}")

        bodies: list[str] = []
        for n in nodes:
            section_html = self.fetch_section(n)
            marker = f"<!-- ele_id={n.ele_id} | {n.text} -->"
            bodies.append(f"{marker}\n{_extract_body(section_html)}")
        return _wrap_full_html(rcept_no, bodies)
