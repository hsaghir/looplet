"""web_search tool - dependency-free web search with pluggable backends.

Mirrors ``web_fetch``'s zero-dependency style: the default backend is
DuckDuckGo's no-key HTML endpoint, parsed with the standard library so the
cartridge keeps zero runtime dependencies. If one of the optional API keys
is present in the environment, that higher-quality backend is used instead
(checked in this order):

* ``BRAVE_API_KEY``  → Brave Search API
* ``TAVILY_API_KEY`` → Tavily Search API
* ``SEARXNG_URL``    → a self-hosted SearXNG instance (JSON API)

Returns ``{"query", "backend", "count", "results": [{"title", "url",
"snippet"}]}`` on success, or ``{"error": ..., "query": ...}`` on failure.
The tool only *finds* pages; use ``web_fetch`` to read one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from looplet.types import ToolContext

_UA = "looplet-coder/0.1 (+https://github.com)"
_TIMEOUT = 20


def _clean_ddg_url(href: str | None) -> str:
    """Resolve DuckDuckGo's ``/l/?uddg=`` redirect wrapper to the real URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return href


class _DDGParser(HTMLParser):
    """Extract result links and snippets from DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._mode: str | None = None  # "title" | "snippet"
        self._href: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr = dict(attrs)
        cls = attr.get("class") or ""
        if "result__a" in cls:
            self._mode = "title"
            self._href = attr.get("href")
            self._buf = []
        elif "result__snippet" in cls:
            self._mode = "snippet"
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._mode is None:
            return
        text = " ".join("".join(self._buf).split())
        if self._mode == "title":
            self.results.append({"title": text, "url": _clean_ddg_url(self._href), "snippet": ""})
        elif self._mode == "snippet" and self.results:
            self.results[-1]["snippet"] = text
        self._mode = None
        self._buf = []


def _get(url: str, *, headers: dict | None = None, data: bytes | None = None) -> str:
    request = urllib.request.Request(url, data=data, headers={"User-Agent": _UA, **(headers or {})})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    body = urllib.parse.urlencode({"q": query}).encode()
    html = _get("https://html.duckduckgo.com/html/", data=body)
    parser = _DDGParser()
    parser.feed(html)
    out = [r for r in parser.results if r["url"]]
    return out[:max_results]


def _search_brave(query: str, max_results: int, key: str) -> list[dict]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": max_results}
    )
    raw = _get(url, headers={"Accept": "application/json", "X-Subscription-Token": key})
    data = json.loads(raw)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in data.get("web", {}).get("results", [])[:max_results]
    ]


def _search_tavily(query: str, max_results: int, key: str) -> list[dict]:
    payload = json.dumps({"api_key": key, "query": query, "max_results": max_results}).encode()
    raw = _get(
        "https://api.tavily.com/search", headers={"Content-Type": "application/json"}, data=payload
    )
    data = json.loads(raw)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])[:max_results]
    ]


def _search_searxng(query: str, max_results: int, base: str) -> list[dict]:
    url = base.rstrip("/") + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    data = json.loads(_get(url))
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])[:max_results]
    ]


def execute(ctx: ToolContext | None, *, query: str, max_results: int = 5) -> dict:
    query = (query or "").strip()
    if not query:
        return {"error": "web_search requires a non-empty query", "query": query}
    n = max(1, min(int(max_results or 5), 20))

    if os.environ.get("BRAVE_API_KEY"):
        backend, run = "brave", lambda: _search_brave(query, n, os.environ["BRAVE_API_KEY"])
    elif os.environ.get("TAVILY_API_KEY"):
        backend, run = "tavily", lambda: _search_tavily(query, n, os.environ["TAVILY_API_KEY"])
    elif os.environ.get("SEARXNG_URL"):
        backend, run = "searxng", lambda: _search_searxng(query, n, os.environ["SEARXNG_URL"])
    else:
        backend, run = "duckduckgo", lambda: _search_duckduckgo(query, n)

    try:
        results = run()
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.reason}", "query": query, "backend": backend}
    except urllib.error.URLError as exc:
        return {"error": f"Search failed: {exc.reason}", "query": query, "backend": backend}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"error": f"Bad response from {backend}: {exc}", "query": query, "backend": backend}

    return {"query": query, "backend": backend, "count": len(results), "results": results}
