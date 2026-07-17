"""web_fetch tool - dependency-free HTTP(S) fetch with HTML text extraction."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

from looplet.types import ToolContext


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = (self.title + " " + text).strip()
        self.parts.append(text)


def _html_to_text(raw: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(raw)
    text = " ".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return parser.title, text.strip()


def execute(ctx: ToolContext | None, *, url: str, prompt: str = "", max_chars: int = 12000) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return {"error": "web_fetch only allows http and https URLs", "url": url}
    limit = max(1000, int(max_chars or 12000))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "looplet-coder/0.1 (+https://github.com)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get("content-type", "")
            body = response.read(limit * 4 + 1)
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.reason}", "url": url, "status": exc.code}
    except urllib.error.URLError as exc:
        return {"error": f"Fetch failed: {exc.reason}", "url": url}

    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.I)
    if match:
        encoding = match.group(1).strip()
    raw = body.decode(encoding, errors="replace")
    if "html" in content_type.lower() or "<html" in raw[:500].lower():
        title, text = _html_to_text(raw)
    else:
        title, text = "", raw.strip()
    truncated = len(text) > limit
    text = text[:limit]
    result: dict = {
        "url": url,
        "status": status,
        "content_type": content_type,
        "title": title,
        "text": text,
        "chars": len(text),
        "truncated": truncated,
    }
    if prompt.strip():
        if ctx is None or ctx.llm is None:
            result["prompt_note"] = "No ctx.llm was supplied, so prompt was not run."
        else:
            answer = ctx.llm.generate(
                "Answer the prompt using only the fetched content.\n\n"
                f"URL: {url}\n\nPROMPT:\n{prompt.strip()}\n\nCONTENT:\n{text}",
                max_tokens=1000,
                temperature=0.0,
            )
            result["answer"] = answer
    return result
