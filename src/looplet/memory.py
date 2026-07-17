"""Persistent memory sources for agent loops.

Many agent frameworks inject a project-level memory file into every
prompt, surviving all compactions. This module gives looplet an
equivalent that is *domain-agnostic*: any object exposing
``load(state) -> str | None`` can be attached to ``LoopConfig`` and the
loop will render it into a stable ``MEMORY`` section at the top of
every prompt.

Three tiny convenience implementations are shipped:

* :class:`StaticMemorySource` - constant text (rubrics, SOPs, style
  notes).
* :class:`CallableMemorySource` - wraps a lambda; receives the current
  ``AgentState`` so memory can vary per turn (e.g. "case id = X,
  pinned entities = [...]").
* :class:`AgentsMdMemorySource` - walks parent directories from a
  starting path collecting ``AGENTS.md`` / ``CLAUDE.md`` files (the
  convention shared by Claude Code, Pi, and Cursor). Read on
  construction; pass to ``LoopConfig.memory_sources``.

For other filesystem-backed sources, callers can compose
``CallableMemorySource(lambda _: Path("RUBRIC.md").read_text())`` -
looplet core stays out of the filesystem otherwise.

Rendering is done by :func:`render_memory`, which returns a single
string joined by blank lines with empty/None returns skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "PersistentMemorySource",
    "StaticMemorySource",
    "CallableMemorySource",
    "AgentsMdMemorySource",
    "render_memory",
]


@runtime_checkable
class PersistentMemorySource(Protocol):
    """Any object with a ``load(state) -> str | None`` method.

    ``state`` is the loop's ``AgentState`` at the moment of rendering.
    Implementations may ignore it (for static memory) or read it (for
    dynamic memory such as current case metadata).
    """

    def load(self, state: Any) -> str | None: ...


@dataclass(frozen=True)
class StaticMemorySource:
    """Constant text returned on every turn.

    Useful for rubrics, style guides, mandatory instructions, etc.
    """

    text: str

    def load(self, state: Any) -> str:  # noqa: ARG002 - state unused
        return self.text


@dataclass(frozen=True)
class CallableMemorySource:
    """Wraps a ``Callable[[state], str | None]`` as a memory source.

    The callable is invoked on every turn; the current ``state`` is
    passed so the memory can vary (e.g. include pinned entity ids).
    """

    fn: Callable[[Any], str | None]

    def load(self, state: Any) -> str | None:
        return self.fn(state)


# Default filenames in walk order - first match per directory wins.
_AGENTS_MD_FILENAMES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")


@dataclass
class AgentsMdMemorySource:
    """Walk parent directories from ``start`` collecting agent context files.

    The convention (shared by Claude Code, Pi, Cursor, and others) is
    that a repository may carry an ``AGENTS.md`` (or ``CLAUDE.md``)
    file holding project-level instructions for an LLM coding agent.
    A user may also place one in ``$HOME`` or any parent directory.

    On construction this class walks from ``start`` (default: cwd) up
    to ``stop`` (default: filesystem root) and concatenates the first
    matching file found in each directory, *outermost first*. The
    resulting text is rendered into the loop's ``MEMORY`` section on
    every turn.

    Args:
        start: Directory to start the upward walk from. Defaults to
            ``Path.cwd()`` evaluated at construction time.
        stop: Directory to stop at (inclusive). Defaults to the
            filesystem root.
        filenames: Filenames to look for in each directory. The first
            existing file per directory wins. Defaults to
            ``("AGENTS.md", "CLAUDE.md")``.
        include_home: If True, also include ``~/AGENTS.md`` /
            ``~/CLAUDE.md`` even when ``$HOME`` is not on the walk
            path. Defaults to False to keep the default behaviour
            local-repo-only.
        max_chars: Soft cap on total characters loaded; entries are
            truncated with a ``[…truncated…]`` marker once exceeded.
            Defaults to 16_000 (~4k tokens at 4 chars/token).

    Notes:
        File contents are read **once at construction**. Callers who
        want hot-reload should reconstruct the source between runs.
        This matches the lifetime of an agent run and keeps the
        ``load(state)`` call cheap.
    """

    start: Path = field(default_factory=Path.cwd)
    stop: Path | None = None
    filenames: tuple[str, ...] = _AGENTS_MD_FILENAMES
    include_home: bool = False
    max_chars: int = 16_000
    _rendered: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self._rendered = self._collect()

    def _collect(self) -> str:
        seen: set[Path] = set()
        chunks: list[str] = []
        budget = self.max_chars

        # Walk root → start so outer (broader) context comes first.
        start = self.start.resolve()
        stop = (self.stop or Path(start.anchor)).resolve()
        chain: list[Path] = []
        cur = start
        while True:
            chain.append(cur)
            if cur == stop or cur.parent == cur:
                break
            cur = cur.parent
        chain.reverse()

        if self.include_home:
            home = Path.home().resolve()
            if home not in chain:
                chain.insert(0, home)

        for d in chain:
            for name in self.filenames:
                f = d / name
                if not f.is_file() or f in seen:
                    continue
                seen.add(f)
                try:
                    text = f.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    continue
                if not text:
                    continue
                header = f"# {f}"
                body = text
                if budget <= 0:
                    return "\n\n".join(chunks)
                if len(body) > budget:
                    body = body[: max(0, budget - 32)] + "\n[…truncated…]"
                chunks.append(f"{header}\n\n{body}")
                budget -= len(body) + len(header) + 2
                break  # one file per directory
        return "\n\n".join(chunks)

    def load(self, state: Any) -> str | None:  # noqa: ARG002 - state unused
        return self._rendered or None


def render_memory(
    sources: list[PersistentMemorySource] | None,
    state: Any,
) -> str:
    """Join every source's ``load(state)`` with a blank line.

    Falsy outputs (``None`` / empty / whitespace-only) are silently
    skipped so adding an optional source never yields stray blank
    sections. Returns an empty string when there is nothing to render.

    Sources are isolated: if one ``load`` raises, the exception is
    logged and the source is skipped, mirroring hook isolation in
    the loop. A single buggy memory source must not bring the loop
    down.
    """
    if not sources:
        return ""
    chunks: list[str] = []
    for src in sources:
        try:
            text = src.load(state)
        except Exception:  # noqa: BLE001
            logger.exception(
                "memory source %s.load() raised; skipping for this turn",
                type(src).__name__,
            )
            continue
        if text is None:
            continue
        s = str(text).strip()
        if s:
            chunks.append(s)
    return "\n\n".join(chunks)
