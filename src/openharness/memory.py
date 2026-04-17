"""Persistent memory sources — the generalization of CLAUDE.md.

Claude Code injects the project's ``CLAUDE.md`` file into every prompt
and it survives all compactions. This module gives open-harness an
equivalent that is *domain-agnostic*: any object exposing
``load(state) -> str | None`` can be attached to ``LoopConfig`` and the
loop will render it into a stable ``MEMORY`` section at the top of
every prompt.

Two tiny convenience implementations are shipped:

* :class:`StaticMemorySource` — constant text (rubrics, SOPs, style
  notes).
* :class:`CallableMemorySource` — wraps a lambda; receives the current
  ``AgentState`` so memory can vary per turn (e.g. "case id = X,
  pinned entities = [...]").

For a filesystem-backed source (CLAUDE.md equivalent), callers can
compose ``CallableMemorySource(lambda _: Path("RUBRIC.md").read_text())``
— openharness core stays out of the filesystem.

Rendering is done by :func:`render_memory`, which returns a single
string joined by blank lines with empty/None returns skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


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


def render_memory(
    sources: list[PersistentMemorySource] | None,
    state: Any,
) -> str:
    """Join every source's ``load(state)`` with a blank line.

    Falsy outputs (``None`` / empty / whitespace-only) are silently
    skipped so adding an optional source never yields stray blank
    sections. Returns an empty string when there is nothing to render.
    """
    if not sources:
        return ""
    chunks: list[str] = []
    for src in sources:
        text = src.load(state)
        if text is None:
            continue
        s = str(text).strip()
        if s:
            chunks.append(s)
    return "\n\n".join(chunks)
