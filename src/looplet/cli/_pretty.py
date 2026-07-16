"""Pretty trace printer for ``looplet new`` and ``looplet run-cartridge``.

A small, dependency-free renderer that turns a stream of looplet
``Step`` objects into a legible terminal trace.

## Why this exists

The default per-step printer (``  ✓ step 7: bash({"command": "..."})``)
is fine for CI logs but hard to follow when you're watching an agent
work in real time. This module renders the same stream as a
human-friendly trace: the LLM's reasoning is shown verbatim, tool
results are summarised in one line, errors and recoveries are
colorised, and a sticky footer tracks step count + elapsed time.

## Design rules

- **Append-only.** No cursor jumping, no flicker, no full-screen
  takeover. Every line printed stays on screen forever. Works in
  ``tee``, ``less +F``, ``asciinema``, plain dumb terminals, and CI
  log scrapers.
- **Stdlib only.** No ``rich``, no ``textual``. ANSI escape sequences
  + ``shutil.get_terminal_size`` are enough.
- **Falls back to plain text.** When stdout isn't a TTY (piped to a
  file, captured by CI), every escape collapses to nothing and the
  output is grep-friendly.
- **Bounded line lengths.** Long tool args / results are summarised,
  not truncated mid-string. Use the ``--trace`` directory if you need
  the full content.

The printer is a *consumer* of looplet ``Step`` objects, not a hook —
it lives in the CLI layer so the loop engine stays uncluttered.
"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Any

# ── ANSI ─────────────────────────────────────────────────────────────


def _supports_color() -> bool:
    """True when stdout is a TTY and the terminal isn't ``dumb``."""
    import os as _os

    return sys.stdout.isatty() and _os.environ.get("TERM", "") != "dumb"


_COLOR = _supports_color()


def _ansi(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def _bold(s: str) -> str:
    return _ansi("1", s)


def _dim(s: str) -> str:
    return _ansi("2", s)


def _italic(s: str) -> str:
    return _ansi("3", s)


def _red(s: str) -> str:
    return _ansi("31", s)


def _green(s: str) -> str:
    return _ansi("32", s)


def _yellow(s: str) -> str:
    return _ansi("33", s)


def _blue(s: str) -> str:
    return _ansi("34", s)


def _cyan(s: str) -> str:
    return _ansi("36", s)


def _grey(s: str) -> str:
    return _ansi("90", s)


# ── helpers ──────────────────────────────────────────────────────────


def _term_width(default: int = 100) -> int:
    """Return the current terminal width, capped at 120 for readability."""
    try:
        cols = shutil.get_terminal_size((default, 20)).columns
    except OSError:
        cols = default
    return min(max(cols, 60), 120)


def _summarize_value(value: Any, max_chars: int = 60) -> str:
    """Render ``value`` as a one-line summary suitable for inline display.

    Examples:
      ``"hello"``                   →  ``'"hello"'``
      ``[1, 2, 3]``                 →  ``[3 items]``
      ``{"a": 1, "b": 2}``          →  ``{2 keys}``
      ``"a very long string..."``   →  ``'"a very long string …"'``
      ``None``                      →  ``None``
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if len(value) <= max_chars:
            return repr(value)
        return repr(value[: max_chars - 1] + "…")
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    return f"<{type(value).__name__}>"


def _format_args(args: dict[str, Any], width: int) -> str:
    """Render ``tool_call.args`` as ``key=value, ...`` within ``width``.

    When the full rendering exceeds ``width``, fall back to showing
    the first arg with its value plus a ``(N args)`` count, instead
    of dropping all values. Keeps the output informative even when
    args are large.
    """
    if not args:
        return ""
    parts = [f"{k}={_summarize_value(v, max_chars=40)}" for k, v in args.items()]
    rendered = ", ".join(parts)
    if len(rendered) <= width:
        return rendered
    # Too wide — keep the first key=value, drop the rest, show count.
    first = parts[0]
    rest_count = len(args) - 1
    if rest_count <= 0:
        return _truncate(first, width)
    suffix = _dim(f"  +{rest_count} more")
    head_budget = width - _visible_len(suffix) - 1
    return _truncate(first, head_budget) + suffix


def _format_result(data: Any, error: str | None) -> str:
    """Render a one-line summary of a tool's return value."""
    if error:
        return _red(f"error: {error[:100]}")
    if data is None:
        return _grey("→ no data")
    if isinstance(data, dict):
        # Surface the most informative keys.
        if "error" in data:
            return _red(f"→ error: {str(data['error'])[:80]}")
        # Common helpful keys to show first.
        for k in ("summary", "result", "count", "n_tools", "tools", "valid"):
            if k in data:
                return _grey(f"→ {k}={_summarize_value(data[k], max_chars=60)}")
        # Fallback — just show the shape.
        return _grey(
            f"→ {{{len(data)} keys: {', '.join(list(data)[:3])}{'…' if len(data) > 3 else ''}}}"
        )
    return _grey(f"→ {_summarize_value(data, max_chars=80)}")


# ── the printer ──────────────────────────────────────────────────────


class PrettyPrinter:
    """Render a live trace of looplet ``Step`` objects.

    Intended use::

        printer = PrettyPrinter(title="gh_triager", subtitle="step")
        for step in composable_loop(...):
            printer.step(step)
        printer.finish(summary="...")
    """

    def __init__(
        self,
        title: str,
        *,
        subtitle: str = "",
        max_steps: int | None = None,
        show_reasoning: bool = True,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.max_steps = max_steps
        self.show_reasoning = show_reasoning
        self._t0 = time.time()
        self._n_steps = 0
        self._n_denies = 0
        self._n_errors = 0
        self._width = _term_width()
        self._header_printed = False

    # ---- header / footer -------------------------------------------------

    def header(self, lines: list[str]) -> None:
        """Print the leading banner: title + a few key=value lines."""
        # Width budget: account for the leading "╭─ " (3 chars) and " " + a
        # trailing "─" so the rule extends to the right margin.
        title_text = f" {self.title} "
        prefix = "╭─"
        # Fill remaining cols with ─. Visible width of the line:
        # len(prefix) + len(title_text) + (fill).
        fill_top = max(self._width - len(prefix) - len(title_text), 2)
        print(_cyan(prefix) + _bold(title_text) + _cyan("─" * fill_top))
        for line in lines:
            print(_cyan("│ ") + line)
        print(_cyan("╰" + "─" * (self._width - 1)))
        print()
        self._header_printed = True

    def finish(
        self,
        *,
        summary: str | None = None,
        success: bool = True,
        extra: list[str] | None = None,
    ) -> None:
        """Print the trailing summary box."""
        elapsed = time.time() - self._t0
        glyph = _green("✓") if success else _red("✗")
        stats = f"in {elapsed:.1f}s — {self._n_steps} steps, {self._n_denies} denies, {self._n_errors} errors"
        print()
        print(f"{glyph} {_bold('done')} {_dim(stats)}")
        if summary:
            print(f"  {_italic('agent says:')} {summary}")
        if extra:
            for line in extra:
                print(f"  {line}")

    # ---- per-step --------------------------------------------------------

    def step(self, step: Any) -> None:
        """Render one step from ``composable_loop``."""
        self._n_steps += 1
        tc = getattr(step, "tool_call", None)
        tr = getattr(step, "tool_result", None)
        if tc is None:
            # Recovery / no-op step — surface it briefly.
            print(_dim(f"  · step {self._n_steps:>2}  (no tool call)"))
            return

        # Detect error / deny.
        err_msg = None
        if tr is not None:
            if getattr(tr, "error", None):
                err_msg = tr.error
            elif isinstance(getattr(tr, "data", None), dict) and "error" in tr.data:
                err_msg = str(tr.data["error"])
        is_done = tc.tool == "done"
        if err_msg:
            self._n_errors += 1
            tag = _red("✗")
        elif is_done:
            tag = _cyan("◆")
        else:
            tag = _green("✓")

        # Header line: "  ✓ step  7  bash(...)               0.34s"
        args_str = _format_args(getattr(tc, "args", {}) or {}, self._width - 32)
        duration_ms = getattr(tr, "duration_ms", None) if tr is not None else None
        right = ""
        if duration_ms is not None:
            secs = duration_ms / 1000.0
            right = _dim(f"{secs:>5.2f}s")
        # Line 1: header
        head = f"  {tag} {_bold(f'step {self._n_steps:>2}')}  {_blue(tc.tool)}({args_str})"
        # Pad and append duration on the right if it fits.
        if right:
            visible_len = _visible_len(head)
            pad = max(self._width - visible_len - len(_strip_ansi(right)), 1)
            print(head + (" " * pad) + right)
        else:
            print(head)

        # Line 2: LLM reasoning, if present and asked-for.
        reasoning = getattr(tc, "reasoning", "") or ""
        if self.show_reasoning and reasoning:
            first = reasoning.strip().splitlines()[0].strip()
            if first:
                print(f"     {_dim('why')} {_italic(_grey(_truncate(first, self._width - 9)))}")

        # Line 3: tool result summary.
        if tr is not None:
            print(f"     {_format_result(getattr(tr, 'data', None), err_msg)}")

        # Track denies (any non-done error).
        if err_msg and not is_done:
            self._n_denies += 1

    # ---- progress helpers used by callers --------------------------------

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_denies(self) -> int:
        return self._n_denies


# ── small string utilities ───────────────────────────────────────────


def _truncate(s: str, max_chars: int) -> str:
    """Truncate ``s`` with an ellipsis when it exceeds ``max_chars``."""
    if max_chars <= 1 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _strip_ansi(s: str) -> str:
    """Strip ANSI escape sequences for length calculations."""
    import re

    return re.sub(r"\033\[[0-9;]*m", "", s)


def _visible_len(s: str) -> int:
    """Length of ``s`` ignoring ANSI escape sequences."""
    return len(_strip_ansi(s))


__all__ = ["PrettyPrinter"]
