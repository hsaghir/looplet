"""Generic stagnation detection hook.

Agents often loop on the same tool-call pattern or stop making
progress — re-issuing the same search, circling the same entity,
re-asking the same question — without actually closing on the task.
Every consumer writes a variant of "if nothing new for N steps, nudge
the LLM."  :class:`StagnationHook` packages the pattern with no
domain assumptions.

The hook is parameterized by two caller-provided callables:

- ``fingerprint(state, tool_call, tool_result) -> Hashable`` returns
  a value describing "what the agent accomplished this step."  Two
  consecutive steps with equal fingerprints are considered stagnant.
  Return ``None`` to mark the step as "definitely made progress"
  (the stagnation counter resets).

- ``progress(state) -> int | None`` (optional) returns a monotonic
  counter of real progress (findings, artifacts, distinct entities,
  lines of code written — anything).  When the counter increases
  between steps, stagnation resets even if the fingerprint matched.

When stagnation exceeds ``threshold`` consecutive steps, the hook
emits ``nudge`` (a caller-supplied string) via its ``post_dispatch``
return value, which looplet injects into the next prompt's briefing
section.  It then resets its counter so the nudge is not repeated
every step.

Typical use::

    from looplet.stagnation import StagnationHook, tool_call_fingerprint

    hook = StagnationHook(
        fingerprint=tool_call_fingerprint,
        threshold=3,
        nudge=(
            "[stagnation] You have repeated the same tool call 3 steps "
            "in a row without new results. Try a different angle."
        ),
    )

The module has zero third-party dependencies.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Protocol

__all__ = [
    "StagnationHook",
    "tool_call_fingerprint",
    "result_size_fingerprint",
]


class _StateLike(Protocol):
    pass


def tool_call_fingerprint(
    state: Any,
    tool_call: Any,
    tool_result: Any,
) -> Hashable:
    """Default fingerprint: the ``(tool, sorted_args)`` tuple.

    Matches when the agent issues the same call with the same
    arguments twice in a row.  Ignores ``tool_result`` so the
    fingerprint is well-defined before dispatch.
    """
    args = getattr(tool_call, "args", None) or {}
    try:
        frozen = tuple(sorted(args.items()))
    except TypeError:
        # Args contain unhashable values; fall back to repr.
        frozen = repr(args)
    return (getattr(tool_call, "tool", ""), frozen)


def result_size_fingerprint(
    state: Any,
    tool_call: Any,
    tool_result: Any,
) -> Hashable:
    """Fingerprint that also considers the shape of the result.

    Includes a coarse content signature alongside the tool+args so
    that calls returning different-sized results are not treated as
    identical.  Useful when a tool returns ``{"hits": []}``
    repeatedly with different args — the args vary but nothing is
    being learned.
    """
    base = tool_call_fingerprint(state, tool_call, tool_result)
    data = getattr(tool_result, "data", None)
    if data is None:
        sig: Hashable = ("none",)
    elif isinstance(data, dict):
        sized = 0
        for v in data.values():
            try:
                sized += len(v)
            except TypeError:
                sized += 1 if v else 0
        sig = ("dict", tuple(sorted(data.keys())), sized)
    elif isinstance(data, (list, tuple, set, str, bytes)):
        sig = (type(data).__name__, len(data))
    else:
        sig = (type(data).__name__, bool(data))
    return (base, sig)


class StagnationHook:
    """LoopHook that nudges the agent when progress stalls.

    Args:
        fingerprint: Callable returning a hashable describing the
            current step's work.  Consecutive equal fingerprints are
            counted as stagnant.  Return ``None`` to reset the
            counter explicitly.
        threshold: Number of consecutive stagnant steps that trigger
            the nudge.  Must be ``>= 2``.
        nudge: Text injected into the next prompt's briefing when
            the threshold is hit.  Can also be a callable
            ``(state, stagnant_steps) -> str`` for dynamic messages.
        progress: Optional callable ``(state) -> int`` returning a
            monotonic progress counter.  Resets the stagnation
            counter whenever the returned value increases.
        reset_after_nudge: If ``True`` (default), counter resets
            after firing so the nudge is not repeated every step.
            If ``False``, the nudge fires on every subsequent step
            while still stagnant.
        ignore_tools: Tool names that never count toward stagnation
            (e.g. ``{"done", "note"}``).  The step is skipped — the
            prior fingerprint is preserved.
    """

    def __init__(
        self,
        *,
        fingerprint: Callable[[Any, Any, Any], Hashable | None] = tool_call_fingerprint,
        threshold: int = 3,
        nudge: str | Callable[[Any, int], str] = (
            "[stagnation detected] You have repeated the same action "
            "without new progress. Try a different approach."
        ),
        progress: Callable[[Any], int] | None = None,
        reset_after_nudge: bool = True,
        ignore_tools: set[str] | None = None,
    ) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self._fingerprint_fn = fingerprint
        self._threshold = threshold
        self._nudge = nudge
        self._progress_fn = progress
        self._reset_after_nudge = reset_after_nudge
        self._ignore_tools = set(ignore_tools or ())

        # Internal state, per-hook-instance (not shared across loops).
        self._last_fingerprint: Hashable | None = None
        self._streak: int = 0
        self._last_progress: int | None = None

    # ── public introspection ──────────────────────────────────

    @property
    def stagnant_steps(self) -> int:
        """Current consecutive-stagnation count."""
        return self._streak

    def reset(self) -> None:
        """Clear internal state — useful between runs."""
        self._last_fingerprint = None
        self._streak = 0
        self._last_progress = None

    def to_config(self) -> dict[str, Any]:
        """Round-trip kwargs for ``preset_to_workspace``.

        Only round-trips the JSON-able fields (``threshold``, string
        ``nudge``, ``reset_after_nudge``, ``ignore_tools``). Callable
        ``fingerprint`` / ``progress`` / callable ``nudge`` are not
        portable across processes; reload uses the default fingerprint
        and a recovered string nudge when the original was a callable.
        """
        return {
            "threshold": self._threshold,
            "nudge": self._nudge
            if isinstance(self._nudge, str)
            else (
                "[stagnation detected] You have repeated the same action "
                "without new progress. Try a different approach."
            ),
            "reset_after_nudge": self._reset_after_nudge,
            "ignore_tools": sorted(self._ignore_tools),
        }

    # ── LoopHook slots ────────────────────────────────────────

    def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        tool_result: Any,
        step_num: int,
    ) -> str | None:
        tool_name = getattr(tool_call, "tool", "")
        if tool_name in self._ignore_tools:
            return None

        # Progress counter short-circuits the fingerprint check.
        if self._progress_fn is not None:
            try:
                current = int(self._progress_fn(state))
            except Exception:  # noqa: BLE001 — defensive: bad counter shouldn't break loop
                current = self._last_progress if self._last_progress is not None else 0
            if self._last_progress is not None and current > self._last_progress:
                self._streak = 0
                self._last_progress = current
                self._last_fingerprint = None
                return None
            self._last_progress = current

        fp = self._fingerprint_fn(state, tool_call, tool_result)
        if fp is None:
            # Caller declared explicit progress — reset.
            self._streak = 0
            self._last_fingerprint = None
            return None

        if self._last_fingerprint == fp:
            self._streak += 1
        else:
            self._streak = 1
            self._last_fingerprint = fp

        if self._streak >= self._threshold:
            if callable(self._nudge):
                msg = self._nudge(state, self._streak)
            else:
                msg = self._nudge
            if self._reset_after_nudge:
                self._streak = 0
                self._last_fingerprint = None
            return msg
        return None
