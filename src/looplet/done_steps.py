"""Helpers for inspecting ``done()`` steps recorded in ``state.steps``.

The composable loop records every ``done()`` invocation as a regular
``Step`` in ``state.steps`` — including attempts that were rejected by
a hook's ``check_done``.  Rejected attempts carry the marker
``tool_result.data = {"rejected": True, "reason": "<gate message>"}``
set by the loop engine (see ``loop.py`` where quality-gate rejections
are recorded).

These helpers let consumers walk the history without reimplementing
the marker convention:

- :func:`iter_done_steps` — yield every ``done()`` step in reverse
  chronological order.
- :func:`last_accepted_done` — return the most recent accepted
  ``done()`` step (payload lives in ``step.tool_result.data``).
- :func:`last_rejected_done` — return the most recent rejected
  ``done()`` step (intent lives in ``step.tool_call.args``).

Typical use — recovering the agent's intended verdict when the gate
rejected ``done()`` and budget then ran out::

    from looplet.done_steps import last_accepted_done, last_rejected_done

    accepted = last_accepted_done(state)
    if accepted is not None:
        payload = accepted.tool_result.data
        ...  # consume the verdict

    rejected = last_rejected_done(state)
    if rejected is not None:
        intent = rejected.tool_call.args  # what the agent wanted to say
        reason = rejected.tool_result.data.get("reason", "")
        ...  # recover with a confidence penalty, log reason, etc.

Pure state-scanning: no I/O, no LLM calls, no domain assumptions
about ``done()``'s arg shape.
"""

from __future__ import annotations

from typing import Iterator

from looplet.types import Step

__all__ = [
    "iter_done_steps",
    "last_accepted_done",
    "last_rejected_done",
    "is_rejected_done",
]


# The tool name the composable loop uses for the termination signal.
# Consumers that rename ``done`` can pass ``tool_name`` to the helpers.
_DEFAULT_DONE_TOOL = "done"


def is_rejected_done(step: Step, *, tool_name: str = _DEFAULT_DONE_TOOL) -> bool:
    """Return ``True`` if ``step`` is a ``done()`` rejected by a hook.

    Matches the convention set by the loop engine:
    ``step.tool_call.tool == tool_name`` and
    ``step.tool_result.data == {"rejected": True, ...}``.
    """
    if step.tool_call.tool != tool_name:
        return False
    data = step.tool_result.data
    return isinstance(data, dict) and bool(data.get("rejected"))


def iter_done_steps(
    state: object,
    *,
    tool_name: str = _DEFAULT_DONE_TOOL,
) -> Iterator[Step]:
    """Yield every ``done()`` step from ``state.steps`` in reverse order.

    Walking in reverse is the common case — consumers usually want the
    most recent attempt.  Any object with a ``steps`` attribute works;
    this matches the ``AgentState`` protocol without importing it.
    """
    steps = getattr(state, "steps", None) or []
    for step in reversed(steps):
        if step.tool_call.tool == tool_name:
            yield step


def last_accepted_done(
    state: object,
    *,
    tool_name: str = _DEFAULT_DONE_TOOL,
) -> Step | None:
    """Return the most recent accepted ``done()`` step, or ``None``.

    "Accepted" means the step is a ``done()`` call whose
    ``tool_result.data`` is not a rejection marker.  The returned
    step's ``tool_result.data`` is the payload the ``done`` tool
    produced (typically a dict of verdict / reasoning / etc. — shape
    is consumer-defined).
    """
    for step in iter_done_steps(state, tool_name=tool_name):
        if not is_rejected_done(step, tool_name=tool_name):
            return step
    return None


def last_rejected_done(
    state: object,
    *,
    tool_name: str = _DEFAULT_DONE_TOOL,
) -> Step | None:
    """Return the most recent rejected ``done()`` step, or ``None``.

    The agent's intended payload lives in ``step.tool_call.args``; the
    gate's rejection reason lives in
    ``step.tool_result.data["reason"]``.
    """
    for step in iter_done_steps(state, tool_name=tool_name):
        if is_rejected_done(step, tool_name=tool_name):
            return step
    return None
