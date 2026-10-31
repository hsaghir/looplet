"""Helpers for inspecting ``done()`` steps recorded in ``state.steps``.

The composable loop records every ``done()`` invocation as a regular
``Step`` in ``state.steps`` - including attempts rejected by a gate
or a tool error. The loop records acceptance in
``step.metadata["terminal_accepted"]``. Business output fields do not
control acceptance. Older steps without this metadata use the legacy
``tool_result.data["rejected"]`` marker.

These helpers let consumers walk the history without reimplementing
the acceptance convention:

- :func:`iter_done_steps` - yield every ``done()`` step in reverse
  chronological order.
- :func:`last_accepted_done` - return the most recent accepted
  ``done()`` step (payload lives in ``step.tool_result.data``).
- :func:`last_rejected_done` - return the most recent rejected
  ``done()`` step (intent lives in ``step.tool_call.args``).

Typical use - recovering the agent's intended output when the gate
rejected ``done()`` and budget then ran out::

    from looplet.done_steps import last_accepted_done, last_rejected_done

    accepted = last_accepted_done(state)
    if accepted is not None:
        payload = accepted.tool_result.data
        ...  # consume the final output

    rejected = last_rejected_done(state)
    if rejected is not None:
        intent = rejected.tool_call.args  # what the agent wanted to say
        reason = rejected.tool_result.data.get("reason", "")
        ...  # recover with a confidence penalty, log reason, etc.

Pure state-scanning: no I/O, no LLM calls, no domain assumptions
about ``done()``'s arg shape.
"""

from __future__ import annotations

from typing import Any, Iterator

from looplet.types import Step

__all__ = [
    "iter_done_steps",
    "last_accepted_done",
    "last_rejected_done",
    "is_rejected_done",
    "done_output",
]


# The tool name the composable loop uses for the termination signal.
# Consumers that rename ``done`` can pass ``tool_name`` to the helpers.
_DEFAULT_DONE_TOOL = "done"


def is_rejected_done(step: Step, *, tool_name: str = _DEFAULT_DONE_TOOL) -> bool:
    """Return ``True`` if ``step`` is a rejected terminal attempt.

    Explicit ``terminal_accepted`` step metadata takes precedence.
    Steps without it retain the legacy ``data["rejected"]`` convention.
    """
    if step.tool_call.tool != tool_name:
        return False
    accepted = step.metadata.get("terminal_accepted")
    if isinstance(accepted, bool):
        return not accepted
    data = step.tool_result.data
    return isinstance(data, dict) and bool(data.get("rejected"))


def iter_done_steps(
    state: object,
    *,
    tool_name: str = _DEFAULT_DONE_TOOL,
) -> Iterator[Step]:
    """Yield every ``done()`` step from ``state.steps`` in reverse order.

    Walking in reverse is the common case - consumers usually want the
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

    Acceptance follows :func:`is_rejected_done`. The returned
    step's ``tool_result.data`` is the payload the ``done`` tool
    produced (typically a dict of summary / reasoning / etc. - shape
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


def done_output(
    state: object,
    *,
    tool_name: str = _DEFAULT_DONE_TOOL,
) -> Any | None:
    """Return the structured output of the accepted ``done()`` sentinel.

    This is the payload an out-of-process consumer (e.g. the RPC ``done``
    event) reports as the run's final ``output``: the
    :attr:`~looplet.types.ToolResult.data` of the most recent *accepted*
    ``done()`` step.  Returns ``None`` when the loop stopped without an
    accepted ``done()`` (budget/stagnation/cancel/error, or a ``done()``
    that a quality gate rejected).

    Thin wrapper over :func:`last_accepted_done` so callers don't have to
    reach through ``step.tool_result.data`` themselves.
    """
    step = last_accepted_done(state, tool_name=tool_name)
    if step is None:
        return None
    return step.tool_result.data
