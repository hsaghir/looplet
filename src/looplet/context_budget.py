"""Central, env-overridable tunables for context-budget management.

Looplet's context pipeline has three nested budgets:

1. **Per-tool-result cap** - shared model projection bounds the complete
    serialized envelope, without mutating raw evidence. Source previews
    respect :data:`TOOL_RESULT_MAX_CHARS` and row limits unless overridden
    by explicit per-tool render policy. Persist-and-preview applies above
   :data:`TOOL_RESULT_PERSIST_THRESHOLD_CHARS` and a persist directory
   is configured.

2. **Per-context-window aggregate cap** - at LLM-prompt assembly time,
   ``state.context_summary()`` shows the last
    :data:`CONTEXT_WINDOW_STEPS` steps, using
    :data:`CONTEXT_INLINE_PER_STEP_CHARS` as the default envelope budget.
    Text history and native result batches are bounded by
    :data:`CONTEXT_WINDOW_TOTAL_CHARS`. Derived views arrive intact or
    are explicitly refused, never clipped past a hidden paging cursor.

3. **Whole-conversation compact** - when the loop's reactive-compact
   layer detects context pressure (handled by
   :class:`looplet.compact.CompactService` and the ``ThresholdCompactHook``).
   Tunables live in those modules; the only knob here is
   :data:`COMPACT_TRIGGER_FRACTION` (used by ThresholdCompactHook).

## Environment variable overrides

Every threshold can be overridden at process start via env vars with
the ``LOOPLET_`` prefix (e.g. ``LOOPLET_TOOL_RESULT_MAX_CHARS=12000``).
This lets ops teams tune budgets per-deployment without code changes.
The mapping is identity: ``TOOL_RESULT_MAX_CHARS`` →
``LOOPLET_TOOL_RESULT_MAX_CHARS``.

Modules that consume these constants should import them lazily
(``from looplet.context_budget import TOOL_RESULT_MAX_CHARS``) so a
test that monkeypatches the module sees the new value.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    """Read ``LOOPLET_<name>`` from the environment as an int.

    Returns ``default`` when the var is unset, empty, or unparseable -
    we never crash a process over a malformed budget knob.
    """
    raw = os.environ.get(f"LOOPLET_{name}", "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read ``LOOPLET_<name>`` from the environment as a float.

    Returns ``default`` when the var is unset, empty, or unparseable.
    """
    raw = os.environ.get(f"LOOPLET_{name}", "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ── Layer 1 - per-tool-result cap ─────────────────────────────────


TOOL_RESULT_MAX_CHARS: int = _env_int("TOOL_RESULT_MAX_CHARS", 6000)
"""Default source-result character cap, applied to the serialized model envelope.

The shared projector also respects the context inline limit. Explicit tool
render policy can override this default. Complete evidence is not truncated.
"""


TOOL_RESULT_MAX_ROWS: int = _env_int("TOOL_RESULT_MAX_ROWS", 50)
"""Max rows in a list-typed tool result before list truncation kicks in.

Applied when ``data`` is a ``list``. Beyond this row count, the result
is wrapped in ``{rows: [...], total: N, note: "..."}``.
"""


TOOL_RESULT_PERSIST_THRESHOLD_CHARS: int = _env_int("TOOL_RESULT_PERSIST_THRESHOLD_CHARS", 50_000)
"""Above this size, a tool result is persisted to disk + replaced with
preview.

The projector writes the full result to ``persist_dir/tool-output-<hash>.txt``
and the model sees only a preview + the file path so it can read more
on demand. Requires ``persist_dir`` to be set on the
:func:`looplet.scaffolding.truncate_tool_result` call.
"""


TOOL_RESULT_PREVIEW_CHARS: int = _env_int("TOOL_RESULT_PREVIEW_CHARS", 2000)
"""Length of the inline preview shown when a tool result is persisted.

Host integrations can use this hint; the shared projector independently
bounds the complete model-visible envelope, including any persist path.
"""


# ── Layer 2 - per-context-window aggregate cap ───────────────────


CONTEXT_WINDOW_STEPS: int = _env_int("CONTEXT_WINDOW_STEPS", 5)
"""How many recent steps to inline in ``state.context_summary()``.

Older steps roll out of the window and are visible only via compact
summaries (Layer 3). This is the sliding window before per-step caps.
"""


CONTEXT_INLINE_PER_STEP_CHARS: int = _env_int("CONTEXT_INLINE_PER_STEP_CHARS", 3000)
"""Default serialized-envelope budget shared by text and native result views.

Source previews can be shortened. Derived views must fit intact or be
replaced by an explicit delivery refusal. Tool render hints can override it.
"""


CONTEXT_WINDOW_TOTAL_CHARS: int = _env_int("CONTEXT_WINDOW_TOTAL_CHARS", 20_000)
"""Aggregate cap across all inlined steps in one prompt.

Text history shrinks the largest entries until the total fits. Native tool
result batches share the same limit. Derived views are refused rather than
partially delivered with a misleading continuation cursor.
"""


# ── Per-run overrides (set by composable_loop from LoopConfig) ─────
#
# Cartridges set ``LoopConfig.context_window_steps`` (etc.) to override
# the module-level defaults for one run without polluting global state.
# The loop pushes the override into a ``contextvars.ContextVar`` before
# yielding each step; ``state.context_summary()`` reads the override
# (falling back to the module constant). This keeps the env-default
# pathway working AND lets cartridges declare a window size in
# ``config.yaml`` instead of forcing every host process to set
# ``CONTEXT_WINDOW_STEPS`` at boot time.

import contextvars as _cv  # noqa: E402

_CONTEXT_WINDOW_STEPS_OVERRIDE: _cv.ContextVar[int | None] = _cv.ContextVar(
    "_CONTEXT_WINDOW_STEPS_OVERRIDE", default=None
)
_CONTEXT_INLINE_PER_STEP_CHARS_OVERRIDE: _cv.ContextVar[int | None] = _cv.ContextVar(
    "_CONTEXT_INLINE_PER_STEP_CHARS_OVERRIDE", default=None
)
_CONTEXT_WINDOW_TOTAL_CHARS_OVERRIDE: _cv.ContextVar[int | None] = _cv.ContextVar(
    "_CONTEXT_WINDOW_TOTAL_CHARS_OVERRIDE", default=None
)


def get_context_window_steps() -> int:
    """Effective recent-results window: per-run override or env default."""
    override = _CONTEXT_WINDOW_STEPS_OVERRIDE.get()
    return override if override is not None else CONTEXT_WINDOW_STEPS


def get_context_inline_per_step_chars() -> int:
    override = _CONTEXT_INLINE_PER_STEP_CHARS_OVERRIDE.get()
    return override if override is not None else CONTEXT_INLINE_PER_STEP_CHARS


def get_context_window_total_chars() -> int:
    override = _CONTEXT_WINDOW_TOTAL_CHARS_OVERRIDE.get()
    return override if override is not None else CONTEXT_WINDOW_TOTAL_CHARS


# ── Layer 3 - whole-conversation compact ──────────────────────────


COMPACT_TRIGGER_FRACTION: float = _env_float("COMPACT_TRIGGER_FRACTION", 0.75)
"""Fraction of model context window at which ``ThresholdCompactHook``
proactively triggers compaction.

When the loop's running token count exceeds this fraction of the
configured model context window, the hook calls the configured
``compact_service`` to summarize older history. Default 0.75.
"""


__all__ = [
    "COMPACT_TRIGGER_FRACTION",
    "CONTEXT_INLINE_PER_STEP_CHARS",
    "CONTEXT_WINDOW_STEPS",
    "CONTEXT_WINDOW_TOTAL_CHARS",
    "TOOL_RESULT_MAX_CHARS",
    "TOOL_RESULT_MAX_ROWS",
    "TOOL_RESULT_PERSIST_THRESHOLD_CHARS",
    "TOOL_RESULT_PREVIEW_CHARS",
    "get_context_inline_per_step_chars",
    "get_context_window_steps",
    "get_context_window_total_chars",
]
