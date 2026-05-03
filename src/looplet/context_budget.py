"""Central, env-overridable tunables for context-budget management.

Looplet's context pipeline has THREE nested budgets that mirror Claude
Code's design:

1. **Per-tool-result cap** — at dispatch time, ``truncate_tool_result``
   caps any single tool's ``data`` to :data:`TOOL_RESULT_MAX_CHARS`.
   Persist-and-preview kicks in when the result exceeds
   :data:`TOOL_RESULT_PERSIST_THRESHOLD_CHARS` and a persist directory
   is configured.

2. **Per-context-window aggregate cap** — at LLM-prompt assembly time,
   ``state.context_summary()`` shows the last
   :data:`CONTEXT_WINDOW_STEPS` steps with each step's data inlined
   (truncated to :data:`CONTEXT_INLINE_PER_STEP_CHARS` per step). When
   the aggregate exceeds :data:`CONTEXT_WINDOW_TOTAL_CHARS`, the
   largest entries get persist-previewed until the total fits.

3. **Whole-conversation compact** — when the loop's reactive-compact
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

    Returns ``default`` when the var is unset, empty, or unparseable —
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


# ── Layer 1 — per-tool-result cap ─────────────────────────────────


TOOL_RESULT_MAX_CHARS: int = _env_int("TOOL_RESULT_MAX_CHARS", 6000)
"""Max characters of any single tool result before in-result truncation kicks in.

The dispatcher calls :func:`looplet.scaffolding.truncate_tool_result`
on every result; that function applies this cap. Choose lower values
(2-4K) to keep the LLM context tight; higher values (12-50K, mirroring
Claude Code's 50K) when tool results are the agent's primary signal.
"""


TOOL_RESULT_MAX_ROWS: int = _env_int("TOOL_RESULT_MAX_ROWS", 50)
"""Max rows in a list-typed tool result before list truncation kicks in.

Applied when ``data`` is a ``list``. Beyond this row count, the result
is wrapped in ``{rows: [...], total: N, note: "..."}``.
"""


TOOL_RESULT_PERSIST_THRESHOLD_CHARS: int = _env_int("TOOL_RESULT_PERSIST_THRESHOLD_CHARS", 50_000)
"""Above this size, a tool result is persisted to disk + replaced with
preview.

Mirrors Claude Code's ``DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000``.
The dispatcher writes the full result to ``persist_dir/tool-output-<hash>.txt``
and the model sees only a preview + the file path so it can read more
on demand. Requires ``persist_dir`` to be set on the
:func:`looplet.scaffolding.truncate_tool_result` call.
"""


TOOL_RESULT_PREVIEW_CHARS: int = _env_int("TOOL_RESULT_PREVIEW_CHARS", 2000)
"""Length of the inline preview shown when a tool result is persisted.

The model gets the first ``TOOL_RESULT_PREVIEW_CHARS`` characters of the
serialized result, plus the persist-path. Mirrors Claude Code's
``PREVIEW_SIZE_BYTES = 2000``.
"""


# ── Layer 2 — per-context-window aggregate cap ───────────────────


CONTEXT_WINDOW_STEPS: int = _env_int("CONTEXT_WINDOW_STEPS", 5)
"""How many recent steps to inline in ``state.context_summary()``.

Older steps roll out of the window and are visible only via compact
summaries (Layer 3). This is the sliding window before per-step caps.
"""


CONTEXT_INLINE_PER_STEP_CHARS: int = _env_int("CONTEXT_INLINE_PER_STEP_CHARS", 3000)
"""Per-step soft cap when serializing tool results into the context window.

Each step's result is JSON-serialized; if the serialization exceeds
this cap, it gets per-step truncation with a "[truncated; full result
N chars]" tail. Independent of Layer 1 — Layer 1 caps at dispatch
time, this layer caps at prompt-assembly time.
"""


CONTEXT_WINDOW_TOTAL_CHARS: int = _env_int("CONTEXT_WINDOW_TOTAL_CHARS", 20_000)
"""Aggregate cap across all inlined steps in one prompt.

When the assembled context_summary exceeds this, the largest steps in
the window are truncated to free budget until the total fits. Mirrors
Claude Code's ``MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200_000``,
scaled down for typical looplet workloads.
"""


# ── Layer 3 — whole-conversation compact ──────────────────────────


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
]
