"""Smoke tests for the three-layer context budget system.

Layer 1 — per-tool-result cap (``TOOL_RESULT_MAX_CHARS``,
``TOOL_RESULT_PERSIST_THRESHOLD_CHARS``).
Layer 2 — per-context-window aggregate cap
(``CONTEXT_WINDOW_STEPS``, ``CONTEXT_INLINE_PER_STEP_CHARS``,
``CONTEXT_WINDOW_TOTAL_CHARS``).
Layer 3 — whole-conversation compact (existing
:class:`looplet.compact.CompactService`; not exercised here, only
the tunable's existence is asserted).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from looplet import context_budget
from looplet.types import DefaultState, Step, ToolCall, ToolResult


def _make_step(number: int, tool: str, data: Any) -> Step:
    return Step(
        number=number,
        tool_call=ToolCall(tool=tool, args={}),
        tool_result=ToolResult(tool=tool, args_summary="", data=data),
    )


# ── Layer 2: context_summary inlines actual data ────────────────


def test_context_summary_inlines_full_data() -> None:
    """The model must SEE the data, not a digest. Regression for the
    fabrication bug where ``commits: (5)`` left the LLM no choice but
    to invent commit shas on the next turn."""
    state = DefaultState()
    commits = [
        {"sha": "7321837", "message": "fix: third-pass audit"},
        {"sha": "7851476", "message": "fix: 10 second-pass audit findings"},
    ]
    state.steps.append(_make_step(1, "fetch_commits", {"commits": commits}))
    summary = state.context_summary()
    # The actual sha values must appear in the LLM-facing string.
    assert "7321837" in summary
    assert "7851476" in summary
    # Tool name and step number too.
    assert "fetch_commits" in summary
    assert "S1" in summary


def test_context_summary_per_step_cap_truncates_with_marker() -> None:
    """A single huge step gets truncated with a clear marker."""
    state = DefaultState()
    big = "x" * 10_000  # bigger than default CONTEXT_INLINE_PER_STEP_CHARS=3000
    state.steps.append(_make_step(1, "fetch", {"blob": big}))
    summary = state.context_summary()
    assert "[truncated; full result" in summary
    # Per-step cap is 3000 chars by default; resulting block ~3000+marker.
    assert len(summary) < 4000


def test_context_summary_window_steps_limits_recent() -> None:
    """Only the last ``CONTEXT_WINDOW_STEPS`` steps appear; older are
    out of scope (Layer 3 / compact handles them)."""
    state = DefaultState()
    for i in range(1, 11):  # 10 steps, names 'tool_a'..'tool_j'
        state.steps.append(_make_step(i, f"tool_{chr(96 + i)}", {"v": i}))
    summary = state.context_summary()
    # Default window = 5; steps 6..10 (tool_f..tool_j) appear, 1..5 (tool_a..tool_e) do not.
    assert "tool_j" in summary  # step 10
    assert "tool_f" in summary  # step 6
    assert "tool_e" not in summary  # step 5 is out of window
    assert "tool_a" not in summary  # step 1 is out of window


def test_context_summary_aggregate_cap_shrinks_largest() -> None:
    """When aggregate exceeds budget, the largest contributor shrinks."""
    state = DefaultState()
    # Five steps; one is far larger than the others.
    for i in range(1, 5):
        state.steps.append(_make_step(i, f"small_{i}", {"v": i}))
    state.steps.append(_make_step(5, "huge_tool", {"blob": "Y" * 20_000}))

    # Tighten the aggregate cap so the test exercises shrinking.
    with patch.object(context_budget, "CONTEXT_WINDOW_TOTAL_CHARS", 5000):
        summary = state.context_summary()
    assert len(summary) <= 6000  # cap + small overhead
    # Most-recent step is the huge one; it must remain visible (truncated).
    assert "huge_tool" in summary
    # Small steps are visible, full.
    assert "small_4" in summary


def test_context_summary_error_block_format() -> None:
    state = DefaultState()
    state.steps.append(
        Step(
            number=1,
            tool_call=ToolCall(tool="x", args={}),
            tool_result=ToolResult(tool="x", args_summary="", data=None, error="boom"),
        )
    )
    summary = state.context_summary()
    assert "✗" in summary
    assert "boom" in summary


def test_context_summary_empty_state() -> None:
    state = DefaultState()
    assert state.context_summary() == ""


# ── env override ────────────────────────────────────────────────


def test_env_override_tool_result_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LOOPLET_TOOL_RESULT_MAX_CHARS`` overrides the default."""
    monkeypatch.setenv("LOOPLET_TOOL_RESULT_MAX_CHARS", "12345")
    # Re-import to pick up new env value.
    importlib.reload(context_budget)
    try:
        assert context_budget.TOOL_RESULT_MAX_CHARS == 12345
    finally:
        monkeypatch.delenv("LOOPLET_TOOL_RESULT_MAX_CHARS", raising=False)
        importlib.reload(context_budget)


def test_env_override_unparseable_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A garbage env value must NOT crash the process; we fall back to
    the hardcoded default."""
    monkeypatch.setenv("LOOPLET_TOOL_RESULT_MAX_CHARS", "not-a-number")
    importlib.reload(context_budget)
    try:
        assert context_budget.TOOL_RESULT_MAX_CHARS == 6000  # the default
    finally:
        monkeypatch.delenv("LOOPLET_TOOL_RESULT_MAX_CHARS", raising=False)
        importlib.reload(context_budget)


def test_env_override_compact_trigger_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPLET_COMPACT_TRIGGER_FRACTION", "0.85")
    importlib.reload(context_budget)
    try:
        assert context_budget.COMPACT_TRIGGER_FRACTION == 0.85
    finally:
        monkeypatch.delenv("LOOPLET_COMPACT_TRIGGER_FRACTION", raising=False)
        importlib.reload(context_budget)


# ── Layer 1: persist-and-preview is wired through LoopConfig ────


def test_loopconfig_has_tool_result_persist_dir() -> None:
    """``LoopConfig`` exposes a knob for the persist directory."""
    from looplet.loop import LoopConfig

    cfg = LoopConfig()
    assert cfg.tool_result_persist_dir is None
    cfg2 = LoopConfig(tool_result_persist_dir="/tmp/looplet-tool-results")
    assert cfg2.tool_result_persist_dir == "/tmp/looplet-tool-results"


def test_truncate_tool_result_persists_when_threshold_exceeded(tmp_path: Path) -> None:
    """When the result is huge AND persist_dir is set, full output goes
    to disk and the model sees a preview + the path."""
    from looplet.scaffolding import truncate_tool_result

    big = "Z" * 60_000
    result = truncate_tool_result(
        {"big_string": big},
        persist_dir=str(tmp_path),
        persist_threshold=50_000,
    )
    assert isinstance(result, dict)
    assert "persisted_output_path" in result
    assert "truncated_output" in result
    persisted = Path(result["persisted_output_path"])
    assert persisted.exists()
    # Full content is on disk.
    assert big in persisted.read_text()


def test_truncate_tool_result_no_persist_when_dir_unset() -> None:
    """Without persist_dir, only inline truncation applies."""
    from looplet.scaffolding import truncate_tool_result

    big = {"k": "Y" * 60_000}
    result = truncate_tool_result(big)
    # No persistence keys.
    assert "persisted_output_path" not in (result or {})
