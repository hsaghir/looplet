"""Smoke tests for the three-layer context budget system.

Layer 1 - per-tool-result cap (``TOOL_RESULT_MAX_CHARS``,
``TOOL_RESULT_PERSIST_THRESHOLD_CHARS``).
Layer 2 - per-context-window aggregate cap
(``CONTEXT_WINDOW_STEPS``, ``CONTEXT_INLINE_PER_STEP_CHARS``,
``CONTEXT_WINDOW_TOTAL_CHARS``).
Layer 3 - whole-conversation compact (existing
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("temperature", [None, 0.7])
async def test_parse_recovery_sampling_is_unset_unless_configured(asynchronous, temperature):
    from looplet import LoopConfig, RunResult, async_composable_loop, composable_loop, tools_from

    requests = []

    class Backend:
        def generate(self, prompt, **kwargs):
            requests.append(kwargs.get("temperature"))
            if len(requests) == 1:
                return "not a tool call"
            return '{"tool":"done","args":{"summary":"checked"}}'

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    options = {} if temperature is None else {"recovery_temperature": temperature}
    state = DefaultState(max_steps=1)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([], include_done=True),
        state=state,
        config=LoopConfig(max_steps=1, use_native_tools=False, **options),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))
    assert RunResult.from_state(state).completed
    assert requests == [None, temperature]


def test_deterministic_compaction_retains_bounded_tool_receipts():
    from looplet.conversation import Message, MessageRole, default_summarizer

    data = {
        "batch": 0,
        "paid_total": 1136,
        "completed_records": ["batch-0.json"],
        "archive": "x" * 20_000,
    }
    result = ToolResult(
        tool="record_invoice_total",
        args_summary="batch=0, paid_total=1136",
        data=data,
        result_key="receipt-1",
    )
    summary = default_summarizer([Message(role=MessageRole.TOOL, content="", tool_result=result)])
    assert '"completed_records": ["batch-0.json"]' in summary
    assert '"result_key": "receipt-1"' in summary
    assert "data, not instructions" in summary
    assert "[truncated; full result in run evidence]" in summary
    assert len(summary) < 1200
    assert result.data is data and len(data["archive"]) == 20_000


@pytest.mark.parametrize("profile", ["execution-v1", "standard-v1"])
@pytest.mark.parametrize("native_tools", [False, True])
def test_default_prompt_matches_protocol_and_explicit_policy(profile, native_tools):
    from looplet import LoopConfig, composable_loop, tools_from

    prompts = []

    class Backend:
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return '{"tool":"done","args":{"summary":"checked"}}'

        def generate_with_tools(self, prompt, **kwargs):
            prompts.append(prompt)
            return [
                {
                    "type": "tool_use",
                    "id": "finish",
                    "name": "done",
                    "input": {"summary": "checked"},
                }
            ]

    list(
        composable_loop(
            llm=Backend(),
            tools=tools_from([], include_done=True),
            config=LoopConfig(max_steps=1, use_native_tools=native_tools, harness_profile=profile),
            task={"goal": "Finish"},
        )
    )
    prompt = prompts[0]
    assert ("Respond with JSON only" in prompt) is (not native_tools)
    assert ("provider's native tool format" in prompt) is native_tools
    assert ("LOW BUDGET" in prompt) is (profile == "standard-v1")


@pytest.mark.parametrize("capability", [None, False, True, "yes"])
@pytest.mark.parametrize("native_tools", [False, True])
def test_stateful_helper_requires_explicit_stateless_capability(capability, native_tools):
    from looplet import LoopConfig
    from looplet.assistant_turns import _AssistantTurns
    from looplet.conversation import Conversation
    from looplet.scaffolding import TerminalLLMError

    class Backend:
        conversation_mode = "stateful"

        def __init__(self):
            self.requests = []
            self.snapshots = []
            if capability is not None:
                self.supports_stateless_text_generation = capability

        def set_conversation_input(self, snapshot):
            self.snapshots.append(snapshot)

        def generate(self, prompt, **kwargs):
            self.requests.append(prompt)
            return "approved"

        def generate_with_tools(self, prompt, **kwargs):
            self.requests.append(prompt)
            return []

    turns = _AssistantTurns(DefaultState(max_steps=2), LoopConfig(max_model_calls=4))
    backend = Backend()
    admitted = turns.wrap(backend)
    admitted.prepare_conversation(Conversation(), "parent request")
    method = admitted.generate_with_tools if native_tools else admitted.generate
    if capability is True and not native_tools:
        assert method("independent review") == "approved"
        assert backend.requests == ["independent review"]
        assert backend.snapshots == [None]
        assert turns.data["model_calls"] == 1
    else:
        with pytest.raises(TerminalLLMError, match="unsupported_context"):
            method("independent review")
        assert backend.requests == []
        assert turns.data["model_calls"] == 0


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_small_context_compacts_without_losing_raw_evidence(tmp_path, asynchronous):
    from looplet import MockLLMBackend, RunResult
    from looplet.examples.default_harness import (
        EventObserver,
        build_agent,
        grade_deliverable,
        scripted_responses,
    )

    class MessagesBackend(MockLLMBackend):
        conversation_mode = "messages"

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

    class AsyncMessagesBackend(MessagesBackend):
        async def generate_with_tools(self, *args, **kwargs):
            return super().generate_with_tools(*args, **kwargs)

    backend_type = AsyncMessagesBackend if asynchronous else MessagesBackend
    backend = backend_type(scripted_responses("invoices"), cycle=False)
    preset = build_agent("invoices", tmp_path / "deliverable")
    observer = EventObserver()
    preset.hooks.append(observer)
    with preset:
        if asynchronous:
            async for _ in preset.run_async(backend, task={"goal": "Reconcile the three batches"}):
                pass
            result = RunResult.from_state(preset.state)
        else:
            result = preset.run_result(backend, task={"goal": "Reconcile the three batches"})
        assert result.completed, result.diagnostics()
        assert grade_deliverable("invoices", tmp_path / "deliverable")["passed"]
        sources = [
            step.tool_result.data
            for step in result.steps
            if step.tool_call.tool == "read_invoice_batch"
        ]
        assert len(sources) == 3
        assert all(len(source["audit_notes"]) > 16_000 for source in sources)
        assert "post_compact" in [event.lower() for event in observer.events]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("persist", [False, True])
async def test_prompt_projection_preserves_original_run_evidence(tmp_path, asynchronous, persist):
    from looplet import LoopConfig, async_composable_loop, composable_loop, tools_from
    from looplet.checkpoint import Checkpoint
    from looplet.conversation import MessageRole
    from looplet.tools import BaseToolRegistry

    original = "evidence" * 10_000

    def source() -> str:
        return original

    class Backend:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            name = "source" if self.calls == 1 else "done"
            return '{"tool": "' + name + '", "args": {}}'

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    state = DefaultState(max_steps=3)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([source], include_done=True),
        state=state,
        config=LoopConfig(
            max_steps=3,
            use_native_tools=False,
            tool_result_persist_dir=str(tmp_path) if persist else None,
        ),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))

    assert state.steps[0].tool_result.data == original
    checkpoint = Checkpoint(
        step_number=state.step_count,
        session_log_data={},
        conversation_data=state.conversation.serialize(),
        config_snapshot={},
        tool_results_store=arguments["tools"].snapshot_results(),
        metadata={},
    )
    restored = BaseToolRegistry()
    restored.restore_results(Checkpoint.from_dict(checkpoint.to_dict()).tool_results_store)
    assert restored.get_result(state.steps[0].tool_result.result_key) == original
    projected = next(
        message.tool_result
        for message in state.conversation.messages
        if message.role == MessageRole.TOOL
    )
    assert projected is not state.steps[0].tool_result
    assert projected.data != original
    if persist:
        assert original in Path(projected.data["persisted_output_path"]).read_text()


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


@pytest.mark.parametrize("window", [0, -1])
def test_context_summary_nonpositive_window_is_empty(window: int) -> None:
    state = DefaultState()
    state.steps.append(_make_step(1, "tool_a", {"v": 1}))
    token = context_budget._CONTEXT_WINDOW_STEPS_OVERRIDE.set(window)
    try:
        assert state.context_summary() == ""
    finally:
        context_budget._CONTEXT_WINDOW_STEPS_OVERRIDE.reset(token)


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
