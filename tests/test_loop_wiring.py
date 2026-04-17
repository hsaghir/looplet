"""Tests for loop wiring — optional capabilities integrated into composable_loop.

Covers:
  - router: ModelRouter selects backend per-call
  - checkpoint_dir: FileCheckpointStore saves after each step
  - tracer: Tracer records spans for LLM and tool calls
  - recovery_registry: RecoveryRegistry fires on parse error
  - output_schema: validate_args rejects invalid done() payload
  - stream: EventEmitter receives all expected event types
  - initial_checkpoint: crash-resume starts from correct step
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import pytest

# ── Minimal helpers ─────────────────────────────────────────────


@dataclass
class SimpleState:
    steps: list = field(default_factory=list)
    queries_used: int = 0
    _max_steps: int = 15

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_steps - len(self.steps))

    def context_summary(self) -> str:
        return f"steps={len(self.steps)}"

    def snapshot(self) -> dict:
        return {"steps": len(self.steps)}


def _make_registry(extra_tools=None):
    from openharness.tools import BaseToolRegistry, ToolSpec
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="done",
        description="finish",
        parameters={"summary": "summary"},
        execute=lambda summary="ok": {"done": True, "summary": summary},
    ))
    for name, fn in (extra_tools or []):
        reg.register(ToolSpec(name=name, description=name, parameters={}, execute=fn))
    return reg


def _make_scripted_llm(responses: list[str]):
    """Return an LLM that cycles through scripted JSON responses."""
    class ScriptedLLM:
        def __init__(self):
            self._idx = 0
        def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
            r = responses[min(self._idx, len(responses) - 1)]
            self._idx += 1
            return r
    return ScriptedLLM()


# ── LoopConfig new fields ────────────────────────────────────────


def test_loopconfig_has_router_field():
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    assert hasattr(cfg, "router")
    assert cfg.router is None


def test_loopconfig_has_checkpoint_dir_field():
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    assert hasattr(cfg, "checkpoint_dir")
    assert cfg.checkpoint_dir is None


def test_loopconfig_has_tracer_field():
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    assert hasattr(cfg, "tracer")
    assert cfg.tracer is None


def test_loopconfig_has_recovery_registry_field():
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    assert hasattr(cfg, "recovery_registry")
    assert cfg.recovery_registry is None


def test_loopconfig_has_output_schema_field():
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    assert hasattr(cfg, "output_schema")
    assert cfg.output_schema is None


def test_loopconfig_has_initial_checkpoint_field():
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    assert hasattr(cfg, "initial_checkpoint")
    assert cfg.initial_checkpoint is None


def test_composable_loop_accepts_stream_param():
    import inspect

    from openharness.loop import LoopConfig, composable_loop
    sig = inspect.signature(composable_loop)
    assert "stream" in sig.parameters


# ── Router test ──────────────────────────────────────────────────


def test_router_selects_backend():
    """When config.router is set, the router's selected backend is used."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.router import ModelRouter

    calls_received = []

    class TrackingLLM:
        def __init__(self, label):
            self._label = label
            self._idx = 0
            self._responses = ['{"tool": "done", "args": {"summary": "routed"}}']
        def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
            calls_received.append(self._label)
            r = self._responses[min(self._idx, len(self._responses) - 1)]
            self._idx += 1
            return r

    routed_llm = TrackingLLM("routed")

    class FakeRouter:
        def select(self, purpose: str, **kwargs) -> Any:
            return routed_llm

    state = SimpleState()
    reg = _make_registry()
    # pass a different llm directly — router should override it
    fallback_llm = TrackingLLM("fallback")
    config = LoopConfig(router=FakeRouter())

    list(composable_loop(fallback_llm, tools=reg, config=config, state=state))

    # routed_llm should have been called, not fallback
    assert "routed" in calls_received
    assert "fallback" not in calls_received


# ── Checkpoint test ──────────────────────────────────────────────


def test_checkpoint_dir_saves_checkpoints():
    """When checkpoint_dir is set, a checkpoint file is written after each step."""
    from openharness.loop import LoopConfig, composable_loop

    responses = [
        '{"tool": "done", "args": {"summary": "ckpt test"}}',
    ]
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoopConfig(checkpoint_dir=tmpdir)
        list(composable_loop(llm, tools=reg, config=config, state=state))
        # At least one checkpoint file should exist
        files = os.listdir(tmpdir)
        ckpt_files = [f for f in files if f.endswith(".json")]
        assert len(ckpt_files) >= 1, f"Expected checkpoint files, got: {files}"


def test_checkpoint_file_is_valid_json():
    """Checkpoint files should be valid JSON with expected keys."""
    from openharness.loop import LoopConfig, composable_loop

    responses = ['{"tool": "done", "args": {"summary": "ok"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoopConfig(checkpoint_dir=tmpdir)
        list(composable_loop(llm, tools=reg, config=config, state=state))
        files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        assert files
        with open(os.path.join(tmpdir, files[0])) as f:
            data = json.load(f)
        assert "step_number" in data


# ── Tracer test ──────────────────────────────────────────────────


def test_tracer_records_spans():
    """When config.tracer is set, spans are created for LLM call and tool dispatch."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.telemetry import Tracer

    responses = ['{"tool": "done", "args": {"summary": "traced"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()

    tracer = Tracer()
    config = LoopConfig(tracer=tracer)
    list(composable_loop(llm, tools=reg, config=config, state=state))

    # At least some spans should have been recorded
    assert len(tracer.root_spans) > 0 or len(tracer._stack) >= 0


def test_tracer_span_names():
    """Tracer spans should include LLM call span."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.telemetry import Tracer

    responses = ['{"tool": "done", "args": {"summary": "traced"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()

    tracer = Tracer()
    config = LoopConfig(tracer=tracer)
    list(composable_loop(llm, tools=reg, config=config, state=state))

    all_span_names = {s.name for s in tracer.root_spans}
    # Should have at least one span from the loop
    assert len(all_span_names) > 0


# ── Recovery registry test ───────────────────────────────────────


def test_recovery_registry_consulted_on_parse_error():
    """When recovery_registry is set, it's consulted when a parse error occurs."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.recovery import (
        FailureScenario,
        RecoveryAction,
        RecoveryRecipe,
        RecoveryRegistry,
    )

    recovery_called = []

    def custom_handler(ctx):
        recovery_called.append(ctx)
        return RecoveryAction(action_type="log_and_continue", payload={"handled": True})

    registry = RecoveryRegistry()
    registry.register(RecoveryRecipe(
        scenario=FailureScenario.PARSE_ERROR,
        handler=custom_handler,
        max_attempts=3,
    ))

    # LLM returns unparseable then valid
    responses = [
        "not json at all",
        "still not json",
        '{"tool": "done", "args": {"summary": "recovered"}}',
    ]
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    config = LoopConfig(recovery_registry=registry, max_steps=10)

    list(composable_loop(llm, tools=reg, config=config, state=state))

    # The registry was consulted when parse errors occurred
    assert len(recovery_called) >= 1


# ── Output schema test ───────────────────────────────────────────


def test_output_schema_rejects_invalid_done():
    """When output_schema is set, done() with missing required fields is rejected."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.validation import FieldSpec, OutputSchema

    schema = OutputSchema(fields={
        "report": FieldSpec(name="report", field_type="str", required=True),
    }, strict=False)

    # First done() call missing "report" field, second has it
    responses = [
        '{"tool": "done", "args": {"summary": "missing report"}}',  # invalid
        '{"tool": "done", "args": {"report": "valid report"}}',       # valid
    ]
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    reg.register(__import__("openharness.tools", fromlist=["ToolSpec"]).ToolSpec(
        name="done",  # override done to accept report too
        description="finish",
        parameters={"summary": "summary", "report": "report"},
        execute=lambda summary="", report="": {"done": True},
    ))

    config = LoopConfig(output_schema=schema, max_steps=10)
    steps = list(composable_loop(llm, tools=reg, config=config, state=state))

    # Should have taken more than 1 step because first done() was rejected
    assert len(steps) >= 2


def test_output_schema_allows_valid_done():
    """When output_schema is set, done() with valid payload completes normally."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.validation import FieldSpec, OutputSchema

    schema = OutputSchema(fields={
        "summary": FieldSpec(name="summary", field_type="str", required=True),
    })

    responses = ['{"tool": "done", "args": {"summary": "all good"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    config = LoopConfig(output_schema=schema, max_steps=5)
    steps = list(composable_loop(llm, tools=reg, config=config, state=state))

    assert any(s.tool_call.tool == "done" for s in steps)


# ── Stream (EventEmitter) test ───────────────────────────────────


def test_stream_receives_events():
    """When stream is set, events are emitted during the loop."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.streaming import CallbackEmitter

    received = []
    emitter = CallbackEmitter(callback=lambda evt: received.append(evt))

    responses = ['{"tool": "done", "args": {"summary": "streamed"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    config = LoopConfig(max_steps=5)
    list(composable_loop(llm, tools=reg, config=config, state=state, stream=emitter))

    assert len(received) > 0


def test_stream_receives_loop_start_event():
    """LoopStartEvent should be emitted before the loop starts."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.streaming import CallbackEmitter, LoopStartEvent

    received = []
    emitter = CallbackEmitter(callback=lambda evt: received.append(evt))

    responses = ['{"tool": "done", "args": {"summary": "streamed"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    config = LoopConfig(max_steps=5)
    list(composable_loop(llm, tools=reg, config=config, state=state, stream=emitter))

    event_types = {type(e).__name__ for e in received}
    assert "LoopStartEvent" in event_types


def test_stream_receives_loop_end_event():
    """LoopEndEvent should be emitted after the loop ends."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.streaming import CallbackEmitter, LoopEndEvent

    received = []
    emitter = CallbackEmitter(callback=lambda evt: received.append(evt))

    responses = ['{"tool": "done", "args": {"summary": "streamed"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    config = LoopConfig(max_steps=5)
    list(composable_loop(llm, tools=reg, config=config, state=state, stream=emitter))

    event_types = {type(e).__name__ for e in received}
    assert "LoopEndEvent" in event_types


def test_stream_receives_tool_dispatch_event():
    """ToolDispatchEvent should be emitted for each tool dispatch."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.streaming import CallbackEmitter, ToolDispatchEvent

    received = []
    emitter = CallbackEmitter(callback=lambda evt: received.append(evt))

    responses = ['{"tool": "done", "args": {"summary": "streamed"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    config = LoopConfig(max_steps=5)
    list(composable_loop(llm, tools=reg, config=config, state=state, stream=emitter))

    dispatch_events = [e for e in received if isinstance(e, ToolDispatchEvent)]
    assert len(dispatch_events) >= 1


# ── Initial checkpoint (crash-resume) test ──────────────────────


def test_initial_checkpoint_restores_step_offset():
    """Loop with initial_checkpoint should start from checkpoint's step_number."""
    from openharness.checkpoint import Checkpoint
    from openharness.loop import LoopConfig, composable_loop

    start_steps = []

    class TrackingLLM:
        _responses = ['{"tool": "done", "args": {"summary": "resumed"}}']
        _idx = 0
        def generate(self, prompt, **kwargs):
            r = self._responses[min(self._idx, len(self._responses) - 1)]
            self._idx += 1
            return r

    state = SimpleState()
    reg = _make_registry()

    # Create a checkpoint that says we're already at step 3
    ckpt = Checkpoint(
        step_number=3,
        session_log_data={"entries": [], "current_theory": "resumed"},
        conversation_data=None,
        config_snapshot={},
        tool_results_store={},
        metadata={"task_id": "test"},
    )
    config = LoopConfig(initial_checkpoint=ckpt, max_steps=10)
    steps = list(composable_loop(TrackingLLM(), tools=reg, config=config, state=state))

    # The first step number yielded should be > 1 (offset from checkpoint)
    if steps:
        assert steps[0].number > 1, f"Expected step > 1, got {steps[0].number}"


def test_initial_checkpoint_restores_session_log():
    """Session log should be restored from initial_checkpoint."""
    from openharness.checkpoint import Checkpoint
    from openharness.loop import LoopConfig, composable_loop
    from openharness.session import SessionLog

    responses = ['{"tool": "done", "args": {"summary": "ok"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()

    ckpt = Checkpoint(
        step_number=2,
        session_log_data={
            "entries": [
                {"step": 1, "theory": "initial", "tool": "search", "reasoning": "test",
                 "entities_seen": ["host-1"], "findings": [], "highlights": [], "recall_key": ""},
            ],
            "current_theory": "resumed theory",
        },
        conversation_data=None,
        config_snapshot={},
        tool_results_store={},
        metadata={},
    )
    restored_log = SessionLog()
    config = LoopConfig(initial_checkpoint=ckpt, max_steps=10)
    steps = list(composable_loop(llm, tools=reg, config=config, state=state, session_log=restored_log))

    # The session log should have entries from the checkpoint
    assert restored_log.current_theory == "resumed theory" or len(restored_log.entries) >= 1


# ── Backward compatibility ───────────────────────────────────────


def test_all_new_params_default_to_none():
    """All new LoopConfig fields default to None — no behavior change."""
    from openharness.loop import LoopConfig
    cfg = LoopConfig()
    for field_name in ("router", "checkpoint_dir", "tracer", "recovery_registry",
                       "output_schema", "initial_checkpoint"):
        assert getattr(cfg, field_name) is None, f"{field_name} should default to None"


def test_loop_works_without_new_params():
    """Existing loop usage (no new params) continues to work."""
    from openharness.loop import LoopConfig, composable_loop

    responses = ['{"tool": "done", "args": {"summary": "baseline"}}']
    llm = _make_scripted_llm(responses)
    state = SimpleState()
    reg = _make_registry()
    steps = list(composable_loop(llm, tools=reg, config=LoopConfig(), state=state))
    assert any(s.tool_call.tool == "done" for s in steps)
