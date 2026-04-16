"""End-to-end integration tests for cadence — all components working together.

Each test scenario exercises multiple cadence components simultaneously to
verify correct composition and cross-module integration.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.integration

# ── Helpers ───────────────────────────────────────────────────────


class _State:
    """Minimal AgentState for integration tests."""

    def __init__(self, max_steps: int = 10) -> None:
        self.steps: list = []
        self.queries_used: int = 0
        self.max_steps = max_steps

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    def context_summary(self) -> str:
        return "\n".join(
            f"Step {s.number}: {s.tool_call.tool}({s.tool_call.args})"
            for s in self.steps
        )

    def snapshot(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "queries_used": self.queries_used}


class _ScriptedLLM:
    """Scripted LLM backend: returns responses from a list, then repeats last."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        resp = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return resp


class _AsyncScriptedLLM:
    """Async scripted LLM backend."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        resp = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return resp


def _tool_response(tool: str, **args: Any) -> str:
    return json.dumps([{"tool": tool, "args": args}])


def _make_registry_with_done() -> Any:
    from openharness.tools import BaseToolRegistry, ToolSpec

    registry = BaseToolRegistry()
    registry.register(ToolSpec("think", "Record a thought", {"thought": "a thought to record"}, lambda thought="": thought))
    registry.register(ToolSpec("done", "Finish task", {"answer": "final answer"}, lambda answer="": answer))
    return registry


# ── Test 1: Full loop with all capabilities ───────────────────────


def test_full_loop_with_all_capabilities():
    """Run a 3-step mock loop with all capabilities active simultaneously."""
    from openharness.checkpoint import Checkpoint, FileCheckpointStore
    from openharness.context import ContextManagerHook
    from openharness.loop import LoopConfig, composable_loop
    from openharness.streaming import CallbackEmitter, LoopEndEvent, LoopStartEvent, StreamingHook
    from openharness.telemetry import MetricsCollector, Tracer

    llm_responses = [
        _tool_response("think", thought="step 1"),
        _tool_response("think", thought="step 2"),
        _tool_response("done", answer="finished"),
    ]
    llm = _ScriptedLLM(llm_responses)
    registry = _make_registry_with_done()
    state = _State(max_steps=10)

    tracer = Tracer()
    collector = MetricsCollector()
    events: list = []
    emitter = CallbackEmitter(events.append)

    # Custom hooks compatible with loop.py's hook calling convention
    class InstrumentHook:
        def pre_loop(self, st: Any, log: Any, ctx: Any) -> None:
            self._span = tracer.start_span("loop.run")

        def post_dispatch(
            self, st: Any, log: Any, tc: Any, tr: Any, step_num: int
        ) -> None:
            collector.record_step(
                tool_name=tc.tool,
                classification="productive",
                input_tokens=0,
                output_tokens=0,
                duration_ms=0.0,
                has_error=tr.error is not None,
            )

        def on_loop_end(self, st: Any, log: Any, ctx: Any, llm_backend: Any) -> int:
            if hasattr(self, "_span"):
                tracer.end_span(self._span)
            return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileCheckpointStore(tmpdir)
        saved_checkpoints: list = []

        # Checkpoint hook compatible with loop.py's hook calling convention
        class SimpleCheckpointHook:
            def post_dispatch(
                self, st: Any, log: Any, tc: Any, tr: Any, step_num: int
            ) -> None:
                cp = Checkpoint(
                    step_number=step_num,
                    session_log_data={},
                    conversation_data=None,
                    config_snapshot={},
                    tool_results_store={},
                    metadata={},
                )
                store.save(cp, f"step_{step_num}")
                saved_checkpoints.append(step_num)

        hooks = [
            ContextManagerHook(llm=None),
            InstrumentHook(),
            StreamingHook(emitter),
            SimpleCheckpointHook(),
        ]
        config = LoopConfig(max_steps=10, done_tool="done")

        steps = list(
            composable_loop(llm, task={"description": "test"}, tools=registry, hooks=hooks, config=config, state=state)
        )

        # Checkpoint files written for each step
        checkpoint_files = list(Path(tmpdir).glob("*.json"))
        assert len(checkpoint_files) >= 1

    # No error — loop completed
    assert len(steps) >= 1

    # Metrics recorded via custom hook
    assert collector.total_steps >= 1

    # Spans recorded via Tracer
    assert len(tracer.root_spans) >= 1

    # Stream events: LoopStartEvent first, LoopEndEvent last
    event_types = [type(e).__name__ for e in events]
    assert "LoopStartEvent" in event_types
    assert "LoopEndEvent" in event_types
    assert event_types[0] == "LoopStartEvent"
    assert event_types[-1] == "LoopEndEvent"


# ── Test 2: Checkpoint resume ─────────────────────────────────────


def test_checkpoint_resume():
    """Run 2-step loop, save checkpoint, verify resume gives step_offset > 0."""
    from openharness.checkpoint import Checkpoint, FileCheckpointStore, resume_loop_state
    from openharness.loop import LoopConfig, composable_loop
    from openharness.session import SessionLog

    llm = _ScriptedLLM([
        _tool_response("think", thought="first step"),
        _tool_response("done", answer="done"),
    ])
    registry = _make_registry_with_done()
    state = _State(max_steps=10)
    session_log = SessionLog()

    config = LoopConfig(max_steps=10, done_tool="done")
    steps = list(composable_loop(llm, task={}, tools=registry, config=config, state=state, session_log=session_log))
    completed_steps = len(steps)
    assert completed_steps >= 1

    # Save a checkpoint at the completed step count
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileCheckpointStore(tmpdir)
        checkpoint = Checkpoint(
            step_number=completed_steps,
            session_log_data=session_log.to_dict() if hasattr(session_log, "to_dict") else {},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={"task": "test"},
        )
        store.save(checkpoint, "step_2")

        # Load it back
        loaded = store.load("step_2")
        assert loaded is not None
        assert loaded.step_number == completed_steps

        # Resume — step_offset should be > 0
        resume_state = resume_loop_state(loaded)
        assert resume_state["step_offset"] > 0
        assert resume_state["step_offset"] == completed_steps


# ── Test 3: Router model selection ────────────────────────────────


def test_router_model_selection():
    """SimpleRouter selects correct backend based on purpose tag."""
    from openharness.router import ModelProfile, SimpleRouter

    backend_a = _ScriptedLLM(["response_a"])
    backend_b = _ScriptedLLM(["response_b"])

    profiles = {
        "reasoning": ModelProfile(name="model-a", backend=backend_a),
        "recovery": ModelProfile(name="model-b", backend=backend_b),
    }
    router = SimpleRouter(profiles, default_profile=profiles["reasoning"])

    selected_reasoning = router.select("reasoning")
    selected_recovery = router.select("recovery")
    selected_unknown = router.select("unknown_purpose")

    assert selected_reasoning is backend_a
    assert selected_recovery is backend_b
    assert selected_unknown is backend_a  # default


# ── Test 4: Recovery fires on parse error ─────────────────────────


def test_recovery_fires_on_parse_error():
    """Custom recovery strategy executes when PARSE_ERROR triggered."""
    from openharness.recovery import FailureScenario, RecoveryAction, RecoveryRecipe, RecoveryRegistry

    fired: list[dict] = []

    def custom_handler(ctx: dict) -> RecoveryAction:
        fired.append(ctx)
        return RecoveryAction(action_type="inject_guidance", payload={"hint": "use valid JSON"})

    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.PARSE_ERROR,
        handler=custom_handler,
        max_attempts=3,
        description="test parse error recovery",
    )
    registry.register(recipe)

    # Trigger recovery
    action = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {"error": "invalid json", "step": 1})
    assert action is not None
    assert action.action_type == "inject_guidance"
    assert len(fired) == 1
    assert fired[0]["error"] == "invalid json"

    # Respects max_attempts
    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {"error": "again", "step": 2})
    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {"error": "third", "step": 3})
    exhausted = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {"error": "fourth", "step": 4})
    assert exhausted is None  # max_attempts=3 exceeded


# ── Test 5: Validation rejects bad done payload ───────────────────


def test_validation_rejects_bad_done():
    """Loop continues (doesn't terminate) when done() payload fails schema validation."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.validation import FieldSpec, OutputSchema, ValidatingToolRegistry

    # Mock LLM: returns done without required 'summary' field, then valid done
    llm = _ScriptedLLM([
        _tool_response("done"),               # missing 'summary' — should fail validation
        _tool_response("done", summary="ok"),  # valid done
    ])

    # ValidatingToolRegistry owns its own internal registry
    val_registry = ValidatingToolRegistry()
    from openharness.tools import ToolSpec
    val_registry.register(ToolSpec("think", "Record thought", {"thought": "a thought"}, lambda thought="": thought))

    schema = OutputSchema(
        fields={"summary": FieldSpec("summary", str, required=True)},
    )
    val_registry.register_with_schema(
        ToolSpec("done", "Finish task", {"summary": "final summary"}, lambda summary="": summary),
        schema,
    )

    state = _State(max_steps=5)
    config = LoopConfig(max_steps=5, done_tool="done")

    steps = list(
        composable_loop(llm, task={}, tools=val_registry, config=config, state=state)
    )

    # The loop should have run at least 1 step (and the first done call should fail validation)
    assert len(steps) >= 1
    # The first step should be the failed done attempt (returns error ToolResult)
    first_done_step = steps[0]
    assert first_done_step.tool_result.error is not None
    assert "Validation failed" in first_done_step.tool_result.error


# ── Test 6: Async loop with ContextManagerHook ────────────────────


async def test_async_loop_with_context():
    """async_composable_loop with ContextManagerHook completes without RuntimeError."""
    from openharness.async_loop import async_composable_loop
    from openharness.context import ContextManagerHook

    llm = _AsyncScriptedLLM([
        _tool_response("think", thought="async step"),
        _tool_response("done", answer="async done"),
    ])

    registry = _make_registry_with_done()
    state = _State(max_steps=10)
    ctx_hook = ContextManagerHook(llm=None)

    from openharness.loop import LoopConfig

    config = LoopConfig(max_steps=10, done_tool="done")

    steps = []
    async for step in async_composable_loop(
        llm, task={}, tools=registry, hooks=[ctx_hook], config=config, state=state
    ):
        steps.append(step)

    # Completed without RuntimeError
    assert len(steps) >= 1


# ── Test 7: Streaming event sequence ─────────────────────────────


def test_streaming_event_sequence():
    """LoopStartEvent arrives first; LoopEndEvent arrives last."""
    from openharness.loop import LoopConfig, composable_loop
    from openharness.streaming import (
        CallbackEmitter,
        LoopEndEvent,
        LoopStartEvent,
        StepStartEvent,
        ToolDispatchEvent,
        StreamingHook,
    )

    llm = _ScriptedLLM([
        _tool_response("think", thought="checking"),
        _tool_response("done", answer="done"),
    ])
    registry = _make_registry_with_done()
    state = _State(max_steps=10)

    events: list = []
    hook = StreamingHook(CallbackEmitter(events.append))
    config = LoopConfig(max_steps=10, done_tool="done")

    list(composable_loop(llm, task={}, tools=registry, hooks=[hook], config=config, state=state))

    event_types = [type(e).__name__ for e in events]
    assert len(events) >= 3

    # LoopStartEvent first
    assert event_types[0] == "LoopStartEvent"

    # LoopEndEvent last
    assert event_types[-1] == "LoopEndEvent"

    # StepStartEvent and ToolDispatchEvent appear in the middle
    assert "StepStartEvent" in event_types
    assert "ToolDispatchEvent" in event_types

    # StepStartEvent appears before ToolDispatchEvent in every step
    for i, etype in enumerate(event_types):
        if etype == "ToolDispatchEvent":
            # There must be a StepStartEvent before this
            preceding = event_types[:i]
            assert "StepStartEvent" in preceding
