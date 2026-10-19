"""Tests for looplet.checkpoint - save/restore loop state."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from looplet import (
    AsyncMockLLMBackend,
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    RunEnvelope,
    ToolSpec,
    async_composable_loop,
    composable_loop,
    register_done_tool,
)
from looplet.checkpoint import (
    Checkpoint,
    CheckpointHook,
    CheckpointStore,
    FileCheckpointStore,
    resume_loop_state,
)
from looplet.session import SessionLog

# ── Helpers ────────────────────────────────────────────────────────


def _make_checkpoint(step: int = 3, task_id: str = "task-1") -> Checkpoint:
    log = SessionLog()
    log.record(1, "initial theory", "search", "looking for data", entities=["host-a"])
    log.record(2, "updated theory", "query", "refining search", findings=["found 42 events"])
    log.record(3, "updated theory", "done", "finished", highlights=["critical"])
    return Checkpoint(
        step_number=step,
        session_log_data={
            "entries": log.to_list(),
            "current_theory": log.current_theory,
        },
        conversation_data=None,
        config_snapshot={
            "max_steps": 15,
            "max_tokens": 2000,
            "temperature": 0.2,
            "done_tool": "done",
            "system_prompt": "",
        },
        tool_results_store={"step_1_search": {"rows": [{"id": 1}]}},
        metadata={"task_id": task_id, "version": "0.1.0", "timestamp": time.time()},
    )


# ── Checkpoint dataclass ────────────────────────────────────────────


def test_message_only_checkpoint_order_and_required_semantics(tmp_path):
    from looplet.assistant_turns import TURN_STATE_KEY, TURN_STATE_SCHEMA

    store = FileCheckpointStore(tmp_path)
    older = _make_checkpoint(step=0)
    older.metadata[TURN_STATE_KEY] = {
        "schema": TURN_STATE_SCHEMA,
        "turn_count": 1,
        "model_calls": 1,
    }
    newer = _make_checkpoint(step=0)
    newer.metadata[TURN_STATE_KEY] = {
        "schema": TURN_STATE_SCHEMA,
        "turn_count": 2,
        "model_calls": 2,
    }
    newer.created_at = older.created_at - 1
    store.save(older, "z_old")
    store.save(newer, "a_new")
    assert store.load_latest().metadata[TURN_STATE_KEY]["turn_count"] == 2
    assert store.load_latest_for_resume().metadata[TURN_STATE_KEY]["turn_count"] == 2
    payload = newer.to_dict()
    assert payload["step_number"] == {
        "tool_step": 0,
        "required_semantics": TURN_STATE_SCHEMA,
    }
    assert Checkpoint.from_dict(payload).step_number == 0
    payload["step_number"]["required_semantics"] = "unknown"
    with pytest.raises(ValueError, match="required semantics"):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("routed", [False, True])
async def test_message_checkpoint_resume_preserves_context_and_terminal_output(
    tmp_path,
    monkeypatch,
    asynchronous,
    routed,
):
    from dataclasses import replace
    from types import SimpleNamespace

    from looplet import RunResult, tools_from
    from looplet.assistant_turns import TURN_STATE_KEY, AssistantTurnDecision

    class Crash(BaseException):
        pass

    calls = []
    effects = []
    candidate = "X" * 6000

    class Backend:
        last_stop_reason = "stop"

        def __init__(self):
            self.items = []

        def checkpoint_state(self):
            return {"items": self.items}

        def restore_checkpoint_state(self, data):
            self.items = data["items"]

        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            self.items.append(len(self.items) + 1)
            return (
                "Retained hypothesis"
                if len(self.items) == 1
                else json.dumps({"tool": "done", "args": {"answer": candidate}})
            )

    def make_backend():
        backend = Backend()
        if asynchronous:
            original = backend.generate

            async def generate(prompt, **kwargs):
                return original(prompt, **kwargs)

            backend.generate = generate
        return backend

    def done(answer: str):
        effects.append(answer)
        return {"answer": answer}

    config = LoopConfig(
        max_steps=3,
        max_model_calls=3,
        use_native_tools=False,
        checkpoint_dir=tmp_path,
        assistant_turn_policy=lambda context: AssistantTurnDecision(
            "continue", feedback="Check the evidence"
        ),
    )
    original_save = FileCheckpointStore.save
    snapshots = []

    def interrupt_after_commit(store, checkpoint, key):
        original_save(store, checkpoint, key)
        turns = checkpoint.metadata.get(TURN_STATE_KEY, {})
        if turns.get("decisions") and not turns.get("pending_model_call"):
            snapshots.append(checkpoint)
            raise Crash()

    monkeypatch.setattr(FileCheckpointStore, "save", interrupt_after_commit)
    first = make_backend()

    async def run(backend, state):
        active_config = (
            replace(config, router=SimpleNamespace(select=lambda **kwargs: backend))
            if routed
            else config
        )
        kwargs = dict(
            llm=make_backend() if routed else backend,
            state=state,
            tools=tools_from([done]),
            config=active_config,
        )
        return (
            [step async for step in async_composable_loop(**kwargs)]
            if asynchronous
            else list(composable_loop(**kwargs))
        )

    with pytest.raises(Crash):
        await run(first, DefaultState(max_steps=3))
    assert len(calls) == 1
    assert effects == []
    assert snapshots[0].step_number == 0
    assert "pending_calls" not in snapshots[0].metadata
    first.items.append("later mutation")
    assert snapshots[0].domain_state["llm"] == {"items": [1]}
    monkeypatch.setattr(FileCheckpointStore, "save", original_save)
    resumed = DefaultState(max_steps=3)
    resumed_backend = make_backend()
    steps = await run(resumed_backend, resumed)
    assert len(steps) == 1
    assert len(calls) == 2
    assert effects == [candidate]
    assert calls[-1].count("Retained hypothesis") == 1
    assert calls[-1].count("Check the evidence") == 1
    assert RunResult.from_state(resumed).output == {"answer": candidate}
    terminal = FileCheckpointStore(tmp_path).load_latest_for_resume()
    assert terminal.is_terminal
    resumed_backend.items.append("after completion")
    assert terminal.domain_state["llm"] == {"items": [1, 2]}
    terminal_state = DefaultState(max_steps=3)
    assert await run(make_backend(), terminal_state) == []
    assert len(calls) == 2
    assert effects == [candidate]
    assert RunResult.from_state(terminal_state).completed
    assert RunResult.from_state(terminal_state).output == {"answer": candidate}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("boundary", ["step_1_pending", "step_1"])
async def test_tool_checkpoints_copy_backend_and_domain_state(
    tmp_path, monkeypatch, asynchronous, boundary
):
    from looplet import AssistantTurnDecision, RunResult, tools_from

    class Crash(BaseException):
        pass

    requests = []
    effects = []
    host_state = {"observations": ["retained"]}
    snapshots = []

    class Backend:
        def __init__(self):
            self.items = []

        def checkpoint_state(self):
            return {"items": self.items}

        def restore_checkpoint_state(self, snapshot):
            self.items = snapshot["items"]

        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            self.items.append(len(self.items) + 1)
            return (
                '{"tool":"lookup","args":{}}'
                if len(self.items) == 1
                else '{"tool":"done","args":{"summary":"accepted"}}'
            )

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    def lookup():
        effects.append("lookup")
        return "X" * 6000

    config = LoopConfig(
        max_steps=3,
        max_model_calls=3,
        use_native_tools=False,
        checkpoint_dir=tmp_path,
        checkpoint_state=lambda context: host_state,
        assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
    )
    original_save = FileCheckpointStore.save

    def interrupt_after_commit(store, checkpoint, key):
        original_save(store, checkpoint, key)
        if key == boundary:
            snapshots.append(checkpoint)
            raise Crash()

    monkeypatch.setattr(FileCheckpointStore, "save", interrupt_after_commit)

    async def run(backend, state):
        arguments = dict(
            llm=backend,
            state=state,
            tools=tools_from([lookup], include_done=True),
            config=config,
        )
        return (
            [step async for step in async_composable_loop(**arguments)]
            if asynchronous
            else list(composable_loop(**arguments))
        )

    first = AsyncBackend() if asynchronous else Backend()
    with pytest.raises(Crash):
        await run(first, DefaultState(max_steps=3))
    assert len(requests) == 1
    assert effects == ([] if boundary.endswith("pending") else ["lookup"])
    first.items.append("later mutation")
    host_state["observations"].append("later mutation")
    assert snapshots[0].domain_state == {
        "observations": ["retained"],
        "llm": {"items": [1]},
    }
    monkeypatch.setattr(FileCheckpointStore, "save", original_save)
    if not boundary.endswith("pending"):
        resumed = DefaultState(max_steps=3)
        steps = await run(AsyncBackend() if asynchronous else Backend(), resumed)
        assert [step.tool_call.tool for step in steps] == ["done"]
        assert RunResult.from_state(resumed).completed
        assert len(requests) == 2
        assert effects == ["lookup"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_started_request_crash_is_not_implicitly_retried(tmp_path, asynchronous):
    from looplet import tools_from
    from looplet.assistant_turns import TURN_STATE_KEY, AssistantTurnDecision
    from looplet.checkpoint import UncertainModelCall

    class Crash(BaseException):
        pass

    calls = []

    class Backend:
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            raise Crash()

    backend = Backend()
    if asynchronous:
        original = backend.generate

        async def generate(prompt, **kwargs):
            return original(prompt, **kwargs)

        backend.generate = generate
    config = LoopConfig(
        max_steps=2,
        max_model_calls=2,
        use_native_tools=False,
        checkpoint_dir=tmp_path,
        assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
    )

    async def run():
        kwargs = dict(llm=backend, tools=tools_from([], include_done=True), config=config)
        return (
            [step async for step in async_composable_loop(**kwargs)]
            if asynchronous
            else list(composable_loop(**kwargs))
        )

    with pytest.raises(Crash):
        await run()
    checkpoint = FileCheckpointStore(tmp_path).load_latest_for_resume()
    assert checkpoint.metadata[TURN_STATE_KEY]["model_calls"] == 1
    assert checkpoint.metadata[TURN_STATE_KEY]["pending_model_call"]
    with pytest.raises(UncertainModelCall):
        await run()
    assert len(calls) == 1


def test_unacknowledged_model_request_blocks_resume():
    from looplet.assistant_turns import TURN_STATE_KEY
    from looplet.checkpoint import UncertainModelCall

    checkpoint = _make_checkpoint(step=0)
    checkpoint.metadata[TURN_STATE_KEY] = {"pending_model_call": {"model_call_number": 1}}
    with pytest.raises(UncertainModelCall, match="duplicate inference"):
        resume_loop_state(checkpoint)


class TestCheckpointDataclass:
    def test_fields_are_accessible(self) -> None:
        cp = _make_checkpoint()
        assert cp.step_number == 3
        assert isinstance(cp.session_log_data, dict)
        assert cp.conversation_data is None
        assert isinstance(cp.config_snapshot, dict)
        assert isinstance(cp.tool_results_store, dict)
        assert isinstance(cp.metadata, dict)
        assert isinstance(cp.created_at, float)

    def test_to_dict_returns_all_fields(self) -> None:
        cp = _make_checkpoint()
        d = cp.to_dict()
        assert d["step_number"] == 3
        assert "session_log_data" in d
        assert "conversation_data" in d
        assert "config_snapshot" in d
        assert "tool_results_store" in d
        assert "metadata" in d
        assert "created_at" in d

    def test_to_dict_is_defensive_and_json_safe(self) -> None:
        cp = _make_checkpoint()
        snapshot = cp.to_dict()
        snapshot["tool_results_store"]["step_1_search"]["rows"].append({"id": 2})
        assert cp.tool_results_store["step_1_search"]["rows"] == [{"id": 1}]

    @pytest.mark.parametrize("step_number", [None, "1", True])
    def test_from_dict_rejects_invalid_step_number(self, step_number) -> None:
        with pytest.raises(ValueError, match="step_number"):
            Checkpoint.from_dict({"step_number": step_number})

    def test_from_dict_round_trip(self) -> None:
        cp = _make_checkpoint()
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.step_number == cp.step_number
        assert restored.session_log_data == cp.session_log_data
        assert restored.conversation_data == cp.conversation_data
        assert restored.config_snapshot == cp.config_snapshot
        assert restored.tool_results_store == cp.tool_results_store
        assert restored.metadata == cp.metadata
        assert abs(restored.created_at - cp.created_at) < 0.001

    def test_from_dict_with_conversation_data(self) -> None:
        cp = _make_checkpoint()
        d = cp.to_dict()
        d["conversation_data"] = {"messages": [{"role": "user", "content": "hello"}]}
        restored = Checkpoint.from_dict(d)
        assert restored.conversation_data is not None
        assert len(restored.conversation_data["messages"]) == 1

    def test_to_dict_is_json_serializable(self) -> None:
        cp = _make_checkpoint()
        cp.domain_state = {"cursor": 4, "seen": ["alpha"]}
        raw = json.dumps(cp.to_dict())
        assert isinstance(raw, str)
        loaded = json.loads(raw)
        assert loaded["step_number"] == 3
        assert loaded["domain_state"] == {"cursor": 4, "seen": ["alpha"]}

    def test_from_dict_missing_optional_fields(self) -> None:
        minimal = {
            "step_number": 1,
            "session_log_data": {"entries": [], "current_theory": ""},
            "config_snapshot": {"max_steps": 10},
        }
        cp = Checkpoint.from_dict(minimal)
        assert cp.step_number == 1
        assert cp.conversation_data is None
        assert cp.tool_results_store == {}
        assert cp.domain_state == {}
        assert cp.metadata == {}

    def test_created_at_defaults_to_now(self) -> None:
        before = time.time()
        cp = Checkpoint(
            step_number=1,
            session_log_data={},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={},
        )
        after = time.time()
        assert before <= cp.created_at <= after


# ── CheckpointStore Protocol ────────────────────────────────────────


class TestCheckpointStoreProtocol:
    def test_file_store_is_instance_of_protocol(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        assert isinstance(store, CheckpointStore)

    def test_protocol_methods_exist(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        assert hasattr(store, "save")
        assert hasattr(store, "load")
        assert callable(store.save)
        assert callable(store.load)


# ── FileCheckpointStore ─────────────────────────────────────────────


class TestFileCheckpointStore:
    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp = _make_checkpoint()
        store.save(cp, "run-001")
        assert (tmp_path / "run-001.json").exists()

    def test_saved_file_is_valid_json(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp = _make_checkpoint()
        store.save(cp, "run-002")
        raw = (tmp_path / "run-002.json").read_text()
        data = json.loads(raw)
        assert data["step_number"] == 3

    def test_load_returns_checkpoint(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp = _make_checkpoint(step=7)
        store.save(cp, "run-003")
        loaded = store.load("run-003")
        assert loaded is not None
        assert loaded.step_number == 7

    def test_load_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        result = store.load("nonexistent-key")
        assert result is None

    def test_save_load_round_trip_preserves_all_data(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp = _make_checkpoint()
        store.save(cp, "full-round-trip")
        loaded = store.load("full-round-trip")
        assert loaded is not None
        assert loaded.session_log_data == cp.session_log_data
        assert loaded.config_snapshot == cp.config_snapshot
        assert loaded.tool_results_store == cp.tool_results_store
        assert loaded.metadata["task_id"] == "task-1"

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp1 = _make_checkpoint(step=1)
        cp2 = _make_checkpoint(step=9)
        store.save(cp1, "overwrite-me")
        store.save(cp2, "overwrite-me")
        loaded = store.load("overwrite-me")
        assert loaded is not None
        assert loaded.step_number == 9

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        subdir = tmp_path / "deep" / "nested"
        store = FileCheckpointStore(subdir)
        cp = _make_checkpoint()
        store.save(cp, "nested-key")
        assert (subdir / "nested-key.json").exists()

    def test_multiple_keys_isolated(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        store.save(_make_checkpoint(step=1), "key-a")
        store.save(_make_checkpoint(step=2), "key-b")
        a = store.load("key-a")
        b = store.load("key-b")
        assert a is not None and a.step_number == 1
        assert b is not None and b.step_number == 2

    def test_load_rejects_conflicting_run_identity(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        checkpoint = _make_checkpoint()
        checkpoint.run_envelope = {"run_id": "run-a"}
        (tmp_path / "step_1.json").write_text(json.dumps(checkpoint.to_dict()))

        with pytest.raises(ValueError, match="belongs to run 'run-a'"):
            store.load("step_1", run_id="run-b")


# ── CheckpointHook ─────────────────────────────────────────────────


class TestCheckpointHook:
    def _make_store(self) -> MagicMock:
        store = MagicMock(spec=["save", "load"])
        return store

    def _make_tool_call_result(self) -> tuple[Any, Any]:
        from looplet.types import ToolCall, ToolResult

        tc = ToolCall(tool="search", args={}, reasoning="r")
        tr = ToolResult(tool="search", args_summary="", data={}, error=None)
        return tc, tr

    def test_hook_saves_at_interval(self) -> None:
        store = self._make_store()
        saved: list[int] = []

        def get_data(step_num: int) -> Checkpoint:
            saved.append(step_num)
            return _make_checkpoint(step=step_num)

        hook = CheckpointHook(store=store, save_every_n_steps=5, get_checkpoint_data=get_data)
        state = MagicMock()
        session_log = MagicMock()
        tc, tr = self._make_tool_call_result()
        for n in range(1, 11):
            hook.post_dispatch(state, session_log, tc, tr, n)

        assert store.save.call_count == 2  # steps 5 and 10
        assert saved == [5, 10]

    def test_hook_saves_at_correct_steps(self) -> None:
        store = self._make_store()
        captured_steps: list[int] = []

        def get_data(step_num: int) -> Checkpoint:
            captured_steps.append(step_num)
            return _make_checkpoint(step=step_num)

        hook = CheckpointHook(store=store, save_every_n_steps=3, get_checkpoint_data=get_data)
        state = MagicMock()
        session_log = MagicMock()
        tc, tr = self._make_tool_call_result()
        for n in range(1, 10):
            hook.post_dispatch(state, session_log, tc, tr, n)

        assert captured_steps == [3, 6, 9]

    def test_hook_does_not_save_at_non_interval_steps(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            save_every_n_steps=10,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        state = MagicMock()
        session_log = MagicMock()
        tc, tr = self._make_tool_call_result()
        for n in range(1, 9):
            hook.post_dispatch(state, session_log, tc, tr, n)
        assert store.save.call_count == 0

    def test_hook_uses_step_number_as_key(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            save_every_n_steps=5,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        state = MagicMock()
        session_log = MagicMock()
        tc, tr = self._make_tool_call_result()
        for n in range(1, 6):
            hook.post_dispatch(state, session_log, tc, tr, n)
        call_args = store.save.call_args
        assert call_args is not None
        _, key = call_args[0]
        assert "5" in key

    def test_real_loop_state_defers_save_until_post_step(self) -> None:
        store = self._make_store()
        captured_lengths: list[int] = []

        def get_data(step_num: int) -> Checkpoint:
            captured_lengths.append(len(state.steps))
            return _make_checkpoint(step=step_num)

        state = MagicMock()
        state.steps = []
        session_log = MagicMock()
        tc, tr = self._make_tool_call_result()
        hook = CheckpointHook(store=store, get_checkpoint_data=get_data, save_every_n_steps=1)

        hook.post_dispatch(state, session_log, tc, tr, 1)
        assert captured_lengths == []
        state.steps.append(object())
        hook.post_step(state, session_log, 1)

        assert captured_lengths == [1]

    def test_loop_checkpoint_hook_sees_recorded_step_and_terminal_lifecycle(self, tmp_path):
        from looplet import (
            BaseToolRegistry,
            DefaultState,
            LoopConfig,
            MockLLMBackend,
            register_done_tool,
        )
        from looplet.loop import composable_loop

        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=1)
        session_log = SessionLog()
        snapshots = []

        def get_data(step_num: int) -> Checkpoint:
            snapshots.append(
                (
                    len(state.steps),
                    len(session_log.entries),
                    getattr(state, "run_status", None),
                    getattr(state, "run_phase", None),
                )
            )
            return _make_checkpoint(step=step_num)

        hook = CheckpointHook(
            store=FileCheckpointStore(tmp_path),
            get_checkpoint_data=get_data,
            save_every_n_steps=1,
        )
        list(
            composable_loop(
                MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
                tools=tools,
                state=state,
                session_log=session_log,
                config=LoopConfig(max_steps=1),
                hooks=[hook],
            )
        )

        assert snapshots == [(1, 1, "completed", "terminal")]

    def test_hook_default_save_every(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        assert hook.save_every_n_steps == 5

    @pytest.mark.parametrize("interval", [0, -1])
    def test_hook_rejects_invalid_save_interval(self, interval: int) -> None:
        with pytest.raises(ValueError, match="save_every_n_steps"):
            CheckpointHook(
                store=self._make_store(),
                get_checkpoint_data=lambda n: _make_checkpoint(step=n),
                save_every_n_steps=interval,
            )

    def test_noop_hook_methods_dont_raise(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        state = MagicMock()
        session_log = MagicMock()
        context = MagicMock()
        tc, tr = self._make_tool_call_result()
        hook.pre_prompt(state, session_log, context, 1)
        hook.pre_dispatch(state, session_log, tc, 1)
        hook.check_done(state, session_log, context, 1)
        hook.should_stop(state, 1, 0)
        hook.on_loop_end(state, session_log, context, MagicMock())

    def test_pre_dispatch_returns_none(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        tc, _ = self._make_tool_call_result()
        result = hook.pre_dispatch(MagicMock(), MagicMock(), tc, 1)
        assert result is None

    def test_check_done_returns_none(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        assert hook.check_done(MagicMock(), MagicMock(), MagicMock(), 1) is None

    def test_should_stop_returns_false(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        assert hook.should_stop(MagicMock(), 1, 0) is False


# ── resume_loop_state ───────────────────────────────────────────────


class TestResumeLoopState:
    def _make_checkpoint_with_log(self) -> tuple[Checkpoint, SessionLog]:
        log = SessionLog()
        log.record(1, "theory A", "search", "initial search", entities=["host-x"])
        log.record(2, "theory B", "query", "deeper look", findings=["42 events"])
        log.record(3, "theory B", "done", "finished", highlights=["critical-finding"])
        cp = Checkpoint(
            step_number=3,
            session_log_data={
                "entries": log.to_list(),
                "current_theory": log.current_theory,
            },
            conversation_data=None,
            config_snapshot={"max_steps": 15},
            tool_results_store={},
            metadata={"task_id": "test-task"},
        )
        return cp, log

    def test_returns_dict_with_required_keys(self) -> None:
        cp, _ = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        assert "session_log" in result
        assert "step_offset" in result
        assert "metadata" in result

    def test_session_log_is_reconstructed(self) -> None:
        cp, original = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        log = result["session_log"]
        assert isinstance(log, SessionLog)

    def test_session_log_entries_match_original(self) -> None:
        cp, original = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        log: SessionLog = result["session_log"]
        assert len(log.entries) == len(original.entries)

    def test_session_log_current_theory_restored(self) -> None:
        cp, original = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        log: SessionLog = result["session_log"]
        assert log.current_theory == original.current_theory

    def test_step_offset_matches_checkpoint_step(self) -> None:
        cp, _ = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        assert result["step_offset"] == cp.step_number

    def test_metadata_passed_through(self) -> None:
        cp, _ = self._make_checkpoint_with_log()
        cp.domain_state = {"cursor": 9}
        result = resume_loop_state(cp)
        assert result["metadata"]["task_id"] == "test-task"
        assert result["domain_state"] == {"cursor": 9}

    def test_session_log_entries_tool_names_preserved(self) -> None:
        cp, original = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        log: SessionLog = result["session_log"]
        tools = [e.tool for e in log.entries]
        assert "search" in tools
        assert "done" in tools

    def test_session_log_entities_preserved(self) -> None:
        cp, _ = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        log: SessionLog = result["session_log"]
        all_ents = log.all_entities()
        assert "host-x" in all_ents

    def test_session_log_findings_preserved(self) -> None:
        cp, _ = self._make_checkpoint_with_log()
        result = resume_loop_state(cp)
        log: SessionLog = result["session_log"]
        all_findings = [f for e in log.entries for f in e.findings]
        assert "42 events" in all_findings

    def test_empty_session_log_round_trip(self) -> None:
        cp = Checkpoint(
            step_number=0,
            session_log_data={"entries": [], "current_theory": ""},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={},
        )
        result = resume_loop_state(cp)
        log: SessionLog = result["session_log"]
        assert len(log.entries) == 0
        assert result["step_offset"] == 0


def test_interrupted_dispatch_blocks_implicit_retry(tmp_path: Path) -> None:
    effects: list[str] = []
    tools = BaseToolRegistry()

    def interrupt() -> None:
        effects.append("started")
        raise KeyboardInterrupt

    tools.register(ToolSpec("effect", "Side effect", {}, interrupt))
    config = LoopConfig(max_steps=2, use_native_tools=False, checkpoint_dir=str(tmp_path))
    response = '{"tool":"effect","args":{},"call_id":"effect-1"}'
    with pytest.raises(KeyboardInterrupt):
        list(
            composable_loop(
                MockLLMBackend([response]),
                tools=tools,
                config=config,
                state=DefaultState(max_steps=2),
            )
        )

    pending = FileCheckpointStore(tmp_path).load_latest_for_resume()
    assert pending is not None
    assert pending.metadata["pending_calls"][0]["execution_status"] == "effect_unknown"
    retry = MockLLMBackend([response])
    with pytest.raises(RuntimeError, match="Tool effects unknown"):
        list(composable_loop(retry, tools=tools, config=config, state=DefaultState(max_steps=2)))
    assert effects == ["started"]
    assert retry.calls == 0


def test_pending_dispatch_stays_scoped_to_its_run(tmp_path: Path) -> None:
    tools = BaseToolRegistry()

    def interrupt() -> None:
        raise KeyboardInterrupt

    tools.register(ToolSpec("effect", "Side effect", {}, interrupt))
    config = LoopConfig(
        max_steps=2,
        use_native_tools=False,
        checkpoint_dir=str(tmp_path),
        run_envelope=RunEnvelope(run_id="run-a"),
    )
    with pytest.raises(KeyboardInterrupt):
        list(
            composable_loop(
                MockLLMBackend(['{"tool":"effect","args":{}}']),
                tools=tools,
                config=config,
                state=DefaultState(max_steps=2),
            )
        )

    store = FileCheckpointStore(tmp_path)
    pending = store.load_latest_for_resume(run_id="run-a")
    assert pending is not None
    assert pending.run_id == "run-a"
    assert store.load_latest_for_resume(run_id="run-b") is None


@pytest.mark.parametrize("async_driver", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize(
    ("remaining", "previous_max_steps"), [(0, 4), (1, 4), (2, 4), (0, 2), (1, 2), (1, 6)]
)
@pytest.mark.parametrize("free_calls", [False, True])
def test_resume_preserves_default_state_budget(
    tmp_path: Path,
    async_driver: bool,
    native: bool,
    identified: bool,
    remaining: int,
    previous_max_steps: int,
    free_calls: bool,
) -> None:
    envelope = RunEnvelope(run_id="budget-run") if identified else None
    store = FileCheckpointStore(tmp_path)
    store.save(
        Checkpoint(
            step_number=previous_max_steps - remaining,
            session_log_data={"entries": []},
            conversation_data=None,
            config_snapshot={
                "max_steps": previous_max_steps,
                "queries_used": previous_max_steps - remaining,
                "budget_remaining": remaining,
            },
            tool_results_store={},
            metadata={},
            run_envelope=envelope.to_dict() if envelope is not None else None,
        ),
        "before-restart",
    )
    effects = []
    observed_budgets = []
    tools = BaseToolRegistry()
    tools.register(ToolSpec("work", "Record synthetic work", {}, lambda: effects.append("work")))
    tools.register(ToolSpec("note", "Record a free note", {}, lambda: "noted", free=True))
    register_done_tool(tools)

    class ObserveBudget:
        def pre_prompt(self, state, session_log, context, step_num):
            observed_budgets.append(state.budget_remaining)
            return None

    backend_class = AsyncMockLLMBackend if async_driver else MockLLMBackend
    free_responses = ['{"tool":"note","args":{}}'] * 3 if free_calls else []
    backend = backend_class(
        free_responses
        + ['{"tool":"work","args":{}}'] * 3
        + ['{"tool":"done","args":{"summary":"Finished."}}']
    )
    state = DefaultState(max_steps=4)
    arguments = {
        "llm": backend,
        "tools": tools,
        "state": state,
        "config": LoopConfig(
            max_steps=4,
            checkpoint_dir=str(tmp_path),
            use_native_tools=native,
            run_envelope=envelope,
        ),
        "hooks": [ObserveBudget()],
    }

    async def run_async():
        return [step async for step in async_composable_loop(**arguments)]

    steps = asyncio.run(run_async()) if async_driver else list(composable_loop(**arguments))
    expected_remaining = max(0, remaining + 4 - previous_max_steps)
    expected_free = ["note"] * 3 if free_calls and expected_remaining else []
    assert observed_budgets == [expected_remaining] * len(expected_free) + list(
        range(expected_remaining, 0, -1)
    )
    assert effects == ["work"] * expected_remaining
    assert [step.tool_call.tool for step in steps] == expected_free + ["work"] * expected_remaining
    assert state.budget_remaining == 0


@pytest.mark.parametrize("remaining", [True, -1, 1.5, "1"])
def test_default_state_rejects_invalid_restored_budget(remaining) -> None:
    state = DefaultState(max_steps=4)
    with pytest.raises(ValueError, match="nonnegative integer"):
        state.budget_remaining = remaining
    assert state.budget_remaining == 4


def test_default_state_budget_restore_preserves_current_charges() -> None:
    state = DefaultState(max_steps=4, steps=[object()])
    state.budget_remaining = 2
    assert state.budget_remaining == 2
    state.steps.append(object())
    assert state.budget_remaining == 1
    state.budget_remaining = 1
    assert state.budget_remaining == 1
    state.budget_remaining = 9
    assert state.budget_remaining == 2


@pytest.mark.parametrize(("max_steps", "expected"), [(None, 1), (0, 0), (2, 0), (4, 1), (6, 3)])
def test_resume_state_rebases_budget_without_changing_checkpoint(max_steps, expected) -> None:
    checkpoint = _make_checkpoint()
    checkpoint.config_snapshot = {"max_steps": 4, "queries_used": 2, "budget_remaining": 1}
    original = checkpoint.config_snapshot.copy()
    resumed = resume_loop_state(checkpoint, max_steps=max_steps)
    assert resumed["state_counters"] == {"queries_used": 2, "budget_remaining": expected}
    assert checkpoint.config_snapshot == original


@pytest.mark.parametrize("max_steps", [None, 6])
def test_resume_legacy_budget_without_saved_cap(max_steps) -> None:
    checkpoint = _make_checkpoint()
    checkpoint.config_snapshot = {"queries_used": 2, "budget_remaining": 1}
    resumed = resume_loop_state(checkpoint, max_steps=max_steps)
    assert resumed["state_counters"] == {"queries_used": 2, "budget_remaining": 1}


@pytest.mark.parametrize("field", ["max_steps", "budget_remaining", "requested"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_resume_rejects_invalid_step_budgets(field, value) -> None:
    checkpoint = _make_checkpoint()
    checkpoint.config_snapshot = {"max_steps": 4, "budget_remaining": 1}
    if field != "requested":
        checkpoint.config_snapshot[field] = value
    with pytest.raises(ValueError, match="nonnegative integers"):
        resume_loop_state(checkpoint, max_steps=value if field == "requested" else 4)


def test_completed_dispatch_does_not_block_resume(tmp_path: Path) -> None:
    tools = BaseToolRegistry()
    tools.register(ToolSpec("effect", "Side effect", {}, lambda: "ok"))
    register_done_tool(tools)
    config = LoopConfig(max_steps=2, use_native_tools=False, checkpoint_dir=str(tmp_path))
    steps = list(
        composable_loop(
            MockLLMBackend(
                [
                    '{"tool":"effect","args":{}}',
                    '{"tool":"done","args":{"summary":"ok"}}',
                ]
            ),
            tools=tools,
            config=config,
            state=DefaultState(max_steps=2),
        )
    )
    assert [step.tool_call.tool for step in steps] == ["effect", "done"]
    assert FileCheckpointStore(tmp_path).load_latest_for_resume() is None


@pytest.mark.parametrize("async_driver", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("damage", ["json", "schema", "unreadable"])
def test_corrupt_pending_checkpoint_blocks_resume(
    tmp_path: Path, monkeypatch, async_driver: bool, native: bool, identified: bool, damage: str
) -> None:
    envelope = RunEnvelope(run_id="effect-run") if identified else None
    store = FileCheckpointStore(tmp_path)
    previous = _make_checkpoint(step=1)
    previous.run_envelope = envelope.to_dict() if envelope is not None else None
    store.save(previous, "step_1")
    pending = _make_checkpoint(step=2)
    pending.run_envelope = previous.run_envelope
    pending.metadata["pending_calls"] = [
        {
            "call_id": "effect-1",
            "tool": "effect",
            "proposed_args": {},
            "effective_args": {},
            "execution_status": "effect_unknown",
        }
    ]
    key = FileCheckpointStore.key_for(pending, "step_2_pending")
    store.save(pending, "step_2_pending")
    path = tmp_path / f"{key}.json"
    if damage == "unreadable":
        read_text = Path.read_text

        def unreadable(candidate, *arguments, **keywords):
            if candidate == path:
                raise PermissionError("Synthetic checkpoint read failure")
            return read_text(candidate, *arguments, **keywords)

        monkeypatch.setattr(Path, "read_text", unreadable)
    else:
        path.write_text(
            '{"step_number":' if damage == "json" else json.dumps({"step_number": "bad"})
        )
    damaged_bytes = path.read_bytes()
    effects = ["original-effect"]
    tools = BaseToolRegistry()
    tools.register(
        ToolSpec("effect", "Record a synthetic effect", {}, lambda: effects.append("repeated"))
    )
    register_done_tool(tools)
    backend_class = AsyncMockLLMBackend if async_driver else MockLLMBackend
    backend = backend_class(
        ['{"tool":"effect","args":{}}', '{"tool":"done","args":{"summary":"Finished."}}']
    )
    arguments = {
        "llm": backend,
        "tools": tools,
        "state": DefaultState(max_steps=4),
        "config": LoopConfig(
            max_steps=4,
            checkpoint_dir=str(tmp_path),
            use_native_tools=native,
            run_envelope=envelope,
        ),
    }

    async def run_async():
        return [step async for step in async_composable_loop(**arguments)]

    with pytest.raises(ValueError, match="Cannot safely resume") as error:
        if async_driver:
            asyncio.run(run_async())
        else:
            list(composable_loop(**arguments))
    assert str(path) in str(error.value)
    assert backend.calls == 0
    assert effects == ["original-effect"]
    assert path.read_bytes() == damaged_bytes


@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_corrupt_only_checkpoint_never_starts_fresh(
    tmp_path: Path, identified: bool, terminal: bool
) -> None:
    checkpoint = _make_checkpoint()
    checkpoint.run_envelope = RunEnvelope(run_id="broken-run").to_dict() if identified else None
    if terminal:
        checkpoint.run_status = "completed"
        checkpoint.run_phase = "terminal"
    store = FileCheckpointStore(tmp_path)
    store.save(checkpoint, "step_3")
    path = tmp_path / f"{FileCheckpointStore.key_for(checkpoint, 'step_3')}.json"
    path.write_text('{"step_number":')
    with pytest.raises(ValueError, match="Cannot safely resume"):
        store.load_latest_for_resume(run_id=checkpoint.run_id)


@pytest.mark.parametrize("identified", [False, True])
def test_corrupt_other_run_does_not_block_resume(tmp_path: Path, identified: bool) -> None:
    checkpoint = _make_checkpoint()
    checkpoint.run_envelope = RunEnvelope(run_id="selected-run").to_dict() if identified else None
    store = FileCheckpointStore(tmp_path)
    store.save(checkpoint, "step_3")
    other = _make_checkpoint(step=9)
    other.run_envelope = RunEnvelope(run_id="other-run").to_dict()
    store.save(other, "step_9_pending")
    path = tmp_path / f"{FileCheckpointStore.key_for(other, 'step_9_pending')}.json"
    path.write_text('{"step_number":')
    selected = store.load_latest_for_resume(run_id=checkpoint.run_id)
    assert selected is not None and selected.run_id == checkpoint.run_id
    assert selected.step_number == checkpoint.step_number
