"""Tests for looplet.checkpoint — save/restore loop state."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

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
        raw = json.dumps(cp.to_dict())
        assert isinstance(raw, str)
        loaded = json.loads(raw)
        assert loaded["step_number"] == 3

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

    def test_hook_default_save_every(self) -> None:
        store = self._make_store()
        hook = CheckpointHook(
            store=store,
            get_checkpoint_data=lambda n: _make_checkpoint(step=n),
        )
        assert hook.save_every_n_steps == 5

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
        result = resume_loop_state(cp)
        assert result["metadata"]["task_id"] == "test-task"

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
