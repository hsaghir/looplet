from __future__ import annotations

import errno
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import pytest

from looplet import RunEnvelope, RunPhase, RunResult, RunStatus, Step, ToolCall, ToolResult
from looplet.checkpoint import Checkpoint
from looplet.run_records import RunEvent
from looplet.run_store import FileRunStore, MemoryRunStore, _InterProcessFileLock


def _save_checkpoint_from_process(directory: str, step: int) -> None:
    store = FileRunStore(directory)
    store.save_checkpoint(
        "process-race",
        Checkpoint(
            step_number=step,
            session_log_data={"entries": []},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={"step": step},
        ),
    )


def _result() -> RunResult:
    return RunResult(
        status=RunStatus.COMPLETED,
        phase=RunPhase.TERMINAL,
        termination_reason="done",
        output={"summary": "ok"},
        steps=(),
        run_envelope=RunEnvelope(run_id="run-1"),
    )


def _exercise(store) -> None:
    envelope = RunEnvelope(run_id="run-1")
    store.create(envelope, metadata={"source": "test"})
    store.append_event(
        RunEvent(envelope=envelope, sequence=1, timestamp=1.0, kind="start", payload={})
    )
    key = store.save_checkpoint(
        "run-1",
        Checkpoint(
            step_number=1,
            session_log_data={"entries": []},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={},
        ),
    )
    assert key == "step_1"
    record = store.complete("run-1", _result())
    assert record.status == "completed"
    loaded = store.load("run-1")
    assert loaded is not None
    assert loaded.metadata["source"] == "test"
    assert loaded.checkpoint_keys == ("step_1",)
    assert len(store.events("run-1")) == 1


def test_memory_run_store() -> None:
    _exercise(MemoryRunStore())


def test_file_run_store(tmp_path) -> None:
    _exercise(FileRunStore(tmp_path))


@pytest.mark.parametrize("store_kind", ["memory", "file"])
@pytest.mark.parametrize("data", [list(range(21)), "x" * 2001], ids=["list", "text"])
def test_run_store_retains_complete_tool_results(tmp_path, store_kind, data) -> None:
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path / "runs")
    envelope = RunEnvelope(run_id="run-1")
    store.create(envelope)
    tool_result = ToolResult(
        tool="inspect",
        args_summary="source=owned",
        data=data,
        duration_ms=12.3456,
        call_id="call-1",
    )
    result = RunResult(
        status=RunStatus.COMPLETED,
        phase=RunPhase.TERMINAL,
        termination_reason="done",
        output={"summary": "ok"},
        steps=(Step(1, ToolCall(tool="inspect", args={}, call_id="call-1"), tool_result),),
        run_envelope=envelope,
    )
    preview = result.to_dict()["steps"][0]["tool_result"]
    expected_preview = data[:20] if isinstance(data, list) else data[:2000]
    assert preview["data"] == expected_preview
    store.complete(envelope.run_id, result)
    loaded = store.load(envelope.run_id)
    assert loaded is not None and loaded.artifact_refs == ()
    retained = loaded.result["steps"][0]["tool_result"]
    assert retained["data"] == data
    assert retained["duration_ms"] == tool_result.duration_ms
    assert retained["call_id"] == tool_result.call_id
    assert result.to_dict()["steps"][0]["tool_result"] == preview


@pytest.mark.parametrize("retained", ["empty", "events", "checkpoints"])
def test_file_run_store_rejects_partial_run_directory(tmp_path, retained) -> None:
    store = FileRunStore(tmp_path / "runs")
    envelope = RunEnvelope(run_id="run-1")
    root = store.directory / envelope.run_id
    root.mkdir()
    if retained == "events":
        event = RunEvent(
            envelope=envelope, sequence=7, timestamp=1.0, kind="old-attempt", payload={}
        )
        (root / "events.jsonl").write_text(json.dumps(event.to_dict()) + "\n", encoding="utf-8")
    elif retained == "checkpoints":
        (root / "checkpoints").mkdir()
        (root / "checkpoints/step_1.json").write_text("{}\n", encoding="utf-8")
    before = {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }
    with pytest.raises(FileExistsError):
        store.create(envelope)
    assert {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    } == before


def test_file_run_store_reads_do_not_reserve_run_id(tmp_path) -> None:
    store = FileRunStore(tmp_path / "runs")
    envelope = RunEnvelope(run_id="run-1")
    assert store.load(envelope.run_id) is None
    assert store.events(envelope.run_id) == ()
    assert store.load_checkpoint(envelope.run_id, "step_1") is None
    assert not (store.directory / envelope.run_id).exists()
    assert store.create(envelope).status == "created"


@pytest.mark.parametrize("store_kind", ["memory", "file"])
def test_run_store_late_event_preserves_terminal_record(tmp_path, store_kind) -> None:
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path / "runs")
    envelope = RunEnvelope(run_id="run-1")
    assert store.create(envelope).status == "created"
    store.append_event(
        RunEvent(envelope=envelope, sequence=1, timestamp=1.0, kind="start", payload={})
    )
    running = store.load(envelope.run_id)
    assert running is not None and running.status == "running"
    terminal = store.complete(envelope.run_id, _result())
    late_event = RunEvent(
        envelope=envelope,
        sequence=2,
        timestamp=terminal.updated_at + 1,
        kind="report",
        payload={"message": "late evidence"},
    )
    store.append_event(late_event)
    loaded = store.load(envelope.run_id)
    assert loaded is not None and loaded.status == terminal.status
    assert loaded.result == terminal.result
    assert loaded.events[-1] == late_event


def test_file_run_store_rejects_escaped_lock_link(tmp_path) -> None:
    outside = tmp_path / "outside.lock"
    outside.touch()
    store = FileRunStore(tmp_path / "runs")
    (store.directory / ".run-store.lock").symlink_to(outside)
    with pytest.raises(ValueError):
        store.create(RunEnvelope(run_id="run-1"))
    assert outside.read_bytes() == b""
    assert not (store.directory / "run-1").exists()


def test_file_run_store_preserves_contained_lock_link(tmp_path) -> None:
    store = FileRunStore(tmp_path / "runs")
    contained = store.directory / "shared.lock"
    contained.touch()
    (store.directory / ".run-store.lock").symlink_to(contained)
    record = store.create(RunEnvelope(run_id="run-1"))
    assert store.load(record.run_id) == record


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_file_run_store_failed_lock_entry_closes_file(tmp_path, monkeypatch, failure_type) -> None:
    fcntl = pytest.importorskip("fcntl")
    descriptors = []
    lock = _InterProcessFileLock(tmp_path / "lock")
    failure = failure_type("lock acquisition refused")

    def deny_lock(descriptor, operation):
        descriptors.append(descriptor)
        assert operation == fcntl.LOCK_EX
        raise failure

    monkeypatch.setattr(fcntl, "flock", deny_lock)
    with pytest.raises(failure_type, match="lock acquisition refused") as raised:
        with lock:
            pytest.fail("failed lock entry reached the transaction")
    assert raised.value is failure
    assert len(descriptors) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    assert lock._handle is None


@pytest.mark.parametrize("run_id", ["..", "run-link"])
def test_file_run_store_rejects_escaped_run_roots(tmp_path, run_id) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store = FileRunStore(tmp_path / "runs")
    (store.directory / "run-link").symlink_to(outside, target_is_directory=True)
    target = tmp_path if run_id == ".." else outside
    names = sorted(path.name for path in target.iterdir())

    with pytest.raises(ValueError, match="run_id"):
        store.create(RunEnvelope(run_id=run_id))

    assert sorted(path.name for path in target.iterdir()) == names


@pytest.mark.parametrize("surface", ["events", "checkpoints", "record"])
def test_file_run_store_rejects_escaped_component_links(tmp_path, surface) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    store = FileRunStore(tmp_path / "runs")
    envelope = RunEnvelope(run_id="run-1")
    store.create(envelope)
    root = store.directory / envelope.run_id
    if surface == "events":
        (root / "events.jsonl").unlink()
        (root / "events.jsonl").symlink_to(sentinel)
    elif surface == "checkpoints":
        (root / "checkpoints").rmdir()
        (root / "checkpoints").symlink_to(outside, target_is_directory=True)
    else:
        contents = (root / "run.json").read_bytes()
        sentinel.write_bytes(contents)
        (root / "run.json").unlink()
        (root / "run.json").symlink_to(sentinel)
    before = {path.name: path.read_bytes() for path in outside.iterdir()}

    with pytest.raises(ValueError, match="outside"):
        if surface == "events":
            store.append_event(
                RunEvent(envelope=envelope, sequence=0, timestamp=1.0, kind="probe", payload={})
            )
        elif surface == "checkpoints":
            store.save_checkpoint(
                envelope.run_id,
                Checkpoint(
                    step_number=1,
                    session_log_data={"entries": []},
                    conversation_data=None,
                    config_snapshot={},
                    tool_results_store={},
                    metadata={},
                ),
            )
        else:
            store.load(envelope.run_id)
    assert {path.name: path.read_bytes() for path in outside.iterdir()} == before


def test_file_run_store_preserves_contained_event_links(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    envelope = RunEnvelope(run_id="run-1")
    store.create(envelope)
    root = store.directory / envelope.run_id
    (root / "events.jsonl").rename(root / "retained-events.jsonl")
    (root / "events.jsonl").symlink_to("retained-events.jsonl")
    event = RunEvent(envelope=envelope, sequence=0, timestamp=1.0, kind="probe", payload={})
    store.append_event(event)
    assert store.events(envelope.run_id) == (event,)
    assert (root / "events.jsonl").is_symlink()


def test_file_run_store_writes_do_not_rescan_event_history(tmp_path, monkeypatch) -> None:
    store = FileRunStore(tmp_path)
    envelope = RunEnvelope(run_id="run-1")
    store.create(envelope)
    original_events = store._events_unlocked
    history_reads = []

    def read_events(run_id):
        history_reads.append(run_id)
        return original_events(run_id)

    monkeypatch.setattr(store, "_events_unlocked", read_events)
    for sequence in range(32):
        store.append_event(
            RunEvent(
                envelope=envelope,
                sequence=sequence,
                timestamp=float(sequence),
                kind="probe",
                payload={},
            )
        )
    store.save_checkpoint(
        envelope.run_id,
        Checkpoint(
            step_number=1,
            session_log_data={"entries": []},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={},
        ),
    )
    assert history_reads == []

    completed = store.complete(envelope.run_id, _result())
    assert history_reads == [envelope.run_id]
    assert tuple(event.sequence for event in completed.events) == tuple(range(32))
    loaded = store.load(envelope.run_id)
    assert loaded is not None
    assert loaded.events == completed.events
    assert history_reads == [envelope.run_id, envelope.run_id]


@pytest.mark.parametrize("store_kind", ["memory", "file"])
@pytest.mark.parametrize(
    "surface",
    [
        "create",
        "event_input",
        "events",
        "load",
        "checkpoint_input",
        "checkpoint_output",
        "complete",
    ],
)
def test_run_store_owns_mutable_evidence_snapshots(tmp_path, store_kind, surface) -> None:
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path)
    envelope = RunEnvelope(run_id="run-1")
    created = store.create(envelope, metadata={"nested": {"value": "original"}})
    event = RunEvent(
        envelope=envelope,
        sequence=0,
        timestamp=1.0,
        kind="probe",
        payload={"nested": {"value": "original"}},
    )
    store.append_event(event)
    checkpoint = Checkpoint(
        step_number=1,
        session_log_data={"entries": []},
        conversation_data=None,
        config_snapshot={},
        tool_results_store={},
        metadata={"nested": {"value": "original"}},
    )
    key = store.save_checkpoint(envelope.run_id, checkpoint)

    if surface == "create":
        created.metadata["nested"]["value"] = "changed"
    elif surface == "event_input":
        event.payload["nested"]["value"] = "changed"
    elif surface == "events":
        store.events(envelope.run_id)[0].payload["nested"]["value"] = "changed"
    elif surface == "load":
        loaded = store.load(envelope.run_id)
        assert loaded is not None
        loaded.metadata["nested"]["value"] = "changed"
    elif surface == "checkpoint_input":
        checkpoint.metadata["nested"]["value"] = "changed"
    elif surface == "checkpoint_output":
        loaded_checkpoint = store.load_checkpoint(envelope.run_id, key)
        assert loaded_checkpoint is not None
        loaded_checkpoint.metadata["nested"]["value"] = "changed"
    else:
        completed = store.complete(envelope.run_id, _result())
        completed.metadata["nested"]["value"] = "changed"

    retained = store.load(envelope.run_id)
    assert retained is not None
    assert retained.metadata["nested"]["value"] == "original"
    assert store.events(envelope.run_id)[0].payload["nested"]["value"] == "original"
    retained_checkpoint = store.load_checkpoint(envelope.run_id, key)
    assert retained_checkpoint is not None
    assert retained_checkpoint.metadata["nested"]["value"] == "original"


@pytest.mark.parametrize("store_kind", ["memory", "file"])
def test_run_store_records_completion_time(tmp_path, monkeypatch, store_kind) -> None:
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path)
    monkeypatch.setattr("looplet.run_store.time.time", lambda: 100.0)
    created = store.create(RunEnvelope(run_id="run-1"))
    monkeypatch.setattr("looplet.run_store.time.time", lambda: 200.0)
    completed = store.complete("run-1", _result())
    assert completed.created_at == created.created_at == 100.0
    assert completed.updated_at == 200.0


def test_file_run_store_serializes_concurrent_checkpoint_updates(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    store.create(RunEnvelope(run_id="race"))

    def save(step: int) -> None:
        store.save_checkpoint(
            "race",
            Checkpoint(
                step_number=step,
                session_log_data={"entries": []},
                conversation_data=None,
                config_snapshot={},
                tool_results_store={},
                metadata={},
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, [1, 2]))

    record = store.load("race")
    assert record is not None
    assert record.checkpoint_keys == ("step_1", "step_2")
    assert store.load_checkpoint("race", "step_1") is not None


def test_file_run_store_serializes_concurrent_process_updates(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    store.create(RunEnvelope(run_id="process-race"))

    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        list(pool.map(_save_checkpoint_from_process, [str(tmp_path)] * 4, [1, 2, 3, 4]))

    record = store.load("process-race")
    assert record is not None
    assert set(record.checkpoint_keys) == {"step_1", "step_2", "step_3", "step_4"}
    assert all(store.load_checkpoint("process-race", f"step_{step}") for step in range(1, 5))


@pytest.mark.parametrize("store_kind", ["memory", "file"])
def test_run_store_rejects_checkpoint_from_another_run(tmp_path, store_kind) -> None:
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path)
    store.create(RunEnvelope(run_id="run-a"))
    checkpoint = Checkpoint(
        step_number=1,
        session_log_data={"entries": []},
        conversation_data=None,
        config_snapshot={},
        tool_results_store={},
        metadata={},
        run_envelope={"run_id": "run-b"},
    )

    with pytest.raises(ValueError, match="belongs to run 'run-b'"):
        store.save_checkpoint("run-a", checkpoint)


@pytest.mark.parametrize("store_kind", ["memory", "file"])
def test_run_store_orders_out_of_order_checkpoint_keys_canonically(tmp_path, store_kind) -> None:
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path)
    store.create(RunEnvelope(run_id="out-of-order"))
    for step in (10, 2, 1, 2):
        store.save_checkpoint(
            "out-of-order",
            Checkpoint(
                step_number=step,
                session_log_data={"entries": [step]},
                conversation_data=None,
                config_snapshot={},
                tool_results_store={},
                metadata={},
            ),
        )
    record = store.load("out-of-order")
    assert record is not None
    assert record.checkpoint_keys == ("step_1", "step_2", "step_10")
    latest = store.load_checkpoint("out-of-order", record.checkpoint_keys[-1])
    assert latest is not None and latest.step_number == 10
