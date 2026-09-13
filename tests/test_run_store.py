from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from looplet import RunEnvelope, RunPhase, RunResult, RunStatus
from looplet.checkpoint import Checkpoint
from looplet.run_records import RunEvent
from looplet.run_store import FileRunStore, MemoryRunStore


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
