"""Compatibility contract for versioned saved provenance and eval artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from looplet import EvalContext, __version__, load_eval_run, save_eval_run
from looplet.artifact_compat import (
    ARTIFACT_DESCRIPTOR,
    ARTIFACT_PENDING,
    ARTIFACT_SCHEMA,
    ARTIFACT_VERSION,
    begin_artifact_write,
    read_artifact_descriptor,
)
from looplet.provenance import ProvenanceSink, replay_loop

_FIXTURES = Path(__file__).parent / "fixtures" / "artifact_compat"


def test_v1_eval_run_fixture_loads_and_ignores_unknown_fields() -> None:
    record = load_eval_run(_FIXTURES / "v1_eval_run")

    assert record.case is not None
    assert record.case.id == "fixture-v1"
    assert record.case.marks == ["regression"]
    assert record.context.completed is True
    assert record.context.final_output == {"summary": "fixture complete"}
    assert record.context.artifacts == {"tests_passing": True}
    assert [(result.name, result.score, result.label) for result in record.results] == [
        ("eval_tests_pass", 1.0, "pass")
    ]


def test_legacy_v0_eval_run_fixture_remains_readable() -> None:
    fixture = _FIXTURES / "legacy_v0_eval_run"

    assert read_artifact_descriptor(fixture) is None
    record = load_eval_run(fixture)

    assert record.case is None
    assert record.context.task == {
        "description": "legacy task",
        "expected_answer": 42,
    }
    assert record.context.final_output == {"answer": "legacy"}
    assert record.context.tool_sequence == ["done"]


def test_eval_run_writer_publishes_v1_descriptor_last(tmp_path: Path) -> None:
    output = save_eval_run(
        tmp_path / "run",
        context=EvalContext(
            steps=[],
            task={"goal": "write a descriptor"},
            artifacts={"observed": True},
            stop_reason="done",
        ),
    )

    descriptor = read_artifact_descriptor(output, expected_kinds=("eval_run",))
    assert descriptor == {
        "schema": ARTIFACT_SCHEMA,
        "version": ARTIFACT_VERSION,
        "kind": "eval_run",
        "producer": {"name": "looplet", "version": __version__},
        "components": ["artifacts", "eval_results", "trajectory"],
    }
    assert not list(output.glob(f".{ARTIFACT_DESCRIPTOR}.*.tmp"))
    assert not (output / ARTIFACT_PENDING).exists()


def test_eval_run_declares_model_calls_from_a_legacy_recorder(tmp_path: Path) -> None:
    class LegacyRecorder:
        def save(self, directory: str | Path) -> Path:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            (root / "trajectory.json").write_text('{"steps": []}')
            (root / "manifest.jsonl").write_text('{"index": 0, "method": "generate"}\n')
            (root / "call_00_prompt.txt").write_text("prompt")
            (root / "call_00_response.txt").write_text("response")
            return root

    output = save_eval_run(tmp_path / "run", recorder=LegacyRecorder())

    descriptor = read_artifact_descriptor(output, expected_kinds=("eval_run",))
    assert descriptor is not None
    assert "model_calls" in descriptor["components"]


def test_pending_write_cannot_be_misread_as_legacy_v0(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(_FIXTURES / "v1_eval_run", run)

    begin_artifact_write(run)

    assert not (run / ARTIFACT_DESCRIPTOR).exists()
    with pytest.raises(ValueError, match="write is still pending"):
        load_eval_run(run)


def test_provenance_sink_declares_the_components_it_wrote(tmp_path: Path) -> None:
    sink = ProvenanceSink(tmp_path / "trace")
    llm = sink.wrap_llm(type("Backend", (), {"generate": lambda self, prompt, **kw: "ok"})())
    llm.generate("hello")
    sink.trajectory_hook()

    output = sink.flush()

    descriptor = read_artifact_descriptor(output, expected_kinds=("provenance",))
    assert descriptor is not None
    assert descriptor["components"] == ["model_calls", "trajectory"]


def test_v1_model_calls_refuse_a_missing_indexed_file(tmp_path: Path) -> None:
    sink = ProvenanceSink(tmp_path / "trace")
    llm = sink.wrap_llm(type("Backend", (), {"generate": lambda self, prompt, **kw: "ok"})())
    llm.generate("hello")
    sink.trajectory_hook()
    output = sink.flush()
    (output / "call_00_prompt.txt").unlink()

    with pytest.raises(ValueError, match="model_calls index 0 requires call_00_prompt.txt"):
        read_artifact_descriptor(output)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"version": 999}, "Unsupported looplet.saved-artifact version 999"),
        ({"schema": "other.schema"}, "unsupported schema"),
        ({"kind": "provenance"}, "kind 'provenance' is not eval_run"),
        ({"components": ["artifacts", "eval_results", "future", "trajectory"]}, "unknown future"),
    ],
)
def test_explicit_incompatible_descriptors_fail_closed(
    tmp_path: Path,
    replacement: dict[str, object],
    message: str,
) -> None:
    run = tmp_path / "run"
    shutil.copytree(_FIXTURES / "v1_eval_run", run)
    descriptor_path = run / ARTIFACT_DESCRIPTOR
    descriptor = json.loads(descriptor_path.read_text())
    descriptor.update(replacement)
    descriptor_path.write_text(json.dumps(descriptor))

    with pytest.raises(ValueError, match=message):
        load_eval_run(run)


def test_v1_missing_required_component_fails_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(_FIXTURES / "v1_eval_run", run)
    (run / "evals.json").unlink()

    with pytest.raises(ValueError, match="component 'eval_results' requires evals.json"):
        load_eval_run(run)


def test_malformed_descriptor_fails_before_payload_is_used(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(_FIXTURES / "v1_eval_run", run)
    (run / ARTIFACT_DESCRIPTOR).write_text("not-json")

    with pytest.raises(ValueError, match="Invalid artifact.json"):
        load_eval_run(run)


def test_replay_refuses_v1_artifact_without_model_calls() -> None:
    with pytest.raises(ValueError, match="does not declare a model_calls component"):
        list(replay_loop(_FIXTURES / "v1_eval_run", tools=object()))
