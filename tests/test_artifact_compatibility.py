"""Compatibility contract for versioned saved provenance and eval artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from looplet import EvalContext, __version__, load_eval_run, save_eval_run, tool, tools_from
from looplet.artifact_compat import (
    ARTIFACT_DESCRIPTOR,
    ARTIFACT_PENDING,
    ARTIFACT_SCHEMA,
    ARTIFACT_VERSION,
    begin_artifact_write,
    read_artifact_descriptor,
)
from looplet.provenance import ProvenanceSink, _load_trace_calls, async_replay_loop, replay_loop

_FIXTURES = Path(__file__).parent / "fixtures" / "artifact_compat"


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize(
    ("filename", "legacy"),
    [
        (filename, legacy)
        for legacy in (False, True)
        for filename in (
            "artifact.json",
            "artifacts.json",
            "evals.json",
            "trajectory.json",
            "case.json",
            "expected.json",
            "metrics.json",
        )
        if not (legacy and filename == ARTIFACT_DESCRIPTOR)
    ],
)
def test_saved_eval_components_cannot_link_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, legacy: bool, absolute: bool
) -> None:
    run = tmp_path / "run"
    shutil.copytree(_FIXTURES / "v1_eval_run", run)
    if legacy:
        (run / ARTIFACT_DESCRIPTOR).unlink()
    component = run / filename
    if not component.exists():
        case = json.loads((run / "case.json").read_text())
        component.write_text(
            json.dumps(case.get("expected", {}) if filename == "expected.json" else {})
        )
    private = tmp_path / "run-private"
    private.mkdir()
    outside = private / filename
    outside.write_bytes(component.read_bytes())
    component.unlink()
    component.symlink_to(outside if absolute else Path("..") / private.name / filename)
    open_file = Path.open
    outside_reads = []

    def observe_open(path: Path, *args, **kwargs):
        if path.resolve() == outside:
            outside_reads.append(path)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_open)

    with pytest.raises(ValueError, match="outside|escape"):
        load_eval_run(run)

    assert outside_reads == []


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize(
    ("filename", "legacy"),
    [
        (filename, legacy)
        for legacy in (False, True)
        for filename in (
            "artifact.json",
            "artifacts.json",
            "evals.json",
            "trajectory.json",
            "case.json",
        )
        if not (legacy and filename == ARTIFACT_DESCRIPTOR)
    ],
)
def test_saved_eval_links_inside_artifact_remain_supported(
    tmp_path: Path, filename: str, legacy: bool, absolute: bool
) -> None:
    run = tmp_path / "run"
    shutil.copytree(_FIXTURES / "v1_eval_run", run)
    if legacy:
        (run / ARTIFACT_DESCRIPTOR).unlink()
    payload = run / "payload"
    payload.mkdir()
    component = run / filename
    target = payload / filename
    component.rename(target)
    component.symlink_to(target if absolute else Path("payload") / filename)

    record = load_eval_run(run)

    assert record.context.final_output == {"summary": "fixture complete"}
    assert record.context.artifacts == {"tests_passing": True}
    assert record.case is not None
    assert record.case.id == "fixture-v1"


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize(
    ("layout", "filename"),
    [
        ("v1", "manifest.jsonl"),
        ("v1", "call_00_prompt.txt"),
        ("v1", "call_00_response.txt"),
        ("legacy_manifest", "manifest.jsonl"),
        ("legacy_manifest", "call_00_response.txt"),
        ("legacy_fallback", "call_00_response.txt"),
    ],
)
def test_replay_call_files_cannot_link_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, filename: str, absolute: bool
) -> None:
    sink = ProvenanceSink(tmp_path / "trace")
    llm = sink.wrap_llm(type("Backend", (), {"generate": lambda self, prompt, **kw: "ok"})())
    llm.generate("hello")
    sink.trajectory_hook()
    trace = sink.flush()
    manifest = trace / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for entry in entries:
        entry.pop("capture", None)
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    if layout != "v1":
        (trace / ARTIFACT_DESCRIPTOR).unlink()
    if layout == "legacy_fallback":
        manifest.unlink()
    component = trace / filename
    private = tmp_path / "trace-private"
    private.mkdir()
    outside = private / filename
    outside.write_bytes(component.read_bytes())
    component.unlink()
    component.symlink_to(outside if absolute else Path("..") / private.name / filename)
    open_file = Path.open
    outside_reads = []

    def observe_open(path: Path, *args, **kwargs):
        if path.resolve() == outside:
            outside_reads.append(path)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_open)

    with pytest.raises(ValueError, match="outside|escape"):
        _load_trace_calls(trace)

    assert outside_reads == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("contained", [False, True])
@pytest.mark.parametrize("filename", ["trajectory.json", "call_00_response.txt"])
async def test_replay_link_boundaries_control_real_tool_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    contained: bool,
    filename: str,
) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "call_00_response.txt").write_text(json.dumps({"tool": "record_effect", "args": {}}))
    (trace / "trajectory.json").write_text(json.dumps({"task": {}, "step_count": 1}))
    payload = trace / "payload" if contained else tmp_path / "trace-private"
    payload.mkdir()
    component = trace / filename
    target = payload / filename
    component.rename(target)
    component.symlink_to(target)
    effect = tmp_path / "effect.txt"
    open_file = Path.open
    outside_reads = []

    def observe_open(path: Path, *args, **kwargs):
        if not contained and path.resolve() == target:
            outside_reads.append(path)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_open)

    @tool
    def record_effect() -> str:
        effect.write_text("executed")
        return "written"

    tools = tools_from([record_effect], include_done=True)

    async def replay():
        if asynchronous:
            return [step async for step in async_replay_loop(trace, tools=tools, max_steps=1)]
        return list(replay_loop(trace, tools=tools, max_steps=1))

    if contained:
        steps = await replay()
        assert len(steps) == 1
        assert steps[0].tool_result.data == "written"
        assert effect.read_text() == "executed"
    else:
        with pytest.raises(ValueError, match="outside|escape"):
            await replay()
        assert outside_reads == []
        assert not effect.exists()


@pytest.mark.parametrize("filename", [ARTIFACT_DESCRIPTOR, ARTIFACT_PENDING, "manifest.jsonl"])
def test_dangling_artifact_markers_do_not_enable_legacy_fallback(
    tmp_path: Path, filename: str
) -> None:
    sink = ProvenanceSink(tmp_path / "trace")
    llm = sink.wrap_llm(type("Backend", (), {"generate": lambda self, prompt, **kw: "ok"})())
    llm.generate("hello")
    trace = sink.flush()
    (trace / ARTIFACT_DESCRIPTOR).unlink()
    marker = trace / filename
    marker.unlink(missing_ok=True)
    marker.symlink_to("missing-target")

    with pytest.raises(ValueError, match="missing target"):
        _load_trace_calls(trace)


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
