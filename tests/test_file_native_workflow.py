"""Compatibility contract for Looplet's file-native evidence workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import BaseToolRegistry, DefaultState, LoopConfig, ProvenanceSink, composable_loop
from looplet.__main__ import main as cli_main
from looplet.evals import EvalContext, eval_run
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _capture_publish_run(
    *,
    trace_dir: Path,
    workspace: Path,
    value: str,
    metadata: dict[str, str] | None = None,
) -> Path:
    workspace.mkdir(parents=True)
    output_path = workspace / "result.json"

    def publish(*, value: str) -> dict[str, str]:
        output_path.write_text(json.dumps({"value": value}))
        return {"path": str(output_path)}

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="publish",
            description="Publish one value to the result artifact.",
            parameters={"value": "str"},
            execute=publish,
        )
    )
    tools.register(
        ToolSpec(
            name="done",
            description="Finish the run.",
            parameters={"summary": "str"},
            execute=lambda *, summary: {"summary": summary},
        )
    )

    responses = [
        json.dumps({"tool": "publish", "args": {"value": value}, "reasoning": "publish"}),
        json.dumps({"tool": "done", "args": {"summary": "published"}, "reasoning": "done"}),
    ]
    sink = ProvenanceSink(dir=trace_dir, metadata=metadata)
    llm = sink.wrap_llm(MockLLMBackend(responses=responses, cycle=False))
    list(
        composable_loop(
            llm=llm,
            tools=tools,
            state=DefaultState(max_steps=3),
            hooks=[sink.trajectory_hook()],
            config=LoopConfig(max_steps=3),
            task={"goal": f"Publish {value}"},
        )
    )
    sink.flush()
    return output_path


def test_capture_inspect_link_and_grade_outcome(tmp_path: Path, capsys) -> None:
    parent_trace = tmp_path / "traces" / "parent"
    _capture_publish_run(
        trace_dir=parent_trace,
        workspace=tmp_path / "workspaces" / "parent",
        value="draft",
    )

    assert cli_main(["show", str(parent_trace), "--json"]) == 0
    parent_view = json.loads(capsys.readouterr().out)
    parent_run_id = parent_view["trajectory"]["run_id"]

    child_trace = tmp_path / "traces" / "child"
    child_output = _capture_publish_run(
        trace_dir=child_trace,
        workspace=tmp_path / "workspaces" / "child",
        value="verified",
        metadata={"parent_run_id": parent_run_id},
    )

    observed = json.loads(child_output.read_text())
    (child_trace / "artifacts.json").write_text(json.dumps({"published_value": observed["value"]}))
    context = EvalContext.from_trajectory_dir(child_trace)

    def eval_published_value(ctx: EvalContext) -> bool:
        return ctx.artifacts.get("published_value") == "verified"

    results = eval_run([eval_published_value], context)

    assert context.completed is True
    assert context.metadata["parent_run_id"] == parent_run_id
    assert results[0].passed is True
