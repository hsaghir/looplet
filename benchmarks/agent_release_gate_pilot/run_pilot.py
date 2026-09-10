"""Run a controlled pilot for Looplet's agent release-gate direction.

The model decisions are captured once and replayed through fresh harness
variants. The pilot compares a trace-only gate with an independent
outcome-grounded gate so a positive result demonstrates incremental value
over ordinary run observability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from looplet import (
    EvalHook,
    ProvenanceSink,
    TrajectoryRecorder,
    cartridge_to_preset,
    load_cartridge_evals,
    replay_loop,
    save_eval_run,
    seed_case_workspace,
)
from looplet.testing import MockLLMBackend

TOOL_REL = Path("tools/publish_report/execute.py")
MODEL_RESPONSES = [
    json.dumps(
        {
            "tool": "publish_report",
            "args": {"revenue": 120, "cost": 80},
            "reasoning": "publish the requested report",
        }
    ),
    json.dumps(
        {
            "tool": "done",
            "args": {"summary": "published report.json"},
            "reasoning": "the requested artifact exists",
        }
    ),
]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    expected_outcome_pass: bool
    mutate: Callable[[str], str]


@dataclass(frozen=True)
class Result:
    name: str
    description: str
    expected_outcome_pass: bool
    trace_gate_pass: bool
    outcome_gate_pass: bool
    outcome_score: float
    observed_profit: int | None
    report_exists: bool
    tool_sequence: tuple[str, ...]
    tool_errors: int
    outcome_labels: tuple[str, ...]


def _repo_assets() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidate = parent / "examples" / "regression_demo"
        if (candidate / "report_agent.cartridge").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate examples/regression_demo assets")


def _case_and_task(cartridge: Path, workspace: Path):
    bundle = load_cartridge_evals(
        cartridge,
        runtime={"project_root": str(workspace)},
        strict=True,
    )
    if len(bundle.cases) != 1:
        raise RuntimeError(f"expected one pilot case, found {len(bundle.cases)}")
    case = bundle.cases[0]
    task = {key: value for key, value in case.task.items() if key != "files"}
    return bundle, case, task


def _make_eval_hook(bundle: Any, case: Any) -> EvalHook:
    return EvalHook(
        evaluators=bundle.graders,
        collectors=bundle.collectors,
        expected=case.expected,
    )


def _capture_baseline(cartridge: Path, workspace: Path, run_dir: Path) -> Path:
    bundle, case, task = _case_and_task(cartridge, workspace)
    seed_case_workspace(case, workspace)
    preset = cartridge_to_preset(cartridge, runtime={"project_root": str(workspace)})
    eval_hook = _make_eval_hook(bundle, case)
    sink = ProvenanceSink(dir=run_dir)
    llm = sink.wrap_llm(MockLLMBackend(responses=list(MODEL_RESPONSES)))
    recorder = sink.trajectory_hook()
    preset.hooks = [*preset.hooks, recorder, eval_hook]
    try:
        list(preset.run(llm, task=task))
    finally:
        preset.close()
    save_eval_run(run_dir, recorder=recorder, eval_hook=eval_hook, case=case)
    return run_dir


def _replay_variant(
    cartridge: Path,
    workspace: Path,
    source_run: Path,
    run_dir: Path,
) -> Result:
    bundle, case, task = _case_and_task(cartridge, workspace)
    seed_case_workspace(case, workspace)
    preset = cartridge_to_preset(cartridge, runtime={"project_root": str(workspace)})
    eval_hook = _make_eval_hook(bundle, case)
    recorder = TrajectoryRecorder()
    try:
        steps = list(
            replay_loop(
                source_run,
                tools=preset.tools,
                state=preset.state,
                hooks=[*preset.hooks, recorder, eval_hook],
                config=preset.config,
                task=task,
            )
        )
    finally:
        preset.close()
    save_eval_run(run_dir, recorder=recorder, eval_hook=eval_hook, case=case)

    tool_sequence = tuple(step.tool_call.tool for step in steps if step.tool_call is not None)
    tool_errors = sum(
        1 for step in steps if step.tool_result is not None and step.tool_result.error
    )
    trace_gate_pass = (
        tool_sequence == ("publish_report", "done") and tool_errors == 0 and bool(steps)
    )
    outcome_results = tuple(eval_hook.results)
    required_names = {
        fn.__name__ for fn in bundle.graders if "required" in getattr(fn, "_eval_marks", set())
    }
    required = [result for result in outcome_results if result.name in required_names]
    outcome_gate_pass = bool(required) and all(result.passed for result in required)
    score = min((float(result.score or 0.0) for result in required), default=0.0)
    artifacts = eval_hook.artifacts
    return Result(
        name=cartridge.parent.name,
        description="",
        expected_outcome_pass=False,
        trace_gate_pass=trace_gate_pass,
        outcome_gate_pass=outcome_gate_pass,
        outcome_score=score,
        observed_profit=artifacts.get("observed_profit"),
        report_exists=bool(artifacts.get("report_exists")),
        tool_sequence=tool_sequence,
        tool_errors=tool_errors,
        outcome_labels=tuple(result.label for result in outcome_results),
    )


def _variants() -> tuple[Variant, ...]:
    return (
        Variant("control_correct", "Shipped implementation", True, lambda source: source),
        Variant(
            "benign_formatting",
            "Sort JSON keys without changing the observed contract",
            True,
            lambda source: source.replace(
                "json.dumps(report, indent=2)",
                "json.dumps(report, indent=2, sort_keys=True)",
            ),
        ),
        Variant(
            "mutation_additive_profit",
            "Adds cost instead of subtracting it",
            False,
            lambda source: source.replace("revenue - cost", "revenue + cost"),
        ),
        Variant(
            "mutation_ignore_cost",
            "Uses gross revenue as profit",
            False,
            lambda source: source.replace("revenue - cost", "revenue"),
        ),
        Variant(
            "mutation_reverse_sign",
            "Reverses the profit sign",
            False,
            lambda source: source.replace("revenue - cost", "cost - revenue"),
        ),
        Variant(
            "mutation_wrong_field",
            "Writes a different field name while returning success",
            False,
            lambda source: source.replace(
                '"profit": revenue - cost',
                '"profit_after_tax": revenue - cost',
            ).replace(
                '"profit": report["profit"]',
                '"profit": report["profit_after_tax"]',
            ),
        ),
        Variant(
            "mutation_skip_write",
            "Returns success without writing the requested artifact",
            False,
            lambda source: source.replace(
                '(root / "report.json").write_text(json.dumps(report, indent=2) + "\\n")',
                "pass",
            ),
        ),
    )


def _render(results: list[Result], output_dir: Path) -> str:
    lines = [
        "# Agent release-gate pilot",
        "",
        "Controlled replay comparison: trace-only gate versus independent outcome gate.",
        "",
        "| Variant | Expected | Trace gate | Outcome gate | Profit | Artifact |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in results:
        expected = "pass" if result.expected_outcome_pass else "fail"
        trace = "PASS" if result.trace_gate_pass else "FAIL"
        outcome = "PASS" if result.outcome_gate_pass else "FAIL"
        profit = "-" if result.observed_profit is None else str(result.observed_profit)
        artifact = "yes" if result.report_exists else "no"
        lines.append(
            f"| `{result.name}` | {expected} | {trace} | {outcome} | {profit} | {artifact} |"
        )
    seeded = [result for result in results if not result.expected_outcome_pass]
    controls = [result for result in results if result.expected_outcome_pass]
    trace_false_negative = sum(result.trace_gate_pass for result in seeded)
    outcome_true_positive = sum(not result.outcome_gate_pass for result in seeded)
    control_false_positive = sum(not result.outcome_gate_pass for result in controls)
    lines.extend(
        [
            "",
            "## Pilot result",
            "",
            f"- Seeded regressions accepted by trace-only gate: {trace_false_negative}/{len(seeded)}.",
            f"- Seeded regressions rejected by outcome gate: {outcome_true_positive}/{len(seeded)}.",
            f"- Expected-safe variants rejected by outcome gate: {control_false_positive}/{len(controls)}.",
            f"- Evidence directory: `{output_dir}`.",
            "",
            "This supports the narrow capability claim when the first two counts are complete and the last is zero.",
            "It does not establish user demand, fresh-model quality, or diagnosis time with human operators.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> list[Result]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    assets = _repo_assets()
    source = assets / "report_agent.cartridge" / TOOL_REL
    baseline_cartridge = output_dir / "cartridges" / "baseline"
    baseline_cartridge.parent.mkdir(parents=True)
    shutil.copytree(assets / "report_agent.cartridge", baseline_cartridge)
    baseline_run = output_dir / "runs" / "baseline"
    _capture_baseline(
        baseline_cartridge,
        output_dir / "workspaces" / "baseline",
        baseline_run,
    )

    results: list[Result] = []
    for variant in _variants():
        cartridge = output_dir / "cartridges" / variant.name
        shutil.copytree(assets / "report_agent.cartridge", cartridge)
        (cartridge / TOOL_REL).write_text(
            variant.mutate(source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        result = _replay_variant(
            cartridge,
            output_dir / "workspaces" / variant.name,
            baseline_run,
            output_dir / "runs" / variant.name,
        )
        results.append(
            Result(
                **{
                    **asdict(result),
                    "name": variant.name,
                    "description": variant.description,
                    "expected_outcome_pass": variant.expected_outcome_pass,
                }
            )
        )

    (output_dir / "pilot.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(_render(results, output_dir), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "looplet-agent-release-gate-pilot",
    )
    args = parser.parse_args()
    results = run(args.out)
    print(_render(results, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
