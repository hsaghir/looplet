"""Run, grade, capture, and replay the Quickstart without creating Python files."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from shlex import quote

from looplet import (
    AnthropicBackend,
    EvalCase,
    EvalContext,
    EvalHook,
    EvalResult,
    MockLLMBackend,
    OpenAIBackend,
    ProvenanceSink,
    RunResult,
    eval_assess,
    load_cases,
    load_eval_run,
    minimal_preset,
    replay_loop,
    save_eval_run,
)
from looplet.bundles import SkillRuntime

EXPECTED_OWNER = {"service": "payments", "owner": "fintech-platform"}


def owner_case() -> EvalCase:
    """Build the default task and grader expectations together."""
    return EvalCase(
        id="owner",
        task={"goal": f"Find the {EXPECTED_OWNER['service']} owner, then finish."},
        expected=dict(EXPECTED_OWNER),
    )


def lookup_owner(service: str) -> dict:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


def done(service: str, owner: str, summary: str = "") -> dict:
    """Finish with the service, its owner, and an optional explanation."""
    return {"service": service, "owner": owner, "summary": summary}


def scripted_backend():
    return MockLLMBackend(
        [
            '{"tool":"lookup_owner","args":{"service":"payments"}}',
            '{"tool":"done","args":{"service":"payments","owner":"fintech-platform","summary":"The payments owner is fintech-platform."}}',
        ],
        cycle=False,
    )


def build(runtime: SkillRuntime):
    return minimal_preset(tools=[lookup_owner, done], max_steps=runtime.max_steps, validate=True)


def run_agent(llm, *, hooks=(), use_native_tools=False, max_steps=2, case: EvalCase | None = None):
    case = case or owner_case()
    runtime = SkillRuntime(workspace=Path.cwd(), max_steps=max_steps)
    with build(runtime) as run:
        run.config.use_native_tools = use_native_tools
        run.hooks.extend(hooks)
        return run.run_result(
            llm,
            task={
                key: value for key, value in case.task.items() if key not in {"files", "expected"}
            },
        )


def eval_owner(ctx: EvalContext) -> EvalResult:
    details = []
    if not ctx.completed:
        details.append(f"Run did not complete; stop reason: {ctx.stop_reason}.")
    if not isinstance(ctx.final_output, dict):
        details.append("Expected an object result containing service and owner.")
    else:
        for field in ("service", "owner"):
            if field not in ctx.expected:
                details.append(f"Missing grader expectation: {field}.")
            elif ctx.final_output.get(field) != ctx.expected[field]:
                details.append(
                    f"{field}: expected {ctx.expected[field]!r}, "
                    f"observed {ctx.final_output.get(field)!r}."
                )
    return EvalResult(
        score=0.0 if details else 1.0,
        label="fail" if details else "pass",
        explanation=" ".join(details),
    )


def assess_result(result: RunResult, *, case: EvalCase | None = None):
    case = case or owner_case()
    observed = EvalContext(
        steps=list(result.steps),
        final_output=result.output if isinstance(result.output, dict) else {},
        stop_reason=result.termination_reason,
        task={**case.task, "expected": dict(case.expected)},
    )
    return eval_assess([eval_owner], observed)


def capture_agent(
    llm, trace_dir: Path, *, max_steps=2, use_native_tools=False, case: EvalCase | None = None
):
    case = case or owner_case()
    sink = ProvenanceSink(dir=trace_dir)
    grading = EvalHook(evaluators=[eval_owner], expected=case.expected)
    try:
        result = run_agent(
            sink.wrap_llm(llm),
            hooks=[sink.trajectory_hook(), grading],
            max_steps=max_steps,
            use_native_tools=use_native_tools,
            case=case,
        )
    finally:
        saved_trace = sink.flush()
    save_eval_run(saved_trace, recorder=sink.trajectory_hook(), eval_hook=grading, case=case)
    try:
        grading.assessment().assert_passed()
    except AssertionError as error:
        raise AssertionError(
            f"{error}\nStop reason: {result.termination_reason}\n"
            f"Inspect with: looplet inspect {quote(str(saved_trace))}"
        ) from error
    return result, saved_trace


def replay_agent(trace_dir: Path, *, output_dir: Path | None = None, max_steps=None):
    saved = load_eval_run(trace_dir)
    source = json.loads((trace_dir / "trajectory.json").read_text(encoding="utf-8"))
    sink = ProvenanceSink(dir=output_dir, metadata={"parent_run_id": source.get("run_id")})
    grading = EvalHook(evaluators=[eval_owner], expected=saved.context.expected)
    runtime = SkillRuntime(
        workspace=Path.cwd(), max_steps=max_steps or max(saved.context.step_count, 1)
    )
    with build(runtime) as run:
        try:
            with closing(
                replay_loop(
                    trace_dir,
                    tools=run.tools,
                    state=run.state,
                    hooks=[*run.hooks, grading],
                    recording_sink=sink,
                    max_steps=max_steps,
                )
            ) as steps:
                for _step in steps:
                    pass
        finally:
            replay_trace = sink.flush()
        result = RunResult.from_state(run.state)
    save_eval_run(replay_trace, recorder=sink.trajectory_hook(), eval_hook=grading, case=saved.case)
    try:
        grading.assessment().assert_passed()
    except AssertionError as error:
        raise AssertionError(
            f"{error}\nStop reason: {result.termination_reason}\n"
            f"Inspect with: looplet inspect {quote(str(replay_trace))}"
        ) from error
    return result, replay_trace


def experiment_agent(
    llm, *, output_dir=None, max_steps=2, use_native_tools=False, case: EvalCase | None = None
):
    if output_dir is None:
        parent = Path(".looplet/experiments")
        parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="owner-", dir=parent))
    else:
        output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir = output_dir.resolve()
    (output_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    capture_dir = output_dir / "capture"
    replay_dir = output_dir / "replay"
    stages = {"capture": "not_run", "rescore": "not_run", "replay": "not_run"}
    report = {
        "capture_dir": None,
        "replay_dir": None,
        "passed": False,
        "stages": stages,
        "error": None,
    }
    stage = "capture"
    try:
        stages[stage] = "failed"
        capture_agent(
            llm, capture_dir, max_steps=max_steps, use_native_tools=use_native_tools, case=case
        )
        stages[stage] = "passed"
        print(f"Capture: looplet inspect {quote(str(capture_dir))}")
        stage = "rescore"
        stages[stage] = "failed"
        saved = load_eval_run(capture_dir)
        eval_assess([eval_owner], saved.context).assert_passed()
        stages[stage] = "passed"
        stage = "replay"
        stages[stage] = "failed"
        result, replay_trace = replay_agent(capture_dir, output_dir=replay_dir)
        stages[stage] = "passed"
        print(f"Replay: looplet inspect {quote(str(replay_trace))}")
        report["passed"] = True
        stage = "complete"
        return result
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["stage"] = stage
        report["capture_dir"] = str(capture_dir) if capture_dir.is_dir() else None
        report["replay_dir"] = str(replay_dir) if replay_dir.is_dir() else None
        report_path = output_dir / "experiment.json"
        with report_path.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        print(f"Experiment: looplet inspect {quote(str(output_dir))}")
        print(f"Experiment report: {report_path}")


def stop_after_lookup(llm, *, max_steps=2, use_native_tools=False, case: EvalCase | None = None):
    case = case or owner_case()
    runtime = SkillRuntime(workspace=Path.cwd(), max_steps=max_steps)
    with build(runtime) as run:
        run.config.use_native_tools = use_native_tools
        with closing(
            run.run(
                llm,
                task={
                    key: value
                    for key, value in case.task.items()
                    if key not in {"files", "expected"}
                },
            )
        ) as steps:
            next(steps, None)
        return RunResult.from_state(run.state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--capture", type=Path, metavar="NEW_TRACE_DIR", help="Run and save graded evidence"
    )
    actions.add_argument(
        "--rescore",
        type=Path,
        metavar="TRACE_DIR",
        help="Grade saved evidence without rerunning tools",
    )
    actions.add_argument(
        "--replay",
        type=Path,
        metavar="TRACE_DIR",
        help="Run tools again with captured model responses",
    )
    actions.add_argument(
        "--stop-after-lookup", action="store_true", help="Close the iterator before completion"
    )
    actions.add_argument(
        "--experiment",
        action="store_true",
        help="Capture, rescore, replay, and save a fresh experiment",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Step limit; defaults to two for fresh runs or the saved replay budget",
    )
    parser.add_argument("--out", type=Path, help="New replay or experiment directory")
    parser.add_argument("--case", type=Path, help="One JSON eval case for a fresh run")
    parser.add_argument("--backend", choices=("scripted", "openai", "anthropic"))
    parser.add_argument(
        "--json-text", action="store_true", help="Use JSON-text instead of native tools"
    )
    arguments = parser.parse_args(argv)
    if arguments.max_steps is not None and arguments.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if arguments.out is not None and not (arguments.replay or arguments.experiment):
        parser.error("--out requires --replay or --experiment")
    if arguments.rescore and arguments.max_steps is not None:
        parser.error("--max-steps does not apply to --rescore")
    if (arguments.replay or arguments.rescore) and (arguments.backend or arguments.json_text):
        parser.error("--replay and --rescore do not use a provider; omit --backend and --json-text")
    if (arguments.replay or arguments.rescore) and arguments.case is not None:
        parser.error("--replay and --rescore use the saved case; omit --case")
    case = None
    if arguments.case is not None:
        try:
            cases = load_cases(arguments.case)
            if len(cases) != 1:
                raise ValueError("--case requires exactly one eval case")
            case = cases[0]
            if case.task.get("files"):
                raise ValueError(
                    "--case does not seed files; use looplet eval run --factory my_agent:build"
                )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    try:
        if arguments.rescore:
            saved = load_eval_run(arguments.rescore)
            report = eval_assess([eval_owner], saved.context)
            print(report.passed)
            report.assert_passed()
            return 0
        if arguments.replay:
            result, trace = replay_agent(
                arguments.replay, output_dir=arguments.out, max_steps=arguments.max_steps
            )
            print(f"Inspect with: looplet inspect {quote(str(trace))}")
        else:
            backend_name = arguments.backend or "scripted"
            if backend_name == "scripted":
                llm = scripted_backend()
            else:
                backend = OpenAIBackend if backend_name == "openai" else AnthropicBackend
                try:
                    llm = backend.from_env()
                except (ImportError, RuntimeError, ValueError) as error:
                    print(f"error: {backend_name} setup failed: {error}", file=sys.stderr)
                    print(
                        f"See https://hsaghir.com/looplet/install/#configure-{backend_name}",
                        file=sys.stderr,
                    )
                    return 1
            options = {
                "max_steps": arguments.max_steps or 2,
                "use_native_tools": backend_name != "scripted" and not arguments.json_text,
                "case": case,
            }
            if arguments.experiment:
                result = experiment_agent(llm, output_dir=arguments.out, **options)
            elif arguments.capture:
                result, trace = capture_agent(llm, arguments.capture, **options)
                print(f"Inspect with: looplet inspect {quote(str(trace))}")
            elif arguments.stop_after_lookup:
                result = stop_after_lookup(llm, **options)
            else:
                result = run_agent(llm, **options)
        print(result.output)
        print(result.completed, result.termination_reason)
        if arguments.stop_after_lookup:
            return 0
        if not arguments.capture and not arguments.replay and not arguments.experiment:
            assess_result(result, case=case).assert_passed()
        print(True)
        return 0 if result.completed else 1
    except (AssertionError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
