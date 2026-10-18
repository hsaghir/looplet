"""Run, grade, capture, and replay the Quickstart without creating Python files."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from looplet import (
    DefaultState,
    EvalContext,
    EvalHook,
    MockLLMBackend,
    ProvenanceSink,
    RunResult,
    eval_assess,
    load_eval_run,
    minimal_preset,
    replay_loop,
    save_eval_run,
    tools_from,
)

EXPECTED_OWNER = {"service": "payments", "owner": "fintech-platform"}


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


def run_agent(llm, *, hooks=(), use_native_tools=False, max_steps=2):
    with minimal_preset(tools=[lookup_owner, done], max_steps=max_steps, validate=True) as run:
        run.config.use_native_tools = use_native_tools
        run.hooks.extend(hooks)
        return run.run_result(llm, task={"goal": "Find the payments owner, then finish."})


def eval_owner(ctx: EvalContext) -> bool:
    return (
        ctx.completed
        and isinstance(ctx.final_output, dict)
        and ctx.final_output.get("service") == ctx.expected["service"]
        and ctx.final_output.get("owner") == ctx.expected["owner"]
    )


def assess_result(result: RunResult):
    observed = EvalContext(
        steps=list(result.steps),
        final_output=result.output if isinstance(result.output, dict) else {},
        stop_reason=result.termination_reason,
        task={"expected": dict(EXPECTED_OWNER)},
    )
    return eval_assess([eval_owner], observed)


def capture_agent(llm, trace_dir: Path, *, max_steps=2):
    sink = ProvenanceSink(dir=trace_dir)
    grading = EvalHook(evaluators=[eval_owner], expected=EXPECTED_OWNER)
    try:
        result = run_agent(
            sink.wrap_llm(llm),
            hooks=[sink.trajectory_hook(), grading],
            max_steps=max_steps,
        )
    finally:
        saved_trace = sink.flush()
    save_eval_run(saved_trace, recorder=sink.trajectory_hook(), eval_hook=grading)
    grading.assessment().assert_passed()
    return result, saved_trace


def replay_agent(trace_dir: Path):
    state = DefaultState(max_steps=2)
    with closing(
        replay_loop(
            trace_dir,
            tools=tools_from([lookup_owner, done], validate=True),
            state=state,
            task={"goal": "Find the payments owner, then finish."},
        )
    ) as steps:
        for _step in steps:
            pass
    result = RunResult.from_state(state)
    assess_result(result).assert_passed()
    return result


def stop_after_lookup(llm):
    with minimal_preset(tools=[lookup_owner, done], max_steps=2, validate=True) as run:
        run.config.use_native_tools = False
        with closing(run.run(llm, task={"goal": "Find the payments owner, then finish."})) as steps:
            next(steps, None)
        return RunResult.from_state(run.state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--max-steps", type=int, default=2, help="Charged-step budget for a fresh run"
    )
    arguments = parser.parse_args(argv)
    if arguments.max_steps <= 0:
        parser.error("--max-steps must be positive")
    try:
        if arguments.rescore:
            saved = load_eval_run(arguments.rescore)
            report = eval_assess([eval_owner], saved.context)
            print(report.passed)
            report.assert_passed()
            return 0
        if arguments.replay:
            result = replay_agent(arguments.replay)
        elif arguments.capture:
            result, trace = capture_agent(
                scripted_backend(), arguments.capture, max_steps=arguments.max_steps
            )
            print(f"Inspect with: looplet inspect {trace}")
        elif arguments.stop_after_lookup:
            result = stop_after_lookup(scripted_backend())
        else:
            result = run_agent(scripted_backend(), max_steps=arguments.max_steps)
        print(result.output)
        print(result.completed, result.termination_reason)
        if arguments.stop_after_lookup:
            return 0
        if not arguments.capture and not arguments.replay:
            assess_result(result).assert_passed()
        print(True)
        return 0 if result.completed else 1
    except (AssertionError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
