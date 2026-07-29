"""Migrate a private tool loop to one outcome-grounded regression contract.

Four stages, no cartridge and no provider:

1. the private while-loop is replaced by ``composable_loop()`` while the
   same tool callables run through one ``tools_from(...)`` registry;
2. the failing run is captured as readable files with ``ProvenanceSink``;
3. an ``EvalHook`` collector reads the resulting workspace and a required
   grader turns that observation into a contract;
4. one tool implementation is fixed and ``replay_loop()`` re-executes the
   recorded model decisions against it.

Replay is the right tool only in stage four, because the model responses
are the variable being held fixed. Tool code executes again, so the
workspace, the collected artifacts, and the verdict are all fresh.

Two notes on how the stages are wired. The scripted ``MockLLMBackend``
stands in for the team's provider client so every stage runs offline;
nothing else about the private harness is a test double. Stages two and
three share one run, because the contract grades the failure it captured.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from looplet import (
    BaseToolRegistry,
    DefaultState,
    EvalHook,
    LoopConfig,
    ProvenanceSink,
    composable_loop,
    replay_loop,
    save_eval_run,
    seed_case_workspace,
    tools_from,
)
from looplet.testing import MockLLMBackend

from .handoff_contract import CASE, build_eval_hook, live_task, required_score
from .handoff_tools import (
    HANDOFF_FILE,
    ToolSuite,
    build_tool_suite,
    changed_implementation_lines,
    select_open_incidents,
)

DONE_TOOL = "done"
MAX_STEPS = 4
MARKER = ".looplet-private-loop-migration"
SYSTEM_PROMPT = "You hand an on-call shift over. Record the work the next owner still carries."

MODEL_RESPONSES = [
    json.dumps(
        {
            "tool": "list_incidents",
            "args": {},
            "reasoning": "read the shift log before writing anything",
        }
    ),
    json.dumps(
        {
            "tool": "publish_handoff",
            "args": {"owner": "day-shift"},
            "reasoning": "write the handoff file",
        }
    ),
    json.dumps(
        {
            "tool": DONE_TOOL,
            "args": {"summary": "handoff file written for day-shift"},
            "reasoning": "the requested file exists",
        }
    ),
]


@dataclass(frozen=True)
class MigrationResult:
    """Evidence produced by :func:`run_migration`."""

    output_dir: Path
    raw_decisions: tuple[str, ...]
    owned_decisions: tuple[str, ...]
    graded_decisions: tuple[str, ...]
    replayed_decisions: tuple[str, ...]
    raw_invocations: tuple[str, ...]
    owned_invocations: tuple[str, ...]
    raw_handoff: dict[str, Any]
    owned_handoff: dict[str, Any]
    before_fix_artifacts: dict[str, Any]
    after_fix_artifacts: dict[str, Any]
    before_fix_score: float
    after_fix_score: float
    recorded_calls: int
    failure_run: Path
    fixed_run: Path
    changed_lines: tuple[str, str]

    @property
    def reuses_private_tools(self) -> bool:
        """True when both loops dispatched the same private callables."""
        return bool(self.raw_invocations) and self.raw_invocations == self.owned_invocations

    @property
    def loop_swap_kept_the_outcome(self) -> bool:
        """True when replacing the loop changed neither decisions nor world state."""
        return self.raw_decisions == self.owned_decisions and self.raw_handoff == self.owned_handoff

    @property
    def same_recorded_decisions(self) -> bool:
        """True when replay consumed the decisions the failing run recorded."""
        return self.replayed_decisions == self.graded_decisions


def build_registry(suite: ToolSuite) -> BaseToolRegistry:
    """The one ``tools_from(...)`` registry every Looplet stage reuses."""
    return tools_from(list(suite.callables.values()), include_done=True)


def run_raw_loop(suite: ToolSuite) -> tuple[str, ...]:
    """The private while-loop this recipe migrates away from.

    It owns prompt assembly, response parsing, dispatch, and the stop
    condition, and it has no interception point for capture, permissions,
    or parse recovery. Those are the responsibilities stage one hands to
    ``composable_loop()``.
    """
    llm = MockLLMBackend(responses=list(MODEL_RESPONSES), cycle=False)
    task = live_task()
    decisions: list[str] = []
    observations: list[str] = []
    for _ in range(MAX_STEPS):
        response = llm.generate(_raw_prompt(task, observations), system_prompt=SYSTEM_PROMPT)
        call = json.loads(response)
        name = str(call["tool"])
        decisions.append(name)
        if name == DONE_TOOL:
            break
        result = suite.callables[name](**call.get("args", {}))
        observations.append(f"{name} -> {json.dumps(result)}")
    return tuple(decisions)


def run_owned_loop(
    suite: ToolSuite,
    *,
    sink: ProvenanceSink | None = None,
    eval_hook: EvalHook | None = None,
) -> tuple[str, ...]:
    """Run the same tools through ``composable_loop()``.

    Passing neither ``sink`` nor ``eval_hook`` is stage one: the loop is
    the only thing that changed. Stages two and three add each argument
    without touching the tools.
    """
    llm: Any = MockLLMBackend(responses=list(MODEL_RESPONSES), cycle=False)
    hooks: list[Any] = []
    if sink is not None:
        llm = sink.wrap_llm(llm)
        hooks.append(sink.trajectory_hook())
    if eval_hook is not None:
        hooks.append(eval_hook)
    steps = list(
        composable_loop(
            llm=llm,
            tools=build_registry(suite),
            hooks=hooks,
            task=live_task(),
            config=_loop_config(),
            state=DefaultState(max_steps=MAX_STEPS),
        )
    )
    if sink is not None:
        sink.flush()
    return tuple(step.tool_call.tool for step in steps)


def replay_with_fixed_tools(
    suite: ToolSuite,
    trace_dir: Path,
    eval_hook: EvalHook,
) -> tuple[str, ...]:
    """Re-execute the recorded decisions against the fixed implementation."""
    steps = list(
        replay_loop(
            trace_dir,
            tools=build_registry(suite),
            state=DefaultState(max_steps=MAX_STEPS),
            hooks=[eval_hook],
            config=_loop_config(),
            task=live_task(),
        )
    )
    return tuple(step.tool_call.tool for step in steps)


def run_migration(output_dir: str | Path | None = None) -> MigrationResult:
    """Run the four cartridge-free stages and return their persisted evidence."""
    root = Path(output_dir or (Path(tempfile.gettempdir()) / "looplet-private-loop-migration"))
    _reset_output(root)
    workspaces = root / "workspaces"
    runs = root / "runs"

    # Stage 1: replace the loop, keep the tools.
    raw_suite = build_tool_suite(seed_case_workspace(CASE, workspaces / "raw_loop"))
    raw_decisions = run_raw_loop(raw_suite)
    owned_suite = build_tool_suite(seed_case_workspace(CASE, workspaces / "composable_loop"))
    owned_decisions = run_owned_loop(owned_suite)

    # Stages 2 and 3: capture the failing run, then grade what the host sees.
    failure_workspace = seed_case_workspace(CASE, workspaces / "captured_failure")
    failure_run = runs / "captured_failure"
    failure_sink = ProvenanceSink(dir=failure_run)
    failure_hook = build_eval_hook(failure_workspace)
    graded_decisions = run_owned_loop(
        build_tool_suite(failure_workspace),
        sink=failure_sink,
        eval_hook=failure_hook,
    )
    save_eval_run(
        failure_run,
        recorder=failure_sink.trajectory_hook(),
        eval_hook=failure_hook,
        case=CASE,
    )

    # Stage 4: fix one implementation, replay the recorded decisions.
    fixed_workspace = seed_case_workspace(CASE, workspaces / "replayed_fix")
    fixed_run = runs / "replayed_fix"
    fixed_hook = build_eval_hook(fixed_workspace)
    replayed_decisions = replay_with_fixed_tools(
        build_tool_suite(fixed_workspace, select_open=select_open_incidents),
        failure_run,
        fixed_hook,
    )
    save_eval_run(fixed_run, eval_hook=fixed_hook, case=CASE)

    return MigrationResult(
        output_dir=root,
        raw_decisions=raw_decisions,
        owned_decisions=owned_decisions,
        graded_decisions=graded_decisions,
        replayed_decisions=replayed_decisions,
        raw_invocations=tuple(raw_suite.invocations),
        owned_invocations=tuple(owned_suite.invocations),
        raw_handoff=_read_json(raw_suite.workspace / HANDOFF_FILE),
        owned_handoff=_read_json(owned_suite.workspace / HANDOFF_FILE),
        before_fix_artifacts=failure_hook.artifacts,
        after_fix_artifacts=fixed_hook.artifacts,
        before_fix_score=required_score(failure_hook.results),
        after_fix_score=required_score(fixed_hook.results),
        recorded_calls=_recorded_call_count(failure_run),
        failure_run=failure_run,
        fixed_run=fixed_run,
        changed_lines=changed_implementation_lines(),
    )


def render(result: MigrationResult) -> str:
    """Render a stable summary of the four stages."""
    expected_open = CASE.expected["open_count"]
    before_line, after_line = result.changed_lines
    return "\n".join(
        [
            "Looplet: one private tool loop -> one outcome-grounded regression contract",
            "",
            "1. REPLACE the loop, keep the private tools",
            f"   raw loop:        {_route(result.raw_decisions)}",
            f"   composable_loop: {_route(result.owned_decisions)}",
            f"   private callables that ran: {_route(result.owned_invocations)}",
            f"   same tools reused:      {_flag(result.reuses_private_tools)}",
            f"   same handoff written:   {_flag(result.loop_swap_kept_the_outcome)}",
            "",
            "2. CAPTURE the failing run as ordinary files",
            f"   model calls recorded: {result.recorded_calls}",
            f"   evidence: {result.failure_run}",
            "",
            "3. GRADE the outcome the host observed, not the route",
            f"   collected open_count: {result.before_fix_artifacts.get('observed_open_count')}"
            f" (expected {expected_open})",
            f"   required eval: {_verdict(result.before_fix_score)}",
            "",
            "4. FIX one tool implementation and replay the recorded decisions",
            f"   - {before_line}",
            f"   + {after_line}",
            f"   replayed:                {_route(result.replayed_decisions)}",
            f"   same recorded decisions: {_flag(result.same_recorded_decisions)}",
            f"   collected open_count: {result.after_fix_artifacts.get('observed_open_count')}"
            f" (expected {expected_open})",
            f"   required eval: {_verdict(result.after_fix_score)}",
            "",
            f"Evidence: {result.output_dir}",
            "No stage above loads a cartridge. Replay holds the model responses fixed;",
            "the tools execute again in a fresh workspace.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Evidence directory (default: system temp)")
    args = parser.parse_args(argv)
    print(render(run_migration(args.out)))
    return 0


def _loop_config() -> LoopConfig:
    return LoopConfig(
        max_steps=MAX_STEPS,
        use_native_tools=False,
        system_prompt=SYSTEM_PROMPT,
    )


def _raw_prompt(task: dict[str, Any], observations: list[str]) -> str:
    """Prompt assembly the private loop hand-rolled before the migration."""
    lines = [
        f"goal: {task['goal']}",
        "",
        "tools: list_incidents(), publish_handoff(owner), done(summary)",
    ]
    if observations:
        lines += ["", "observations:", *observations]
    lines += ["", 'Reply with {"tool": ..., "args": {...}}.']
    return "\n".join(lines)


def _reset_output(root: Path) -> None:
    """Reset a prior evidence directory without deleting unrelated data."""
    if root.exists():
        entries = list(root.iterdir())
        if entries and not (root / MARKER).is_file():
            raise ValueError(
                f"refusing to replace non-recipe directory {root}; choose an empty --out path"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / MARKER).write_text(
        "generated by examples/private_loop_migration/run_recipe.py\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _recorded_call_count(run_dir: Path) -> int:
    manifest = (run_dir / "manifest.jsonl").read_text(encoding="utf-8")
    return len([line for line in manifest.splitlines() if line.strip()])


def _route(names: tuple[str, ...]) -> str:
    return " -> ".join(names)


def _flag(value: bool) -> str:
    return str(value).lower()


def _verdict(score: float) -> str:
    return f"{'PASS' if score >= 0.5 else 'FAIL'} ({score:.2f})"


if __name__ == "__main__":
    raise SystemExit(main())
