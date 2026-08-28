"""Migrate a private tool loop to one outcome-grounded regression contract.

The four offline stages replace the control loop, capture a failed run,
grade observed world state, and replay fixed tool code while holding the
recorded model decisions constant. No stage loads a cartridge or provider.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from looplet import BaseToolRegistry, EvalHook, ProvenanceSink, save_eval_run, seed_case_workspace

from .execution import (
    build_registry,
    replay_with_fixed_tools,
    run_owned_loop,
    run_raw_loop,
)
from .handoff_contract import CASE, build_eval_hook, required_score
from .handoff_tools import (
    HANDOFF_FILE,
    ToolSuite,
    build_tool_suite,
    changed_implementation_lines,
    select_open_incidents,
)
from .result import MigrationResult

MARKER = ".looplet-private-loop-migration"


@dataclass(frozen=True)
class _Baseline:
    suite: ToolSuite
    registry: BaseToolRegistry
    raw_decisions: tuple[str, ...]
    owned_decisions: tuple[str, ...]
    raw_invocations: tuple[str, ...]
    owned_invocations: tuple[str, ...]
    raw_handoff: dict[str, Any]
    owned_handoff: dict[str, Any]


@dataclass(frozen=True)
class _Captured:
    decisions: tuple[str, ...]
    run_dir: Path
    hook: EvalHook


@dataclass(frozen=True)
class _Fixed:
    decisions: tuple[str, ...]
    run_dir: Path
    hook: EvalHook


def run_migration(output_dir: str | Path | None = None) -> MigrationResult:
    """Run the four cartridge-free stages and return their evidence."""
    root = _prepare_output(output_dir)
    workspaces, runs = root / "workspaces", root / "runs"
    baseline = _run_baseline(workspaces)
    captured = _capture_failure(workspaces, runs, baseline)
    fixed = _replay_fix(workspaces, runs, captured)
    return _assemble_result(root, baseline, captured, fixed)


def _run_baseline(workspaces: Path) -> _Baseline:
    raw_workspace = seed_case_workspace(CASE, workspaces / "raw_loop")
    owned_workspace = seed_case_workspace(CASE, workspaces / "composable_loop")
    suite = build_tool_suite(raw_workspace)
    registry = build_registry(suite)
    raw_decisions = run_raw_loop(suite, registry=registry, workspace=raw_workspace)
    raw_invocations = tuple(suite.invocations)
    raw_handoff = _read_json(raw_workspace / HANDOFF_FILE)
    owned_decisions = run_owned_loop(suite, registry=registry, workspace=owned_workspace)
    return _Baseline(
        suite=suite,
        registry=registry,
        raw_decisions=raw_decisions,
        owned_decisions=owned_decisions,
        raw_invocations=raw_invocations,
        owned_invocations=tuple(suite.invocations[len(raw_invocations) :]),
        raw_handoff=raw_handoff,
        owned_handoff=_read_json(owned_workspace / HANDOFF_FILE),
    )


def _capture_failure(workspaces: Path, runs: Path, baseline: _Baseline) -> _Captured:
    workspace = seed_case_workspace(CASE, workspaces / "captured_failure")
    run_dir = runs / "captured_failure"
    sink = ProvenanceSink(dir=run_dir)
    hook = build_eval_hook(workspace)
    decisions = run_owned_loop(
        baseline.suite,
        registry=baseline.registry,
        workspace=workspace,
        sink=sink,
        eval_hook=hook,
    )
    save_eval_run(run_dir, recorder=sink.trajectory_hook(), eval_hook=hook, case=CASE)
    return _Captured(decisions=decisions, run_dir=run_dir, hook=hook)


def _replay_fix(workspaces: Path, runs: Path, captured: _Captured) -> _Fixed:
    workspace = seed_case_workspace(CASE, workspaces / "replayed_fix")
    run_dir = runs / "replayed_fix"
    hook = build_eval_hook(workspace)
    suite = build_tool_suite(workspace, select_open=select_open_incidents)
    decisions = replay_with_fixed_tools(
        suite,
        captured.run_dir,
        hook,
        registry=build_registry(suite),
        workspace=workspace,
    )
    save_eval_run(run_dir, eval_hook=hook, case=CASE)
    return _Fixed(decisions=decisions, run_dir=run_dir, hook=hook)


def _assemble_result(
    root: Path,
    baseline: _Baseline,
    captured: _Captured,
    fixed: _Fixed,
) -> MigrationResult:
    return MigrationResult(
        output_dir=root,
        raw_decisions=baseline.raw_decisions,
        owned_decisions=baseline.owned_decisions,
        graded_decisions=captured.decisions,
        replayed_decisions=fixed.decisions,
        raw_invocations=baseline.raw_invocations,
        owned_invocations=baseline.owned_invocations,
        raw_handoff=baseline.raw_handoff,
        owned_handoff=baseline.owned_handoff,
        before_fix_artifacts=captured.hook.artifacts,
        after_fix_artifacts=fixed.hook.artifacts,
        before_fix_score=required_score(captured.hook.results),
        after_fix_score=required_score(fixed.hook.results),
        recorded_calls=_recorded_call_count(captured.run_dir),
        failure_run=captured.run_dir,
        fixed_run=fixed.run_dir,
        changed_lines=changed_implementation_lines(),
        raw_registry=baseline.registry,
        owned_registry=baseline.registry,
        private_callables=baseline.suite.callables,
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
    parser.add_argument(
        "--out",
        type=Path,
        help="Evidence directory (default: a unique directory under system temp)",
    )
    args = parser.parse_args(argv)
    print(render(run_migration(args.out)))
    return 0


def _reset_output(root: Path) -> None:
    """Reset only a caller-supplied prior recipe directory."""
    if root.exists():
        entries = list(root.iterdir())
        if entries and not (root / MARKER).is_file():
            raise ValueError(
                f"refusing to replace non-recipe directory {root}; choose an empty --out path"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _write_marker(root)


def _prepare_output(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        root = Path(output_dir)
        _reset_output(root)
        return root
    root = Path(tempfile.mkdtemp(prefix="looplet-private-loop-migration-"))
    _write_marker(root)
    return root


def _write_marker(root: Path) -> None:
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
