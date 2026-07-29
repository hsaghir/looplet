"""The private-loop migration recipe must stay executable and cartridge-free."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import private_loop_migration as recipe

pytestmark = pytest.mark.smoke

RECIPE_DIR = Path(recipe.__file__ or "").parent

# The four stages must reach the same seam without a cartridge loader.
CARTRIDGE_SYMBOLS = (
    "cartridge_to_preset",
    "load_cartridge_evals",
    "run_cartridge_evals",
    "looplet.cartridge",
    "AgentPreset",
    ".cartridge",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_recipe_migrates_a_private_loop_to_a_green_outcome_contract(tmp_path: Path) -> None:
    result = recipe.run_migration(tmp_path / "evidence")

    for module in sorted(RECIPE_DIR.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        for symbol in CARTRIDGE_SYMBOLS:
            assert symbol not in source, f"{module.name} reaches for {symbol}"

    # 1. The original tool implementation is reused through the migration.
    assert result.raw_decisions == ("list_incidents", "publish_handoff", "done")
    assert result.owned_decisions == result.raw_decisions
    assert result.raw_invocations == ("list_incidents", "publish_handoff")
    assert result.reuses_private_tools
    assert result.loop_swap_kept_the_outcome
    assert result.raw_handoff["open_count"] == 4

    before_fix = recipe.build_tool_suite(tmp_path / "suite_before")
    after_loop_swap = recipe.build_tool_suite(tmp_path / "suite_after")
    assert before_fix.select_open is recipe.select_open_incidents_with_bug
    assert (
        before_fix.callables["publish_handoff"].__code__
        is after_loop_swap.callables["publish_handoff"].__code__
    )
    registry = recipe.build_registry(after_loop_swap)
    assert registry.tool_names == ["list_incidents", "publish_handoff", "done"]

    # 2. The collector reads world state rather than a preferred tool sequence.
    assert result.before_fix_artifacts["observed_open_count"] == 4
    assert result.after_fix_artifacts["observed_open_count"] == 2
    assert result.after_fix_artifacts["observed_open_ids"] == recipe.CASE.expected["open_ids"]
    collector = recipe.make_handoff_collector(result.output_dir / "workspaces" / "replayed_fix")
    assert collector(None) == result.after_fix_artifacts

    # 3. The required grader fails before the intentional fix and passes after it.
    assert result.before_fix_score == 0.0
    assert result.after_fix_score == 1.0
    assert result.changed_lines == (
        "return list(incidents)",
        'return [incident for incident in incidents if incident["status"] != "resolved"]',
    )

    # 4. Replay consumed the recorded decisions while the observed outcome changed.
    assert result.recorded_calls == 3
    assert result.replayed_decisions == result.graded_decisions
    assert result.same_recorded_decisions
    assert result.before_fix_artifacts != result.after_fix_artifacts

    # The verdict came from collected artifacts, not from trajectory shape.
    failure_run = result.failure_run
    fixed_run = result.fixed_run
    assert _read_json(failure_run / "artifacts.json") == result.before_fix_artifacts
    assert _read_json(fixed_run / "artifacts.json") == result.after_fix_artifacts
    assert (failure_run / "manifest.jsonl").is_file()
    assert (failure_run / "call_00_response.txt").is_file()

    # Grader-only expected data stayed out of the run the model saw.
    assert _read_json(failure_run / "expected.json") == recipe.CASE.expected
    assert "expected" not in _read_json(failure_run / "trajectory.json")["task"]
    assert '"expected"' not in (failure_run / "call_00_prompt.txt").read_text(encoding="utf-8")
