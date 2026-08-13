"""The private-loop migration recipe must stay executable and cartridge-free."""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

    _assert_cartridge_free()
    _assert_initial_migration(result, tmp_path)
    _assert_outcome_contract(result)
    _assert_capture_and_replay(result)


def _assert_cartridge_free() -> None:
    for module in sorted(RECIPE_DIR.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        for symbol in CARTRIDGE_SYMBOLS:
            assert symbol not in source, f"{module.name} reaches for {symbol}"


def _assert_initial_migration(result, tmp_path: Path) -> None:
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


def _assert_outcome_contract(result) -> None:
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


def _assert_capture_and_replay(result) -> None:
    assert result.recorded_calls == 3
    assert result.replayed_decisions == result.graded_decisions
    assert result.same_recorded_decisions
    assert result.before_fix_artifacts != result.after_fix_artifacts

    failure_run = result.failure_run
    fixed_run = result.fixed_run
    assert _read_json(failure_run / "artifacts.json") == result.before_fix_artifacts
    assert _read_json(fixed_run / "artifacts.json") == result.after_fix_artifacts
    assert (failure_run / "manifest.jsonl").is_file()
    assert (failure_run / "call_00_response.txt").is_file()

    assert _read_json(failure_run / "expected.json") == recipe.CASE.expected
    assert "expected" not in _read_json(failure_run / "trajectory.json")["task"]
    assert '"expected"' not in (failure_run / "call_00_prompt.txt").read_text(encoding="utf-8")


def test_raw_and_migrated_stages_share_exact_registry_and_tools(tmp_path: Path) -> None:
    """Replacing the loop must not rebuild the approved tool boundary."""
    result = recipe.run_migration(tmp_path / "identity-evidence")

    assert result.raw_registry is result.owned_registry
    for name in ("list_incidents", "publish_handoff"):
        raw_spec = result.raw_registry._tools[name]
        owned_spec = result.owned_registry._tools[name]
        assert raw_spec is owned_spec
        assert raw_spec.execute is result.private_callables[name]


def _run_with_default_output(_index: int):
    return recipe.run_migration()


def test_overlapping_default_runs_receive_distinct_evidence_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent library calls must not share or reset an evidence tree."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_run_with_default_output, range(4)))

    roots = {result.output_dir for result in results}
    assert len(roots) == len(results)
    assert len({id(result.raw_registry) for result in results}) == len(results)
    for result in results:
        assert result.raw_handoff == result.owned_handoff
        assert result.before_fix_artifacts["observed_open_count"] == 4
        assert result.after_fix_artifacts["observed_open_count"] == 2
        assert (result.failure_run / "manifest.jsonl").is_file()
        assert (result.fixed_run / "artifacts.json").is_file()


def test_sequential_default_runs_preserve_previous_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later implicit run must leave the prior packet untouched."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    first = recipe.run_migration()
    note = first.output_dir / "consumer-note.txt"
    note.write_text("retain this packet\n", encoding="utf-8")

    second = recipe.run_migration()

    assert second.output_dir != first.output_dir
    assert second.raw_registry is not first.raw_registry
    assert note.read_text(encoding="utf-8") == "retain this packet\n"


def test_explicit_output_path_is_the_only_resettable_target(tmp_path: Path) -> None:
    """A caller-supplied path deliberately replaces its prior recipe packet."""
    output = tmp_path / "chosen-evidence"
    first = recipe.run_migration(output)
    note = output / "consumer-note.txt"
    note.write_text("replace me\n", encoding="utf-8")

    second = recipe.run_migration(output)

    assert first.output_dir == second.output_dir == output
    assert not note.exists()
