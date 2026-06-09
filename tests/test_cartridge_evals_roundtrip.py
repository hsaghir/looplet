"""Evals ship *inside* the cartridge directory, but as an adjacent
artifact owned by ``looplet.evals`` — not by the cartridge package.

This is the first slice of "evals ship with the agent version":
``<cartridge>/evals/cases/`` carries the eval corpus as pure data, so it
is version-controlled alongside the prompt / tools / hooks it protects.
Crucially the cartridge *loader* stays evals-agnostic (see
``test_cartridge_package_layout.py``: ``looplet.evals`` is a FORBIDDEN
dep of the cartridge package); the read/write of the eval slot lives in
``looplet.evals`` instead. Graders (``eval_*``) and the ``looplet eval``
runner are later slices — this file pins only the case-data slot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet import (
    DefaultState,
    LoopConfig,
    cartridge_to_preset,
    preset_to_cartridge,
)
from looplet.evals import (
    CARTRIDGE_CASES_SUBPATH,
    EvalCase,
    load_cartridge_cases,
    save_cartridge_cases,
)
from looplet.presets import AgentPreset
from looplet.tools import BaseToolRegistry, ToolSpec


def _done_execute(*, answer: str = "") -> dict:
    return {"status": "completed", "answer": answer}


def _build_cartridge(tmp_path: Path) -> Path:
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="done",
            description="Submit final answer.",
            parameters={"answer": "str"},
            execute=_done_execute,
        )
    )
    preset = AgentPreset(
        config=LoopConfig(max_steps=6, system_prompt="case-carrying agent"),
        hooks=[],
        tools=registry,
        state=DefaultState(max_steps=6),
    )
    return preset_to_cartridge(preset, tmp_path / "ws").path


def _sample_cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="001_floor_div",
            task={"goal": "safe_div(-7, 2) == -4", "expr": "-7 // 2"},
            expected={"tests_passing": True},
            marks=["regression"],
            notes="Floor division on negatives, not truncation.",
        ),
        EvalCase(
            id="002_mode_tie",
            task={"goal": "mode([1,1,2,2]) == 1"},
            expected={"tests_passing": True},
        ),
    ]


def test_cases_written_under_cartridge_eval_slot(tmp_path: Path) -> None:
    cartridge = _build_cartridge(tmp_path)
    save_cartridge_cases(cartridge, _sample_cases())

    cases_dir = cartridge / CARTRIDGE_CASES_SUBPATH
    assert cases_dir.is_dir()
    assert (cases_dir / "001_floor_div.json").is_file()
    assert (cases_dir / "002_mode_tie.json").is_file()


def test_cartridge_cases_round_trip_lossless(tmp_path: Path) -> None:
    cartridge = _build_cartridge(tmp_path)
    original = _sample_cases()
    save_cartridge_cases(cartridge, original)

    reloaded = load_cartridge_cases(cartridge)
    assert [c.id for c in reloaded] == ["001_floor_div", "002_mode_tie"]
    by_id = {c.id: c for c in reloaded}
    for c in original:
        assert by_id[c.id].to_dict() == c.to_dict()


def test_missing_eval_slot_returns_empty_not_error(tmp_path: Path) -> None:
    cartridge = _build_cartridge(tmp_path)
    assert not (cartridge / "evals").exists()
    assert load_cartridge_cases(cartridge) == []


def test_hand_authored_cases_are_discovered(tmp_path: Path) -> None:
    # The corpus grows by dropping JSON in by hand (no Python) — the
    # whole point of "cases live as data".
    cartridge = _build_cartridge(tmp_path)
    cases_dir = cartridge / CARTRIDGE_CASES_SUBPATH
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "003_added_by_hand.json").write_text(
        '{"id": "003_added_by_hand", "task": {"goal": "x"}, "expected": {"tests_passing": true}}'
    )
    reloaded = load_cartridge_cases(cartridge)
    assert [c.id for c in reloaded] == ["003_added_by_hand"]


def test_jsonl_corpus_is_discovered(tmp_path: Path) -> None:
    cartridge = _build_cartridge(tmp_path)
    cases_dir = cartridge / CARTRIDGE_CASES_SUBPATH
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "corpus.jsonl").write_text(
        '{"id": "a", "task": {"goal": "1"}}\n{"id": "b", "task": {"goal": "2"}}\n'
    )
    reloaded = load_cartridge_cases(cartridge)
    assert sorted(c.id for c in reloaded) == ["a", "b"]


def test_malformed_case_is_loud(tmp_path: Path) -> None:
    cartridge = _build_cartridge(tmp_path)
    cases_dir = cartridge / CARTRIDGE_CASES_SUBPATH
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "broken.json").write_text("{ not valid json ]")
    with pytest.raises(ValueError):
        load_cartridge_cases(cartridge)


def test_eval_slot_survives_cartridge_reserialize(tmp_path: Path) -> None:
    # preset_to_cartridge does NOT manage (or clobber) the eval slot, so
    # hand-curated cases survive a re-serialise of the agent definition.
    cartridge = _build_cartridge(tmp_path)
    save_cartridge_cases(cartridge, _sample_cases())

    # Re-serialise the agent definition over the same dir.
    reloaded_preset = cartridge_to_preset(cartridge)
    preset_to_cartridge(reloaded_preset, cartridge, overwrite=True)

    # Evals are untouched.
    assert sorted(c.id for c in load_cartridge_cases(cartridge)) == [
        "001_floor_div",
        "002_mode_tie",
    ]
