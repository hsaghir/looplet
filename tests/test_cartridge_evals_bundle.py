"""The full eval bundle (cases + graders + collectors) ships in one
cartridge slot and is discovered by a single ``load_cartridge_evals``.

Extends the case-only round-trip (``test_cartridge_evals_roundtrip.py``)
with the grader + collector discovery that completes "evals ship with
the agent version": ``evals/eval_*.py`` (graders), ``evals/collect_*.py``
(collectors), ``evals/cases/*.json`` (cases) - all under one ``evals/``
directory, all returned by one call. This is the slice that removes the
hand-wiring the dogfood exposed (importing the collector by path).

Like the case slot, this lives entirely in ``looplet.evals`` - the
cartridge package never learns about evals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet import (
    CartridgeEvals,
    EvalCase,
    discover_collectors,
    load_cartridge_evals,
    load_evals,
    save_cartridge_cases,
)

GRADERS_PY = """\
def eval_completed(ctx):
    return "done" in [getattr(s.tool_call, "tool", "") for s in ctx.steps]

def eval_tests_pass(ctx):
    return bool(ctx.artifacts.get("tests_passing", False))

def _helper_not_a_grader(ctx):
    return 1.0
"""

COLLECTORS_PY = """\
def collect_static(state):
    # No runtime needed - fixed outcome.
    return {"saw_state": state is not None}

def collect_with_runtime(state, runtime):
    # Runtime-parameterised: reads a value bound at discovery time.
    return {"workspace": runtime.get("workspace_dir")}
"""


def _seed_eval_bundle(cartridge: Path) -> None:
    evals = cartridge / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    (evals / "eval_correctness.py").write_text(GRADERS_PY)
    (evals / "collect_outcome.py").write_text(COLLECTORS_PY)
    save_cartridge_cases(
        cartridge,
        [
            EvalCase(id="c1", task={"goal": "x"}, expected={"tests_passing": True}),
            EvalCase(id="c2", task={"goal": "y"}),
        ],
    )


def test_load_cartridge_evals_returns_full_bundle(tmp_path: Path) -> None:
    cartridge = tmp_path / "ws"
    cartridge.mkdir()
    _seed_eval_bundle(cartridge)

    bundle = load_cartridge_evals(cartridge)
    assert isinstance(bundle, CartridgeEvals)

    # Cases (data).
    assert sorted(c.id for c in bundle.cases) == ["c1", "c2"]
    # Graders: eval_* only, NOT the _helper.
    assert sorted(g.__name__ for g in bundle.graders) == ["eval_completed", "eval_tests_pass"]
    # Collectors: collect_* only.
    # (names survive for the non-bound one; the runtime-bound one is a partial)
    collector_names = {getattr(c, "__name__", "partial") for c in bundle.collectors}
    assert "collect_static" in collector_names


def test_unpacks_as_cases_graders_collectors(tmp_path: Path) -> None:
    cartridge = tmp_path / "ws"
    cartridge.mkdir()
    _seed_eval_bundle(cartridge)

    cases, graders, collectors = load_cartridge_evals(cartridge)
    assert len(cases) == 2
    assert len(graders) == 2
    assert len(collectors) == 2


def test_runtime_is_bound_into_collectors(tmp_path: Path) -> None:
    cartridge = tmp_path / "ws"
    cartridge.mkdir()
    _seed_eval_bundle(cartridge)

    bundle = load_cartridge_evals(cartridge, runtime={"workspace_dir": "/sandbox/42"})

    # Every collector is now uniformly (state) -> dict, ready for EvalHook.
    merged: dict = {}
    for collector in bundle.collectors:
        merged.update(collector(state=object()))
    # The runtime-parameterised collector saw the bound runtime value.
    assert merged["workspace"] == "/sandbox/42"
    # The plain collector still ran.
    assert merged["saw_state"] is True


def test_graders_are_case_agnostic_reused_across_cases(tmp_path: Path) -> None:
    # The whole point of eval_* discovery: one grader, many cases (N×M).
    cartridge = tmp_path / "ws"
    cartridge.mkdir()
    _seed_eval_bundle(cartridge)

    bundle = load_cartridge_evals(cartridge)
    # graders carry no per-case state - the same callables apply to every case.
    assert all(callable(g) for g in bundle.graders)
    assert len(bundle.graders) == 2
    assert len(bundle.cases) == 2


def test_missing_evals_dir_yields_empty_bundle(tmp_path: Path) -> None:
    cartridge = tmp_path / "ws"
    cartridge.mkdir()
    bundle = load_cartridge_evals(cartridge)
    assert bundle == CartridgeEvals(cases=[], graders=[], collectors=[])


def test_discover_collectors_standalone(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "collect_outcome.py").write_text(COLLECTORS_PY)

    # Without runtime, the runtime-param collector is still returned but
    # binding an empty runtime dict is harmless.
    collectors = discover_collectors(evals, runtime={"workspace_dir": "/x"})
    assert len(collectors) == 2
    merged: dict = {}
    for c in collectors:
        merged.update(c(state=object()))
    assert merged["workspace"] == "/x"


def test_collectors_without_runtime_default_to_empty(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "collect_outcome.py").write_text(COLLECTORS_PY)

    # No runtime supplied → runtime-param collector gets {} and returns None value.
    collectors = discover_collectors(evals)
    merged: dict = {}
    for c in collectors:
        merged.update(c(state=object()))
    assert merged["workspace"] is None  # runtime.get("workspace_dir") on {}
    assert merged["saw_state"] is True


def test_generic_bundle_loader_matches_cartridge_definitions(tmp_path: Path) -> None:
    _seed_eval_bundle(tmp_path)
    runtime = {"workspace_dir": "/sandbox/42"}
    bundle = load_evals(tmp_path / "evals", runtime=runtime)
    compatible = load_cartridge_evals(tmp_path, runtime=runtime, strict=True)
    assert isinstance(bundle, CartridgeEvals)
    assert bundle.cases == compatible.cases
    assert [grader.__name__ for grader in bundle.graders] == [
        grader.__name__ for grader in compatible.graders
    ]
    assert len(bundle.collectors) == len(compatible.collectors) == 2
    assert bundle.collectors[1](object())["workspace"] == "/sandbox/42"


def test_generic_bundle_loader_rejects_missing_directory_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Eval definitions not found"):
        load_evals(tmp_path / "missing")
    assert load_evals(tmp_path / "missing", strict=False) == CartridgeEvals([], [], [])
    assert load_cartridge_evals(tmp_path, strict=True) == CartridgeEvals([], [], [])


@pytest.mark.parametrize("filename", ["eval_broken.py", "collect_broken.py"])
def test_generic_bundle_loader_rejects_broken_definition_imports(tmp_path: Path, filename: str):
    (tmp_path / filename).write_text("raise RuntimeError('broken definition')\n")
    with pytest.raises(RuntimeError, match="broken definition"):
        load_evals(tmp_path)


def test_generic_bundle_loader_preserves_partial_definition_bundles(tmp_path: Path):
    (tmp_path / "eval_answer.py").write_text("def eval_answer(ctx):\n    return True\n")
    bundle = load_evals(tmp_path)
    assert bundle.cases == []
    assert bundle.collectors == []
    assert [grader.__name__ for grader in bundle.graders] == ["eval_answer"]


@pytest.mark.parametrize("as_directory", [False, True])
def test_one_python_contract_keeps_graders_and_collectors_together(tmp_path, as_directory):
    source = tmp_path / "eval_contract.py"
    source.write_text(
        "observed = []\n\n"
        "def collect_value(state, runtime):\n"
        "    observed.append(runtime['value'])\n"
        "    return {'value': runtime['value']}\n\n"
        "def eval_value(ctx):\n"
        "    return observed == [42]\n"
    )
    bundle = load_evals(tmp_path if as_directory else source, runtime={"value": 42})
    assert bundle.cases == []
    assert [grader.__name__ for grader in bundle.graders] == ["eval_value"]
    assert len(bundle.collectors) == 1
    assert bundle.collectors[0](object()) == {"value": 42}
    assert bundle.graders[0](object()) is True


def test_definition_roles_preserve_exported_names_when_callables_are_renamed(tmp_path):
    source = tmp_path / "eval_contract.py"
    source.write_text(
        "def eval_answer(ctx):\n    return True\n\n"
        "eval_answer.__name__ = 'grade'\n\n"
        "def collect_answer(state):\n    return {'answer': 42}\n\n"
        "collect_answer.__name__ = 'observe'\n"
    )
    definitions = load_evals(source)
    assert [grader.__name__ for grader in definitions.graders] == ["grade"]
    assert [collector.__name__ for collector in definitions.collectors] == ["observe"]
