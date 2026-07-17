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

from looplet import (
    CartridgeEvals,
    EvalCase,
    discover_collectors,
    load_cartridge_evals,
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
