"""Cross-runtime conformance: ``tinyloop`` (the second runtime)
loads the same fixtures and produces the same spec-pinned summary
as the reference loader.

This is the core portability evidence behind SPEC.md: the cartridge
declares no runtime, so any conformant loader written from scratch
should produce equivalent output for the spec-pinned subset. If this
test fails, either ``tinyloop`` has drifted or the spec-pinned
summary in ``test_conformance._summarise_preset`` has widened
without updating ``tinyloop.conformance_summary``.

Only fixture ``01_minimal`` is exercised, deliberately:
``tinyloop`` does not implement permissions, model binding, output
schemas, or memory. The other fixtures exercise those slots and
are out of ``tinyloop``'s declared scope.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TINYLOOP_PATH = REPO_ROOT / "examples" / "alt_runtime" / "tinyloop.py"
FIXTURE = REPO_ROOT / "tests" / "conformance" / "fixtures" / "01_minimal"
V2_FIXTURE = REPO_ROOT / "tests" / "conformance" / "fixtures" / "04_v2_tier_split"


@pytest.fixture(scope="module")
def tinyloop_module():
    """Import tinyloop.py without polluting the looplet package."""
    spec = importlib.util.spec_from_file_location("tinyloop_alt", TINYLOOP_PATH)
    assert spec and spec.loader, f"failed to load {TINYLOOP_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["tinyloop_alt"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("tinyloop_alt", None)


def test_tinyloop_loads_minimal_fixture(tinyloop_module) -> None:
    cart = tinyloop_module.load_cartridge(FIXTURE / "cartridge")
    assert cart.name == "minimal"
    assert cart.schema_version == 2
    assert "done" in cart.tools


def test_tinyloop_summary_matches_expected_for_minimal_fixture(tinyloop_module) -> None:
    """Cross-runtime spec compliance: tinyloop's summary == expected.json."""
    cart = tinyloop_module.load_cartridge(FIXTURE / "cartridge")
    summary = tinyloop_module.conformance_summary(cart)
    expected = json.loads((FIXTURE / "expected.json").read_text())
    assert summary == expected, (
        "tinyloop produced a different conformance summary than the reference "
        "loader for fixture 01_minimal. Either tinyloop drifted, or the "
        "spec-pinned summary subset widened without updating tinyloop."
    )


def test_tinyloop_runs_scripted_done_call(tinyloop_module) -> None:
    cart = tinyloop_module.load_cartridge(FIXTURE / "cartridge")
    steps = tinyloop_module.run_scripted(
        cart,
        [{"tool": "done", "args": {"summary": "ok"}}],
    )
    assert len(steps) == 1
    assert steps[0]["tool"] == "done"
    assert steps[0]["error"] is None
    assert steps[0]["result"] == {"summary": "ok"}


def test_tinyloop_rejects_missing_manifest(tinyloop_module, tmp_path) -> None:
    """SPEC clause 7: structured error naming the offending file path."""
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        tinyloop_module.load_cartridge(empty)
    msg = str(exc_info.value)
    assert "workspace.json" in msg or "cartridge.json" in msg


def test_tinyloop_accepts_workspace_json_legacy_alias(tinyloop_module, tmp_path) -> None:
    """The legacy workspace.json manifest filename still loads.

    The fixtures now ship cartridge.json (canonical name); this test
    pins backwards compatibility for the historical workspace.json
    spelling so external cartridges built before the rename still
    load on tinyloop.
    """
    import shutil

    dst = tmp_path / "minimal.cartridge"
    shutil.copytree(FIXTURE / "cartridge", dst)
    (dst / "cartridge.json").rename(dst / "workspace.json")
    cart = tinyloop_module.load_cartridge(dst)
    assert cart.name == "minimal"


# ── v2 cross-runtime portability ───────────────────────────────────


def test_tinyloop_loads_v2_tier_split_fixture(tinyloop_module) -> None:
    """A v2 cartridge with config.yaml + runtime.yaml loads on tinyloop.

    Proves the spec-v2 tier split is genuinely cross-runtime: tinyloop
    (a different loader written from scratch) merges runtime.yaml on
    top of config.yaml and produces the same flat conformance summary
    as the reference loader does for the equivalent v1 cartridge.
    """
    cart = tinyloop_module.load_cartridge(V2_FIXTURE / "cartridge")
    assert cart.name == "v2_tier_split"
    assert cart.schema_version == 2
    summary = tinyloop_module.conformance_summary(cart)
    expected = json.loads((V2_FIXTURE / "expected.json").read_text())
    assert summary == expected, (
        "tinyloop produced a different conformance summary than the "
        "v2 reference loader. Either tinyloop has not honoured the "
        "config.yaml + runtime.yaml tier split, or the spec-pinned "
        "summary has widened without updating tinyloop."
    )
    # The runtime knobs came from runtime.yaml, not config.yaml.
    assert cart.config.get("max_tokens") == 2000
    assert cart.config.get("temperature") == 0.2


# ── Minimal-conforming: all v1.0 declarative slots ─────────────────


@pytest.mark.parametrize(
    "fixture_name",
    [
        "02_permissions",
        "03_model_and_output_schema",
        "04_long_term_memory",
    ],
)
def test_tinyloop_summary_matches_for_declarative_slot_fixtures(
    tinyloop_module, fixture_name: str
) -> None:
    """tinyloop is now minimal-conforming for all v1.0 declarative slots.

    Cross-runtime evidence that ``permissions``, ``model``,
    ``output_schema`` and ``memory_sources`` are part of the
    cartridge contract — not loader-specific decoration. If a fixture
    is added or its expected.json widens, tinyloop must keep up.
    """
    fixture_dir = REPO_ROOT / "tests" / "conformance" / "fixtures" / fixture_name
    cart = tinyloop_module.load_cartridge(fixture_dir / "cartridge")
    summary = tinyloop_module.conformance_summary(cart)
    expected = json.loads((fixture_dir / "expected.json").read_text())
    assert summary == expected, (
        f"tinyloop summary for fixture {fixture_name!r} drifted from the "
        f"reference loader. Expected:\n{json.dumps(expected, indent=2)}\n"
        f"Actual:\n{json.dumps(summary, indent=2)}"
    )


# ── Spec v2 hard-rejections (cross-runtime) ────────────────────────


@pytest.mark.parametrize(
    "fixture_name, expected_substr",
    [
        ("06_v2_setup_py_rejected", "setup.py"),
        ("07_v2_magic_files_rejected", "prompts/briefing.md"),
    ],
)
def test_tinyloop_rejects_v2_forbidden_files(
    tinyloop_module, fixture_name: str, expected_substr: str
) -> None:
    """tinyloop also fails-loud on v2 forbidden files.

    Locks the v2 hard-rejection contract as cross-runtime, not a
    quirk of the reference loader. Both runtimes raise their own
    ``CartridgeSerializationError`` with a message naming the
    offending file.
    """
    fixture_dir = REPO_ROOT / "tests" / "conformance" / "fixtures" / fixture_name
    with pytest.raises(tinyloop_module.CartridgeSerializationError) as excinfo:
        tinyloop_module.load_cartridge(fixture_dir / "cartridge")
    assert expected_substr in str(excinfo.value), (
        f"tinyloop rejection message for {fixture_name!r} did not contain "
        f"{expected_substr!r}; got: {excinfo.value}"
    )
