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
    assert cart.schema_version == 1
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
