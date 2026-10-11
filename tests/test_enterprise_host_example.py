"""Dogfood the small enterprise-host reference integration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from examples.enterprise_host import __main__ as host
from examples.enterprise_host import run_host


def test_enterprise_host_runs_coder_cartridge(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.enterprise_host",
            "--cartridge",
            "tests/fixtures/coder_skill_bundle",
            "--workspace",
            str(workspace),
            "--task",
            "Greet the requested people and finish.",
            "--scripted",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed"
    assert summary["termination_reason"] == "done"
    assert summary["steps"] == 5
    assert summary["run_envelope"]["tenant_id"] == "reference-tenant"
    assert summary["policy_decisions"]
    assert summary["diagnostics"]["completed"] is True
    assert summary["diagnostics"]["run_id"] == summary["run_envelope"]["run_id"]
    assert summary["diagnostics"]["cost_usd"] is None
    assert summary["configuration"]["system_prompt"]["redacted"] is True
    assert "done" in summary["terminal_outcomes"]
    assert Path(summary["trace_dir"]).joinpath("trajectory.json").is_file()
    assert list(Path(summary["checkpoint_dir"]).glob("*.json"))


@pytest.mark.parametrize("scripted", [True, False])
def test_enterprise_host_api_returns_audit_summary(tmp_path: Path, monkeypatch, scripted) -> None:
    bundle = host.load_skill_bundle("tests/fixtures/coder_skill_bundle")
    original_build = bundle.build
    built = []
    closed = []

    def build(runtime):
        preset = original_build(runtime)
        built.append(preset)
        original_close = preset.close

        def close():
            closed.append(preset)
            original_close()

        preset.close = close
        return preset

    monkeypatch.setattr(host, "load_skill_bundle", lambda _path: replace(bundle, build=build))
    if not scripted:
        with pytest.raises(ValueError, match="requires --scripted"):
            run_host(
                cartridge=Path("tests/fixtures/coder_skill_bundle"),
                workspace=tmp_path / "workspace",
                task="Finish.",
                run_id="unsupported-run",
                tenant_id="tenant-api",
                actor_id="actor-api",
                scripted=False,
            )
        assert len(built) == 1
        assert closed == built
        return

    summary = run_host(
        cartridge=Path("tests/fixtures/coder_skill_bundle"),
        workspace=tmp_path / "workspace",
        task="Greet the requested people and finish.",
        run_id="api-run",
        tenant_id="tenant-api",
        actor_id="actor-api",
        scripted=True,
    )

    assert summary["status"] == "completed"
    assert summary["run_envelope"]["run_id"] == "api-run"
    assert summary["run_envelope"]["tenant_id"] == "tenant-api"
    assert summary["policy_decisions"]
    assert summary["diagnostics"]["run_id"] == "api-run"
    assert summary["diagnostics"]["completed"] is True
    assert len(built) == 1
    assert closed == built
