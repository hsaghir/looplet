"""Dogfood the small enterprise-host reference integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
    assert Path(summary["trace_dir"]).joinpath("trajectory.json").is_file()
    assert list(Path(summary["checkpoint_dir"]).glob("*.json"))


def test_enterprise_host_api_returns_audit_summary(tmp_path: Path) -> None:
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
