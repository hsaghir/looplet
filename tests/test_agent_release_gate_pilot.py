"""Reproducibility test for the agent release-gate direction pilot."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.agent_release_gate_pilot.run_pilot import run

pytestmark = pytest.mark.smoke


def test_outcome_gate_catches_seeded_regressions_without_false_rejections(tmp_path: Path):
    results = run(tmp_path / "pilot")

    safe = [result for result in results if result.expected_outcome_pass]
    regressions = [result for result in results if not result.expected_outcome_pass]

    assert len(safe) == 2
    assert len(regressions) == 5
    assert all(result.trace_gate_pass for result in regressions)
    assert all(not result.outcome_gate_pass for result in regressions)
    assert all(result.outcome_gate_pass for result in safe)
