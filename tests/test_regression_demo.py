"""The public failure-to-regression demo must stay executable."""

from __future__ import annotations

import json
from pathlib import Path

from examples.regression_demo import run_demo


def test_regression_demo_replays_same_decisions_to_green(tmp_path: Path) -> None:
    result = run_demo(tmp_path / "proof")

    assert result.v1_score == 0.0
    assert result.v2_score == 1.0
    assert result.v1_report["profit"] == 200
    assert result.v2_report["profit"] == 40
    assert result.v1_tools == ("publish_report", "done")
    assert result.v2_tools == result.v1_tools
    assert '"profit": revenue + cost' in result.diff
    assert '"profit": revenue - cost' in result.diff
    removed = [
        line
        for line in result.diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line
        for line in result.diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert removed == ['-        "profit": revenue + cost,']
    assert added == ['+        "profit": revenue - cost,']

    v1 = result.output_dir / "runs" / "v1"
    v2 = result.output_dir / "runs" / "v2"
    assert (v1 / "manifest.jsonl").is_file()
    assert (v1 / "call_00_response.txt").is_file()
    assert json.loads((v1 / "artifacts.json").read_text())["observed_profit"] == 200
    assert json.loads((v2 / "artifacts.json").read_text())["observed_profit"] == 40
    assert json.loads((v1 / "expected.json").read_text()) == {"profit": 40}
    assert "expected" not in json.loads((v1 / "trajectory.json").read_text())["task"]
    assert '"expected"' not in (v1 / "call_00_prompt.txt").read_text()
