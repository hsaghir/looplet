"""Regression: `looplet run-cartridge` must not crash when a tool returns a
non-dict ``ToolResult.data`` (e.g. an out-of-process MCP tool that returns an
``int``).

Before the fix, the per-step printer in ``cmd_run_workspace`` did
``tool_result.data.get("error")`` unconditionally, which raised
``AttributeError: 'int' object has no attribute 'get'`` for the bundled
``mcp_demo`` cartridge whose ``add`` MCP tool returns an integer sum.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from looplet.__main__ import main
from looplet.cli import factory_commands
from looplet.testing import MockLLMBackend

pytestmark = pytest.mark.smoke

_MCP_DEMO = Path(__file__).resolve().parents[1] / "examples" / "mcp_demo.cartridge"


def _args(
    *,
    task: str = "What is 12345 + 67890?",
    project_root: Path | None = None,
    trace_dir: Path | None = None,
    no_trace: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        workspace=_MCP_DEMO,
        task=task,
        max_steps=5,
        project_root=project_root,
        trace_dir=trace_dir,
        no_trace=no_trace,
        quiet=False,
        pretty=False,
    )


def _patch_runtime(monkeypatch) -> None:
    responses = [
        '{"tool":"add","args":{"a":12345,"b":67890},"reasoning":"sum"}',
        '{"tool":"done","args":{"total":80235},"reasoning":"report"}',
    ]
    monkeypatch.setenv("OPENAI_MODEL", "mock-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(factory_commands, "_check_env", lambda: 0)
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: MockLLMBackend(responses=responses, cycle=False),
    )


@pytest.mark.skipif(not _MCP_DEMO.is_dir(), reason="mcp_demo cartridge not present")
def test_run_cartridge_survives_int_tool_result(monkeypatch, capsys, tmp_path):
    """The non-quiet printer tolerates an int ToolResult.data without crashing."""
    _patch_runtime(monkeypatch)

    trace_dir = tmp_path / "trace"
    rc = factory_commands.cmd_run_workspace(_args(trace_dir=trace_dir))

    assert rc == 0
    out = capsys.readouterr().out
    # The add step printed without raising, and the run reached `done`.
    assert "add(" in out
    assert "80235" in out
    assert f"Trace: {trace_dir}" in out
    assert (trace_dir / "trajectory.json").is_file()
    assert (trace_dir / "manifest.jsonl").is_file()
    trajectory = json.loads((trace_dir / "trajectory.json").read_text())
    assert len(trajectory["steps"]) == 2
    assert main(["show", str(trace_dir)]) == 0


@pytest.mark.skipif(not _MCP_DEMO.is_dir(), reason="mcp_demo cartridge not present")
def test_run_cartridge_can_disable_trace(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    trace_dir = tmp_path / "trace"

    rc = factory_commands.cmd_run_workspace(_args(trace_dir=trace_dir, no_trace=True))

    assert rc == 0
    assert not trace_dir.exists()


@pytest.mark.skipif(not _MCP_DEMO.is_dir(), reason="mcp_demo cartridge not present")
def test_run_cartridge_defaults_trace_to_project_root(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)

    rc = factory_commands.cmd_run_workspace(_args(project_root=tmp_path))

    assert rc == 0
    traces = list((tmp_path / ".looplet" / "traces").glob("mcp_demo-*"))
    assert len(traces) == 1
    assert (traces[0] / "trajectory.json").is_file()


@pytest.mark.skipif(not _MCP_DEMO.is_dir(), reason="mcp_demo cartridge not present")
def test_run_cartridge_reads_task_from_stdin(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    task = "Review this exact piped input.\nSecond line stays intact.\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(task))
    trace_dir = tmp_path / "trace"

    rc = factory_commands.cmd_run_workspace(_args(task="-", trace_dir=trace_dir))

    assert rc == 0
    trajectory = json.loads((trace_dir / "trajectory.json").read_text())
    assert trajectory["task"]["goal"] == task


@pytest.mark.skipif(not _MCP_DEMO.is_dir(), reason="mcp_demo cartridge not present")
def test_run_cartridge_rejects_empty_stdin(monkeypatch, capsys):
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: pytest.fail("empty stdin must fail before backend construction"),
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(" \n"))

    rc = factory_commands.cmd_run_workspace(_args(task="-", no_trace=True))

    assert rc == 1
    assert "task read from stdin is empty" in capsys.readouterr().err
