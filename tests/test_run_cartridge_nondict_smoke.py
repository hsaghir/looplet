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
from pathlib import Path

import pytest

from looplet.cli import factory_commands
from looplet.testing import MockLLMBackend

pytestmark = pytest.mark.smoke

_MCP_DEMO = Path(__file__).resolve().parents[1] / "examples" / "mcp_demo.cartridge"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        workspace=_MCP_DEMO,
        task="What is 12345 + 67890?",
        max_steps=5,
        quiet=False,
        pretty=False,
    )


@pytest.mark.skipif(not _MCP_DEMO.is_dir(), reason="mcp_demo cartridge not present")
def test_run_cartridge_survives_int_tool_result(monkeypatch, capsys):
    """The non-quiet printer tolerates an int ToolResult.data without crashing."""
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

    rc = factory_commands.cmd_run_workspace(_args())

    assert rc == 0
    out = capsys.readouterr().out
    # The add step printed without raising, and the run reached `done`.
    assert "add(" in out
    assert "80235" in out
