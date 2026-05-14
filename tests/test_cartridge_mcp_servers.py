"""Tests for the cartridge ``mcp_servers:`` block in config.yaml.

The loader spawns one :class:`looplet.mcp.MCPToolAdapter` per declared
server, registers each server's discovered tools into the cartridge's
tool registry, and stashes the adapters on ``preset.mcp_adapters`` so
the caller can ``preset.close()`` them when done.

These tests monkeypatch ``MCPToolAdapter`` to avoid spawning real
subprocesses; the adapter itself has its own smoke tests in
``tests/test_mcp_adapter_smoke.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import cartridge_to_preset
from looplet.cartridge import CartridgeSerializationError
from looplet.tools import ToolSpec


class _FakeAdapter:
    """Pretend MCPToolAdapter — records its construction args and
    exposes a fixed list of ``ToolSpec`` objects."""

    instances: list["_FakeAdapter"] = []

    def __init__(self, command: str, *, env=None, timeout: float = 30.0) -> None:
        self.command = command
        self.env = env
        self.timeout = timeout
        self.closed = False
        type(self).instances.append(self)

    def tools(self) -> list[ToolSpec]:
        # Two pretend tools per adapter; the cartridge's allow-list
        # filtering is exercised by the `tools:` field test below.
        return [
            ToolSpec(
                name=f"{self.command.split()[0]}_read",
                description="fake read",
                parameters={"path": "string"},
                execute=lambda *, path: {"path": path},
            ),
            ToolSpec(
                name=f"{self.command.split()[0]}_write",
                description="fake write",
                parameters={"path": "string", "content": "string"},
                execute=lambda *, path, content: {"path": path, "len": len(content)},
            ),
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    _FakeAdapter.instances = []
    yield
    _FakeAdapter.instances = []


def _write_minimal_cartridge(root: Path, *, mcp_servers_yaml: str) -> None:
    root.mkdir()
    (root / "cartridge.json").write_text(json.dumps({"name": "mcp_test", "schema_version": 2}))
    (root / "config.yaml").write_text(f"max_steps: 5\ndone_tool: done\n{mcp_servers_yaml}")
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("test\n")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )


def test_mcp_servers_registers_tools_from_each_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tool returned by an adapter ends up in the registry."""
    monkeypatch.setattr("looplet.mcp.MCPToolAdapter", _FakeAdapter)

    root = tmp_path / "x.cartridge"
    _write_minimal_cartridge(
        root,
        mcp_servers_yaml=(
            "mcp_servers:\n"
            "  fs:\n"
            '    command: "filesystem /tmp"\n'
            "  gh:\n"
            '    command: "github --repo demo/x"\n'
        ),
    )
    preset = cartridge_to_preset(str(root), strict=True)

    names = set(preset.tools.tool_names)
    assert "filesystem_read" in names
    assert "filesystem_write" in names
    assert "github_read" in names
    assert "github_write" in names
    assert "done" in names

    assert len(preset.mcp_adapters) == 2
    assert all(isinstance(a, _FakeAdapter) for a in preset.mcp_adapters)


def test_mcp_servers_tools_allowlist_filters_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tools:`` on a server entry restricts which discovered tools
    are registered. The adapter still discovers all of them; only the
    allow-listed ones reach the registry."""
    monkeypatch.setattr("looplet.mcp.MCPToolAdapter", _FakeAdapter)

    root = tmp_path / "x.cartridge"
    _write_minimal_cartridge(
        root,
        mcp_servers_yaml=(
            'mcp_servers:\n  fs:\n    command: "filesystem /tmp"\n    tools: [filesystem_read]\n'
        ),
    )
    preset = cartridge_to_preset(str(root), strict=True)
    names = set(preset.tools.tool_names)
    assert "filesystem_read" in names
    assert "filesystem_write" not in names


def test_preset_close_terminates_mcp_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``preset.close()`` and the context-manager protocol both close
    every spawned adapter exactly once."""
    monkeypatch.setattr("looplet.mcp.MCPToolAdapter", _FakeAdapter)

    root = tmp_path / "x.cartridge"
    _write_minimal_cartridge(
        root,
        mcp_servers_yaml=('mcp_servers:\n  fs:\n    command: "filesystem /tmp"\n'),
    )

    with cartridge_to_preset(str(root), strict=True) as preset:
        assert len(preset.mcp_adapters) == 1
        adapter = preset.mcp_adapters[0]
        assert adapter.closed is False
    assert adapter.closed is True
    # close() is idempotent.
    preset.close()


def test_mcp_servers_missing_command_raises_in_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("looplet.mcp.MCPToolAdapter", _FakeAdapter)

    root = tmp_path / "x.cartridge"
    _write_minimal_cartridge(
        root,
        mcp_servers_yaml=("mcp_servers:\n  bad:\n    timeout_s: 5\n"),
    )
    with pytest.raises(CartridgeSerializationError, match="command:"):
        cartridge_to_preset(str(root), strict=True)


def test_mcp_servers_failure_cleans_up_already_started_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the second server fails to start, the first one must be
    closed before the loader propagates the error — otherwise we
    leak subprocesses."""

    started: list[_FakeAdapter] = []

    class _MaybeFailing(_FakeAdapter):
        def __init__(self, command: str, *, env=None, timeout: float = 30.0) -> None:
            if "broken" in command:
                raise RuntimeError("simulated MCP startup failure")
            super().__init__(command, env=env, timeout=timeout)
            started.append(self)

    monkeypatch.setattr("looplet.mcp.MCPToolAdapter", _MaybeFailing)

    root = tmp_path / "x.cartridge"
    _write_minimal_cartridge(
        root,
        mcp_servers_yaml=(
            "mcp_servers:\n"
            "  ok:\n"
            '    command: "filesystem /tmp"\n'
            "  bad:\n"
            '    command: "broken-server"\n'
        ),
    )
    with pytest.raises(CartridgeSerializationError, match="failed to start"):
        cartridge_to_preset(str(root), strict=True)
    assert len(started) == 1
    assert started[0].closed is True
