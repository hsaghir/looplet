"""Cartridge loading must not leak subprocess-backed resources on errors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from looplet import cartridge_to_preset, minimal_preset
from looplet.cartridge import CartridgeSerializationError
from looplet.lep import LEPHookAdapter


def _write_late_failure_cartridge(root: Path) -> None:
    root.mkdir()
    (root / "cartridge.json").write_text(
        json.dumps({"name": "cleanup", "schema_version": 2}),
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "max_steps: 3\n"
        "done_tool: done\n"
        "state_services:\n"
        "  cache:\n"
        '    command: "fake-state"\n'
        "mcp_servers:\n"
        "  tools:\n"
        '    command: "fake-mcp"\n',
        encoding="utf-8",
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("test", encoding="utf-8")
    # This is validated after services and MCP adapters start.
    (prompts / "briefing.md").write_text("undeclared", encoding="utf-8")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: Finish.\nparameters: {}\n",
        encoding="utf-8",
    )
    (done / "execute.py").write_text(
        "def execute(ctx):\n    return {}\n",
        encoding="utf-8",
    )


def test_late_load_error_closes_all_spawned_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[str] = []

    class FakeGateway:
        @classmethod
        def start(cls):
            return cls()

        def close(self) -> None:
            closed.append("gateway")

    class FakeStateHandle:
        def __init__(self, name: str) -> None:
            self.name = name
            self.client = object()
            self.socket_path = f"/tmp/{name}.sock"

        @classmethod
        def spawn(cls, command, *, name, timeout_s=10.0, env=None):
            del command, timeout_s, env
            return cls(name)

        def close(self) -> None:
            closed.append("state")
            env_name = f"LOOPLET_STATE_{self.name.upper()}"
            if os.environ.get(env_name) == self.socket_path:
                os.environ.pop(env_name, None)

        def export_env(self) -> None:
            os.environ[f"LOOPLET_STATE_{self.name.upper()}"] = self.socket_path

    class FakeAdapter:
        def __init__(self, command, *, env=None, timeout=30.0) -> None:
            del command, env, timeout

        def tools(self):
            return []

        def close(self) -> None:
            closed.append("mcp")

    monkeypatch.setattr("looplet.model_gateway.ModelGatewayHandle", FakeGateway)
    monkeypatch.setattr("looplet.state_service.StateServiceHandle", FakeStateHandle)
    monkeypatch.setattr("looplet.mcp.MCPToolAdapter", FakeAdapter)

    cartridge = tmp_path / "cleanup.cartridge"
    _write_late_failure_cartridge(cartridge)

    with pytest.raises(CartridgeSerializationError, match="prompts/briefing.md"):
        cartridge_to_preset(cartridge, strict=True)

    assert closed == ["mcp", "state", "gateway"]
    assert "LOOPLET_STATE_CACHE" not in os.environ


def test_preset_close_closes_lep_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    hook = LEPHookAdapter(["unused"])
    monkeypatch.setattr(hook, "close", lambda: closed.append(True))
    preset = minimal_preset()
    preset.hooks.append(hook)

    preset.close()

    assert closed == [True]


def _write_resource_cartridge(root: Path, *, late_failure: bool = False) -> None:
    root.mkdir()
    (root / "cartridge.json").write_text(
        json.dumps({"name": "resource-lifecycle", "schema_version": 2}),
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "max_steps: 3\ndone_tool: done\n",
        encoding="utf-8",
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("test", encoding="utf-8")
    if late_failure:
        (prompts / "briefing.md").write_text("undeclared", encoding="utf-8")
    resources = root / "resources"
    resources.mkdir()
    (resources / "tracker.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "class Tracker:\n"
        "    def __init__(self, marker):\n"
        "        self.marker = marker\n"
        "\n"
        "    def close(self):\n"
        "        Path(self.marker).write_text('closed', encoding='utf-8')\n"
        "\n"
        "def build(runtime):\n"
        "    return Tracker(runtime['resource_marker'])\n",
        encoding="utf-8",
    )
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: Finish.\nparameters: {}\n",
        encoding="utf-8",
    )
    (done / "execute.py").write_text(
        "def execute(ctx):\n    return {}\n",
        encoding="utf-8",
    )


def test_cartridge_preset_close_closes_owned_resources(tmp_path: Path) -> None:
    cartridge = tmp_path / "resource.cartridge"
    marker = tmp_path / "closed.txt"
    _write_resource_cartridge(cartridge)

    preset = cartridge_to_preset(
        cartridge,
        runtime={"resource_marker": str(marker)},
        strict=True,
    )
    assert preset.resources["tracker"].marker == str(marker)

    preset.close()
    preset.close()

    assert marker.read_text(encoding="utf-8") == "closed"
    assert preset.owned_resources == []


def test_late_cartridge_load_failure_closes_owned_resources(tmp_path: Path) -> None:
    cartridge = tmp_path / "resource-late-failure.cartridge"
    marker = tmp_path / "closed.txt"
    _write_resource_cartridge(cartridge, late_failure=True)

    with pytest.raises(CartridgeSerializationError, match="prompts/briefing.md"):
        cartridge_to_preset(
            cartridge,
            runtime={"resource_marker": str(marker)},
            strict=True,
        )

    assert marker.read_text(encoding="utf-8") == "closed"
