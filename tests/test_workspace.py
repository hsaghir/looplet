"""Tests for the Composable Harness Workspace (CHW) round-trip.

Verifies:

* fresh-empty-directory write succeeds; non-empty fails without overwrite
* round-trip of a hand-built ``AgentPreset`` preserves config, system
  prompt, tools (with parameters + execute behaviour), and a hook with
  an opt-in ``to_config()`` method
* ``StaticMemorySource`` instances round-trip via ``memory/*.md``
* warnings are recorded for non-round-trippable config callables
  (``strict=False``) and raised under ``strict=True``
* layout discovery: the workspace.json metadata file is required
* ``Workspace.to_preset()`` materialises a runnable preset that the
  composable loop can execute end-to-end with a scripted MockLLM
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    Workspace,
    WorkspaceLayout,
    WorkspaceSerializationError,
    composable_loop,
    preset_to_workspace,
    workspace_to_preset,
)
from looplet.memory import StaticMemorySource
from looplet.presets import AgentPreset
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

# ── fixtures ────────────────────────────────────────────────────


def lookup_execute(*, key: str) -> dict:
    """Top-level execute so it round-trips through inspect.getsource."""
    return {"key": key, "value": {"x": 1, "y": 2}.get(key, "MISSING")}


def done_execute(*, answer: str) -> dict:
    return {"answer": answer}


class DemoCounter:
    """Hook with opt-in ``to_config()`` for round-trip kwargs."""

    def __init__(self, *, threshold: int = 3) -> None:
        self.threshold = threshold
        self.seen = 0

    def to_config(self) -> dict:
        return {"threshold": self.threshold}

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):  # noqa: D401
        self.seen += 1
        return None


def _build_demo_preset() -> AgentPreset:
    config = LoopConfig(
        max_steps=8,
        max_tokens=512,
        temperature=0.1,
        system_prompt="lookup agent",
        memory_sources=[StaticMemorySource(text="prefer x over y when both apply")],
    )
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup",
            description="Return the value for key.",
            parameters={"key": "str"},
            execute=lookup_execute,
        )
    )
    registry.register(
        ToolSpec(
            name="done",
            description="Submit final answer.",
            parameters={"answer": "str"},
            execute=done_execute,
        )
    )
    return AgentPreset(
        config=config,
        hooks=[DemoCounter(threshold=5)],
        tools=registry,
        state=DefaultState(max_steps=8),
    )


# ── basic IO ────────────────────────────────────────────────────


def test_write_to_empty_directory(tmp_path: Path) -> None:
    workspace = preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    assert (workspace.path / WorkspaceLayout.WORKSPACE_JSON).is_file()
    assert (workspace.path / WorkspaceLayout.SYSTEM_PROMPT_MD).read_text() == "lookup agent"
    assert (workspace.path / WorkspaceLayout.CONFIG_YAML).is_file()
    assert (workspace.path / WorkspaceLayout.TOOLS_DIR / "lookup" / "tool.yaml").is_file()
    assert (workspace.path / WorkspaceLayout.TOOLS_DIR / "lookup" / "execute.py").is_file()
    assert (workspace.path / WorkspaceLayout.HOOKS_DIR / "00_DemoCounter" / "hook.py").is_file()
    assert (workspace.path / WorkspaceLayout.MEMORY_DIR / "00_static.md").is_file()


def test_non_empty_directory_requires_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "ws"
    out.mkdir()
    (out / "stale").write_text("hi")
    with pytest.raises(FileExistsError):
        preset_to_workspace(_build_demo_preset(), out)
    # overwrite=True succeeds and wipes managed subdirs.
    workspace = preset_to_workspace(_build_demo_preset(), out, overwrite=True)
    assert (workspace.path / WorkspaceLayout.WORKSPACE_JSON).is_file()


def test_workspace_metadata_round_trips(tmp_path: Path) -> None:
    preset_to_workspace(
        _build_demo_preset(),
        tmp_path / "ws",
        name="demo",
        description="just a test",
    )
    loaded = Workspace.from_directory(tmp_path / "ws")
    assert loaded.name == "demo"
    assert loaded.description == "just a test"
    assert loaded.schema_version == 1


def test_missing_metadata_raises(tmp_path: Path) -> None:
    out = tmp_path / "ws"
    out.mkdir()
    (out / "config.yaml").write_text("max_steps: 5\n")
    with pytest.raises(FileNotFoundError):
        Workspace.from_directory(out)
    with pytest.raises(FileNotFoundError):
        workspace_to_preset(out)


# ── round-trip preset structure ─────────────────────────────────


def test_round_trip_preserves_config_subset(tmp_path: Path) -> None:
    preset = _build_demo_preset()
    preset_to_workspace(preset, tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    assert loaded.config.max_steps == preset.config.max_steps
    assert loaded.config.max_tokens == preset.config.max_tokens
    assert loaded.config.temperature == pytest.approx(preset.config.temperature)
    assert loaded.config.system_prompt == "lookup agent"
    assert loaded.config.done_tool == preset.config.done_tool


def test_round_trip_preserves_tools(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    names = {spec.name for spec in loaded.tools._tools.values()}  # type: ignore[attr-defined]
    assert names == {"lookup", "done"}
    lookup_spec = loaded.tools._tools["lookup"]  # type: ignore[attr-defined]
    assert lookup_spec.execute(key="x") == {"key": "x", "value": 1}
    assert lookup_spec.execute(key="y") == {"key": "y", "value": 2}
    assert lookup_spec.parameters == {"key": "str"}


def test_round_trip_preserves_hook_with_to_config(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    assert len(loaded.hooks) == 1
    hook = loaded.hooks[0]
    assert type(hook).__name__ == "DemoCounter"
    assert hook.threshold == 5


def test_round_trip_preserves_static_memory(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    sources = list(loaded.config.memory_sources)
    assert len(sources) == 1
    assert isinstance(sources[0], StaticMemorySource)
    assert sources[0].text == "prefer x over y when both apply"


# ── runnable round-trip ─────────────────────────────────────────


def test_round_tripped_preset_runs_end_to_end(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")

    llm = MockLLMBackend(
        responses=[
            '{"tool":"lookup","args":{"key":"x"},"reasoning":"check"}',
            '{"tool":"done","args":{"answer":"x=1"},"reasoning":""}',
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=loaded.tools,
            state=loaded.state,
            config=loaded.config,
            hooks=loaded.hooks,
            task={"goal": "lookup x"},
        )
    )
    assert [s.tool_call.tool for s in steps] == ["lookup", "done"]
    # The DemoCounter's post_dispatch ran for the lookup step.
    assert loaded.hooks[0].seen >= 1


# ── warnings + strict ──────────────────────────────────────────


def test_non_serializable_config_field_warns_in_loose_mode(tmp_path: Path) -> None:
    preset = _build_demo_preset()
    # Set a callable on a known non-serializable field.
    preset.config.build_briefing = lambda **_: "x"
    workspace = preset_to_workspace(preset, tmp_path / "ws")
    assert any("build_briefing" in w for w in workspace.serialization_warnings)


def test_non_serializable_config_field_raises_in_strict_mode(tmp_path: Path) -> None:
    preset = _build_demo_preset()
    preset.config.build_briefing = lambda **_: "x"
    with pytest.raises(WorkspaceSerializationError):
        preset_to_workspace(preset, tmp_path / "ws", strict=True)


def test_warnings_are_empty_for_clean_preset(tmp_path: Path) -> None:
    workspace = preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    # The demo preset uses only round-trippable fields; no warnings expected.
    assert workspace.serialization_warnings == []


# ── layout sanity ──────────────────────────────────────────────


def test_workspace_json_is_stable(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws", name="demo")
    payload = json.loads((tmp_path / "ws" / "workspace.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["name"] == "demo"
    assert "metadata" in payload


def test_layout_constants_match_written_paths(tmp_path: Path) -> None:
    workspace = preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    expected = {
        WorkspaceLayout.WORKSPACE_JSON,
        WorkspaceLayout.SYSTEM_PROMPT_MD,
        WorkspaceLayout.CONFIG_YAML,
    }
    for relative in expected:
        assert (workspace.path / relative).exists(), f"missing {relative}"
