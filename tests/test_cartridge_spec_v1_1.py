"""Regression tests for Cartridge Spec v1.1 additions.

Five additive features, each tested both at the loader level and
end-to-end with the loop:

1. Tool ``tags:`` (declarative metadata, advisory).
2. Tool ``render:`` hints (advisory rendering hints).
3. Single-file tool form (``tools/<name>.py``).
4. ``done_tools: [a, b]`` (additional terminal sentinels).
5. ``prompts/briefing.md`` + ``prompts/recovery.md`` auto-attached hooks.

Plus one parser bug surfaced while writing the dogfood cartridge:

6. Inline YAML comments must be stripped from values
   (``done_tool: done  # foo`` → registered name was the literal
   string ``"done  # foo"``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import looplet
from looplet import (
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    cartridge_to_preset,
    composable_loop,
)


def _write_minimal(root: Path) -> None:
    """Write a minimal valid cartridge to ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 1}\n')
    (root / "config.yaml").write_text("max_steps: 3\n")
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("test agent")


# ── 1. tool tags ─────────────────────────────────────────────────


def test_tool_tags_round_trip_via_tool_yaml(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\n"
        "description: done\n"
        "parameters:\n  summary: { type: string }\n"
        "tags: [terminal, normal-path]\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    spec = preset.tools._tools["done"]
    assert spec.tags == ["terminal", "normal-path"]


def test_tool_tags_default_to_empty_list(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    spec = preset.tools._tools["done"]
    assert spec.tags == []


# ── 2. tool render hints ─────────────────────────────────────────


def test_tool_render_hints_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\n"
        "description: done\n"
        "parameters:\n  summary: { type: string }\n"
        "render:\n  preview: 5\n  max_chars: 800\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    spec = preset.tools._tools["done"]
    assert spec.render == {"preview": 5, "max_chars": 800}


# ── 3. single-file tool form ─────────────────────────────────────


def test_single_file_tool_form_loads(tmp_path: Path) -> None:
    """tools/echo.py with module-level dunders becomes a ToolSpec."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "echo.py").write_text(
        '"""Echo back what you got."""\n'
        '__name__ = "echo"\n'
        '__description__ = "Echo back what you got."\n'
        '__parameters__ = {"text": {"type": "string"}}\n'
        '__tags__ = ["test"]\n'
        '__render__ = {"preview": 3}\n'
        "\n"
        "def execute(ctx, *, text):\n"
        "    return {'echoed': text}\n"
    )
    # Need a done tool too.
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    assert "echo" in preset.tools.tool_names
    spec = preset.tools._tools["echo"]
    assert spec.name == "echo"
    assert spec.description == "Echo back what you got."
    assert spec.tags == ["test"]
    assert spec.render == {"preview": 3}
    assert "text" in spec.parameter_names()


def test_single_file_tool_dispatches_correctly(tmp_path: Path) -> None:
    """Agent loop can actually invoke a single-file tool."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "echo.py").write_text(
        '__name__ = "echo"\n'
        '__description__ = "echo"\n'
        '__parameters__ = {"text": {"type": "string"}}\n'
        "\n"
        "def execute(ctx, *, text):\n"
        "    return {'echoed': text}\n"
    )
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "echo", "args": {"text": "hi"}, "reasoning": "", "call_id": "1"}),
            json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "", "call_id": "2"}
            ),
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"goal": "test"},
        )
    )
    assert len(steps) == 2
    assert steps[0].tool_call.tool == "echo"
    assert steps[0].tool_result.data == {"echoed": "hi"}


def test_single_file_tool_missing_execute_raises_strict(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "broken.py").write_text("# no execute function defined\n")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    with pytest.raises(looplet.CartridgeSerializationError):
        cartridge_to_preset(str(root), strict=True)


# ── 4. done_tools plural ─────────────────────────────────────────


def test_done_tools_plural_terminates_on_either(tmp_path: Path) -> None:
    """Agent that calls EITHER ``report`` or ``escalate`` ends the loop."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    (root / "config.yaml").write_text("max_steps: 5\ndone_tool: report\ndone_tools: [escalate]\n")
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "report.py").write_text(
        '__name__ = "report"\n__description__ = "report"\n'
        '__parameters__ = {"summary": {"type": "string"}}\n\n'
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    (tools / "escalate.py").write_text(
        '__name__ = "escalate"\n__description__ = "escalate"\n'
        '__parameters__ = {"reason": {"type": "string"}}\n\n'
        "def execute(ctx, *, reason):\n    return {'reason': reason}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    assert preset.config.done_tool == "report"
    assert preset.config.done_tools == ["escalate"]

    # Path 1: agent finishes via ``escalate`` (the secondary sentinel).
    llm = MockLLMBackend(
        responses=[
            json.dumps(
                {"tool": "escalate", "args": {"reason": "x"}, "reasoning": "", "call_id": "1"}
            ),
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"goal": "test"},
        )
    )
    assert len(steps) == 1
    assert steps[0].tool_call.tool == "escalate"


def test_done_tools_plural_default_empty(tmp_path: Path) -> None:
    """Default ``done_tools`` is empty list (back-compat)."""
    cfg = LoopConfig(max_steps=5)
    assert cfg.done_tools == []


# ── 5. prompts/briefing.md + recovery.md auto-attach ─────────────


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_briefing_md_attaches_static_briefing_hook(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    (root / "prompts" / "briefing.md").write_text("Always be polite.")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    hook_names = [type(h).__name__ for h in preset.hooks]
    assert "StaticBriefingHook" in hook_names
    sb = next(h for h in preset.hooks if type(h).__name__ == "StaticBriefingHook")
    assert sb.text == "Always be polite."


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_recovery_md_attaches_recovery_hint_hook(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    (root / "prompts" / "recovery.md").write_text("Try again with smaller args.")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    hook_names = [type(h).__name__ for h in preset.hooks]
    assert "RecoveryHintHook" in hook_names


def test_no_prompt_files_means_no_extra_hooks(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    hook_names = [type(h).__name__ for h in preset.hooks]
    assert "StaticBriefingHook" not in hook_names
    assert "RecoveryHintHook" not in hook_names


# ── 6. inline YAML comments must be stripped ─────────────────────


def test_inline_comment_stripped_from_scalar_value(tmp_path: Path) -> None:
    """``done_tool: done  # primary terminal`` registered ``done  # primary terminal`` as
    the literal name before the round-17 fix.
    """
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    (root / "config.yaml").write_text(
        "max_steps: 3\n"
        "done_tool: done                   # this is the terminal sentinel\n"
        "done_tools: [escalate]            # alternate path\n"
    )
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    (tools / "escalate.py").write_text(
        '__name__ = "escalate"\n__description__ = "escalate"\n'
        '__parameters__ = {"reason": {"type": "string"}}\n\n'
        "def execute(ctx, *, reason):\n    return {'reason': reason}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    assert preset.config.done_tool == "done", f"inline comment leaked: {preset.config.done_tool!r}"
    assert preset.config.done_tools == ["escalate"], (
        f"inline comment leaked or flow list mis-parsed: {preset.config.done_tools!r}"
    )


def test_inline_comment_does_not_eat_url_anchor() -> None:
    """``url: \"https://example.com#anchor\"`` must keep its fragment."""
    from looplet.cartridge import _load_yaml

    parsed = _load_yaml('url: "https://example.com#anchor"\n')
    assert parsed["url"] == "https://example.com#anchor"


def test_inline_comment_inside_flow_collection_not_stripped() -> None:
    """``# inside [...]`` is part of the flow value, not a comment."""
    from looplet.cartridge import _load_yaml

    # Empty list with a real trailing comment.
    parsed = _load_yaml("done_tools: []          # nothing here\n")
    assert parsed["done_tools"] == []
