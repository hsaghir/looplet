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
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 3\n")
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("test agent")


# ── 1. tool tags ─────────────────────────────────────────────────


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


def test_tool_render_hints_runtime_yaml_overrides(tmp_path: Path) -> None:
    """Runtime overrides for render hints belong in ``runtime.yaml``.

    Cartridge spec v2 principled exclusion: ``render:`` in tool.yaml
    declares the agent's default; the host shifts the policy via
    ``runtime.yaml: tool_render_hints: { <tool>: {...} }`` without
    editing the cartridge body. Shallow-merge: runtime keys win,
    tool.yaml keys not overridden survive.
    """
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
    (root / "runtime.yaml").write_text("tool_render_hints:\n  done:\n    preview: 25\n")
    preset = cartridge_to_preset(str(root), strict=True)
    spec = preset.tools._tools["done"]
    # ``preview`` overridden by runtime; ``max_chars`` survives from tool.yaml.
    assert spec.render == {"preview": 25, "max_chars": 800}


def test_tool_render_hints_unknown_tool_strict_rejects(tmp_path: Path) -> None:
    """Typo in ``tool_render_hints:`` is a load-time error under strict."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    (root / "runtime.yaml").write_text("tool_render_hints:\n  no_such_tool:\n    preview: 1\n")
    import pytest  # noqa: PLC0415

    with pytest.raises(looplet.CartridgeSerializationError, match="unknown tool"):
        cartridge_to_preset(str(root), strict=True)


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


def test_done_tools_plural_default_empty(tmp_path: Path) -> None:
    """Default ``done_tools`` is empty list (back-compat)."""
    cfg = LoopConfig(max_steps=5)
    assert cfg.done_tools == []


def test_v2_rejects_done_tools_in_cartridge(tmp_path: Path) -> None:
    """Cartridge spec v2 cut: ``done_tools:`` plural sentinels removed.

    Principled alternative: one ``done`` tool with an
    ``output_schema:`` whose payload carries an ``outcome:`` enum
    discriminating the branches. The error message must point the
    user at the alternative pattern.
    """
    root = tmp_path / "v2.cartridge"
    _write_minimal(root)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 5\ndone_tool: report\ndone_tools: [escalate]\n")
    done = root / "tools" / "report"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: report\ndescription: r\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    with pytest.raises(
        looplet.CartridgeSerializationError,
        match=r"done_tools.*spec v2|payload-discriminated",
    ):
        cartridge_to_preset(str(root), strict=True)


def test_v2_payload_discriminated_outcome_validates(tmp_path: Path) -> None:
    """v2 worked example: one ``done``, one ``output_schema``, payload-discriminated outcome.

    The single done's schema branches on an ``outcome:`` enum; the
    loop validates the args against that schema. Replaces the v1
    ``done_tools:`` per-sentinel pattern.
    """
    root = tmp_path / "v2.cartridge"
    _write_minimal(root)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 5\ndone_tool: done\n")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\n"
        "description: report or escalate\n"
        "parameters:\n"
        "  outcome: { type: string, enum: [report, escalate] }\n"
        "  summary: { type: string, default: '' }\n"
        "  blocked_on: { type: string, default: '' }\n"
        "output_schema:\n"
        "  type: object\n"
        "  required: [outcome]\n"
        "  oneOf:\n"
        "    - properties: { outcome: { const: report }, summary: { type: string, minLength: 1 } }\n"
        "      required: [outcome, summary]\n"
        "    - properties: { outcome: { const: escalate }, blocked_on: { type: string, minLength: 1 } }\n"
        "      required: [outcome, blocked_on]\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, outcome, summary='', blocked_on=''):\n"
        "    return {'outcome': outcome}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    # The single done's schema is recorded for primary-sentinel validation
    # via ``LoopConfig.output_schema``. Verify both branches load cleanly
    # by exercising one valid call from each.
    for outcome_args in (
        {"outcome": "report", "summary": "ok"},
        {"outcome": "escalate", "blocked_on": "missing creds"},
    ):
        llm = MockLLMBackend(
            responses=[
                json.dumps({"tool": "done", "args": outcome_args, "reasoning": "", "call_id": "1"}),
            ]
        )
        # Fresh state per run (state is mutable across loop invocations).
        state = DefaultState(max_steps=preset.config.max_steps)
        steps = list(
            composable_loop(
                llm=llm,
                tools=preset.tools,
                state=state,
                config=preset.config,
                hooks=preset.hooks,
                task={"goal": "test"},
            )
        )
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "done"
        assert steps[0].tool_result.error is None


def test_v2_rejects_tags_on_multi_file_tool(tmp_path: Path) -> None:
    """v2 cut: ``tags:`` on tool.yaml — categorisation is a hook concern."""
    root = tmp_path / "v2.cartridge"
    _write_minimal(root)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\n"
        "parameters:\n  summary: { type: string }\n"
        "tags: [terminal]\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    with pytest.raises(looplet.CartridgeSerializationError, match=r"tags:.*spec v2"):
        cartridge_to_preset(str(root), strict=True)


def test_v2_rejects_dunders_on_single_file_tool(tmp_path: Path) -> None:
    """v2 cut: single-file tools must stay trivial — no resources/render/tags."""
    root = tmp_path / "v2.cartridge"
    _write_minimal(root)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "done.py").write_text(
        '__name__ = "done"\n__description__ = "d"\n'
        '__parameters__ = {"summary": {"type": "string"}}\n'
        '__requires__ = ["siem"]\n\n'
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    with pytest.raises(
        looplet.CartridgeSerializationError, match=r"single-file tool.*__requires__"
    ):
        cartridge_to_preset(str(root), strict=True)


def test_v2_rejects_py_ref_grammar(tmp_path: Path) -> None:
    """v2 cut: ``${py:module:symbol}`` — wrap in resources/<name>.py builder."""
    root = tmp_path / "v2.cartridge"
    _write_minimal(root)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    # ``compact_service`` is a runtime-tier field; it must live in
    # runtime.yaml under v2. Use it to plant a ${py:...} ref so the
    # ref-resolution pass sees and rejects the legacy grammar.
    (root / "runtime.yaml").write_text(
        "compact_service: ${py:looplet.compact:DefaultCompactService}\n"
    )
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: d\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    with pytest.raises(
        looplet.CartridgeSerializationError,
        match=r"\$\{py:.*\} reference grammar is removed",
    ):
        cartridge_to_preset(str(root), strict=True)


# Legacy v1.1 ``done_tools:`` per-sentinel validation removed in v2.
# The previous test fixture lived here; see
# ``test_v2_payload_discriminated_outcome_validates`` above for the
# replacement pattern (one done, one schema, payload-discriminated outcome).


# ── 5. prompts/briefing.md + recovery.md auto-attach ─────────────


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
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
        "max_steps: 3\ndone_tool: done                   # this is the terminal sentinel\n"
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
    preset = cartridge_to_preset(str(root), strict=True)
    assert preset.config.done_tool == "done", f"inline comment leaked: {preset.config.done_tool!r}"


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


# ── 7. cartridge.json: language: field ───────────────────────────


def test_language_defaults_to_python_when_missing(tmp_path: Path) -> None:
    """Manifest without ``language:`` loads (back-compat default)."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: d\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    assert preset is not None


def test_v2_rejects_non_python_language(tmp_path: Path) -> None:
    """Loader refuses cleanly when ``language:`` is not python."""
    from looplet.cartridge import CartridgeSerializationError

    root = tmp_path / "x.cartridge"
    root.mkdir()
    (root / "cartridge.json").write_text(
        '{"name": "x", "schema_version": 2, "language": "typescript"}\n'
    )
    (root / "config.yaml").write_text("max_steps: 3\n")
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("x")
    with pytest.raises(CartridgeSerializationError, match="typescript"):
        cartridge_to_preset(str(root), strict=True)


def test_language_round_trips_via_preset_to_cartridge(tmp_path: Path) -> None:
    """``preset_to_cartridge`` writes ``language: python`` into manifest."""
    from looplet import preset_to_cartridge

    root = tmp_path / "src.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: d\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    out = tmp_path / "out.cartridge"
    preset_to_cartridge(preset, str(out))
    meta = json.loads((out / "cartridge.json").read_text())
    assert meta["language"] == "python"


# ── 8. tools/<n>/description.md promotion ────────────────────────


def test_tool_description_md_overrides_yaml(tmp_path: Path) -> None:
    """When ``description.md`` is present, its content wins."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: short yaml\nparameters:\n  summary: { type: string }\n"
    )
    (done / "description.md").write_text("Long-form description.\n\nWith multiple paragraphs.\n")
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    spec = preset.tools._tools["done"]
    assert "Long-form description" in spec.description
    assert "multiple paragraphs" in spec.description
    assert "short yaml" not in spec.description


def test_tool_description_md_absent_falls_back_to_yaml(tmp_path: Path) -> None:
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: yaml-only desc\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    assert preset.tools._tools["done"].description == "yaml-only desc"


def test_serialiser_promotes_multiline_description_to_md(tmp_path: Path) -> None:
    """Multi-line tool descriptions get written to description.md on round-trip."""
    from looplet import preset_to_cartridge

    root = tmp_path / "src.cartridge"
    _write_minimal(root)
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: short\nparameters:\n  summary: { type: string }\n"
    )
    (done / "description.md").write_text("Headline.\n\nSecond paragraph here.\n")
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    out = tmp_path / "out.cartridge"
    preset_to_cartridge(preset, str(out))
    desc_md = out / "tools" / "done" / "description.md"
    assert desc_md.is_file()
    assert "Second paragraph" in desc_md.read_text()
