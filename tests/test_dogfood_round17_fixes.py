"""Regression tests for dogfood round 17 — fixes surfaced by building
the support_l1 cartridge (extends support_base, 3 terminal sentinels,
sub-agent invocation, EscalationGateHook).

Two real loader / loop bugs:

1. **output_schema validated against the wrong sentinel.** The loop
   was running ``config.output_schema`` validation against EVERY
   terminal sentinel — including the v1.1 secondary ``done_tools``,
   even though the schema was authored for the primary ``done_tool``.
   A perfectly valid ``escalate(...)`` payload was rejected because
   it didn't match the ``resolve`` schema.

2. **Single-file ↔ multi-file tool name collision was masked.** When
   a cartridge had both ``tools/escalate.py`` (single-file) and an
   empty ``tools/escalate/`` (leftover from setup), the loader walked
   the empty dir and failed with a confusing "missing tool.yaml or
   execute.py" error instead of naming the collision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import (
    CartridgeSerializationError,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    cartridge_to_preset,
    composable_loop,
)


def _write_minimal(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 5\n")
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("test")


# ── output_schema only validates the primary done_tool ──────────


def test_output_schema_still_validates_primary_done_tool(tmp_path: Path) -> None:
    """Sanity check: the primary done_tool's schema is still enforced —
    the fix only narrows when validation runs, doesn't disable it."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    (root / "config.yaml").write_text("max_steps: 5\ndone_tool: resolve\n")
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    resolve_dir = tools / "resolve"
    resolve_dir.mkdir()
    (resolve_dir / "tool.yaml").write_text(
        "name: resolve\n"
        "description: resolve\n"
        "parameters:\n"
        "  draft: { type: string }\n"
        "output_schema:\n"
        "  type: object\n"
        "  required: [draft, evidence]\n"
        "  properties:\n"
        "    draft: { type: string }\n"
        "    evidence: { type: array }\n"
    )
    (resolve_dir / "execute.py").write_text("def execute(ctx, *, draft): return {'draft': draft}\n")

    # Call resolve WITHOUT the required ``evidence`` field — must be
    # rejected by the loop's output-schema gate.
    llm = MockLLMBackend(
        responses=[
            json.dumps(
                {"tool": "resolve", "args": {"draft": "hello"}, "reasoning": "", "call_id": "1"}
            ),
            json.dumps(
                {
                    "tool": "resolve",
                    "args": {"draft": "hello", "evidence": []},
                    "reasoning": "",
                    "call_id": "2",
                }
            ),
        ]
    )
    preset = cartridge_to_preset(str(root), strict=True)
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
    # Step 1 is rejected by output_schema (no error on the dispatch
    # itself, but a gate_warning that re-prompts the LLM); step 2
    # passes the schema and terminates.
    assert len(steps) >= 1
    # The first call should have an error indicating output_schema rejection
    # (the loop converts the gate_warning into a tool_result.error).
    assert steps[0].tool_result.error is not None and "schema" in steps[0].tool_result.error.lower()


# ── single-file ↔ multi-file collision ───────────────────────────


def test_single_file_multi_file_collision_raises_strict(tmp_path: Path) -> None:
    """When both ``tools/foo.py`` AND ``tools/foo/`` exist (a common
    leftover from ``mkdir`` setup), the loader fails with an actionable
    collision message, not a confusing "missing tool.yaml" error."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "foo.py").write_text(
        '__name__ = "foo"\n__description__ = "foo"\n'
        "__parameters__ = {}\n\n"
        "def execute(ctx): return {'ok': True}\n"
    )
    # The collision: an empty foo/ dir alongside foo.py
    (tools / "foo").mkdir()

    with pytest.raises(CartridgeSerializationError) as exc_info:
        cartridge_to_preset(str(root), strict=True)
    msg = str(exc_info.value)
    assert "collision" in msg.lower()
    assert "foo" in msg
    # Names the actionable fix.
    assert "rmdir" in msg or "Pick one form" in msg


def test_single_file_multi_file_collision_warns_loose(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """In loose mode, the loader warns and prefers the single-file form."""
    import logging

    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "foo.py").write_text(
        '__name__ = "foo"\n__description__ = "foo"\n'
        "__parameters__ = {}\n\n"
        "def execute(ctx): return {'from_single_file': True}\n"
    )
    (tools / "foo").mkdir()
    # Need a done tool to load.
    done = tools / "done"
    done.mkdir()
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text("def execute(ctx, *, summary): return {'summary': summary}\n")
    with caplog.at_level(logging.WARNING, logger="looplet.cartridge"):
        preset = cartridge_to_preset(str(root), strict=False)
    matches = [
        r
        for r in caplog.records
        if "collision" in r.getMessage().lower() and "foo" in r.getMessage()
    ]
    assert matches, "expected a warning naming the collision"
    # Single-file form wins; the empty multi-file dir is dropped.
    assert "foo" in preset.tools.tool_names
    spec = preset.tools._tools["foo"]
    # Sanity: it's the single-file form (returns from_single_file=True).
    result = spec.execute(type("Ctx", (), {"resources": {}})())
    assert result == {"from_single_file": True}


def test_no_collision_when_only_single_file_form_present(tmp_path: Path) -> None:
    """Sanity: a normal single-file tool (no sibling dir) loads fine."""
    root = tmp_path / "x.cartridge"
    _write_minimal(root)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "foo.py").write_text(
        '__name__ = "foo"\n__description__ = "foo"\n'
        "__parameters__ = {}\n\n"
        "def execute(ctx): return {'ok': True}\n"
    )
    done = tools / "done"
    done.mkdir()
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text("def execute(ctx, *, summary): return {'summary': summary}\n")
    preset = cartridge_to_preset(str(root), strict=True)
    assert "foo" in preset.tools.tool_names
    assert "done" in preset.tools.tool_names
