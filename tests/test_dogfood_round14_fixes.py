"""Regression tests for friction found in the code_reviewer dogfood
(round 14). Each test pins one of the loader fixes that came out of
that session.

The dogfood built a complex code-review cartridge that extended
``coder.cartridge``, declared ``permissions:`` with a wrong shape,
put ``output_schema:`` at the top level (wrong place per SPEC.md),
and registered ``builtin_tools: [subagent]`` even though the parent
already had its own subagent tool. Each of these surfaced a real
loader bug:

* The wrong ``permissions:`` shape was silently dropped.
* The top-level ``output_schema:`` left a raw dict on
  ``LoopConfig.output_schema`` (no validation triggered).
* The duplicate ``subagent`` registration produced a confusing
  "name collision" warning even when the two registrations referred
  to the same ``ToolSpec``.

These tests pin the fixes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import looplet
from looplet import CartridgeSerializationError, cartridge_to_preset

REPO = Path(__file__).resolve().parents[1]


def _write_minimal_cartridge(root: Path, *, extra_config: str = "") -> Path:
    """Write a minimal valid cartridge plus optional extra config.yaml lines."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 1}\n')
    (root / "config.yaml").write_text(f"max_steps: 5\n{extra_config}")
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "system.md").write_text("you are a tester.")
    done = root / "tools" / "done"
    done.mkdir(parents=True, exist_ok=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    return root


# ── unknown top-level config keys raise in strict mode ───────────


def test_unknown_top_level_config_key_raises_in_strict(tmp_path: Path) -> None:
    """Typos like ``temprature:`` or hand-rolled slot shapes must fail
    loudly under ``strict=True`` instead of being silently dropped."""
    root = _write_minimal_cartridge(tmp_path / "x.cartridge", extra_config="temprature: 0.5\n")
    with pytest.raises(CartridgeSerializationError) as exc_info:
        cartridge_to_preset(str(root), strict=True)
    msg = str(exc_info.value)
    assert "temprature" in msg
    assert "unknown top-level key" in msg.lower()


def test_unknown_top_level_config_key_warns_when_not_strict(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Without ``strict=True`` we still want a visible warning."""
    root = _write_minimal_cartridge(tmp_path / "x.cartridge", extra_config="bogus_key: 1\n")
    with caplog.at_level(logging.WARNING, logger="looplet.cartridge"):
        cartridge_to_preset(str(root), strict=False)
    matches = [r for r in caplog.records if "bogus_key" in r.getMessage()]
    assert matches, "expected a warning naming the unknown key"


# ── top-level output_schema gives an actionable error ───────────


def test_top_level_output_schema_in_config_yaml_raises_strict(tmp_path: Path) -> None:
    """``output_schema:`` belongs in ``tools/done/tool.yaml``, not in
    ``config.yaml`` — putting it at the top level is a common mistake
    that previously left a raw dict on LoopConfig.output_schema."""
    extra = (
        "output_schema:\n"
        "  type: object\n"
        "  required: [summary]\n"
        "  properties:\n"
        "    summary: { type: string }\n"
    )
    root = _write_minimal_cartridge(tmp_path / "x.cartridge", extra_config=extra)
    with pytest.raises(CartridgeSerializationError) as exc_info:
        cartridge_to_preset(str(root), strict=True)
    msg = str(exc_info.value)
    assert "output_schema" in msg
    assert "tools/done" in msg


def test_top_level_output_schema_warns_when_not_strict(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    extra = (
        "output_schema:\n"
        "  type: object\n"
        "  required: [summary]\n"
        "  properties:\n"
        "    summary: { type: string }\n"
    )
    root = _write_minimal_cartridge(tmp_path / "x.cartridge", extra_config=extra)
    with caplog.at_level(logging.WARNING, logger="looplet.cartridge"):
        preset = cartridge_to_preset(str(root), strict=False)
    matches = [r for r in caplog.records if "output_schema" in r.getMessage()]
    assert matches, "expected a warning about the misplaced output_schema"
    assert preset.config.output_schema is None, (
        "raw dict must NOT be left on LoopConfig.output_schema"
    )


# ── duplicate ToolSpec registration with same identity is silent ──


def test_register_same_toolspec_twice_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Re-registering the *same* ToolSpec object (common when a
    cartridge inherits a built-in via ``extends:`` and also lists
    it under ``builtin_tools:``) should be a no-op, not a warning.
    Warning was confusing 'name collision' noise.
    """
    from looplet import tool, tools_from
    from looplet.tools import BaseToolRegistry

    @tool
    def my_tool(*, x: int) -> dict:
        return {"x": x}

    registry = BaseToolRegistry()
    spec = next(iter(tools_from([my_tool])._tools.values()))
    registry.register(spec)
    with caplog.at_level(logging.WARNING, logger="looplet.tools"):
        registry.register(spec)  # same object — must NOT warn
    msgs = [r.getMessage() for r in caplog.records if "already registered" in r.getMessage()]
    assert not msgs, f"unexpected warning(s): {msgs}"


def test_register_different_toolspec_with_same_name_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sanity: actual collisions (different ToolSpec objects, same
    name) MUST still warn — that's the original purpose of the message.
    """
    from looplet import tool, tools_from
    from looplet.tools import BaseToolRegistry

    @tool
    def my_tool(*, x: int) -> dict:
        return {"x": x}

    @tool
    def my_tool2(*, x: int) -> dict:
        return {"x": x * 2}

    spec_a = next(iter(tools_from([my_tool])._tools.values()))
    spec_b = next(iter(tools_from([my_tool2])._tools.values()))
    spec_b.name = spec_a.name  # force a real collision
    registry = BaseToolRegistry()
    registry.register(spec_a)
    with caplog.at_level(logging.WARNING, logger="looplet.tools"):
        registry.register(spec_b)
    msgs = [r.getMessage() for r in caplog.records if "already registered" in r.getMessage()]
    assert msgs, "expected a warning when two DIFFERENT specs share a name"
