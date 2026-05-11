"""Tests for the v1.0 declarative slots: model, permissions, memory, output_schema.

Each block has a small unit test plus an integration test that goes
through ``cartridge_to_preset`` end-to-end on a tmp cartridge.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from looplet.cartridge import cartridge_to_preset
from looplet.permissions import PermissionDecision, PermissionEngine, PermissionHook
from looplet.spec_slots import (
    compile_model_block,
    compile_output_schema,
    compile_permissions_block,
    default_long_term_memory_path,
)
from looplet.validation import OutputSchema, validate_args

# ── unit tests: compile_permissions_block ──────────────────────────


def test_permissions_block_default_only() -> None:
    hook = compile_permissions_block({})
    assert isinstance(hook, PermissionHook)
    assert hook.engine.default == PermissionDecision.ALLOW
    assert hook.engine.rules == []


def test_permissions_block_default_explicit() -> None:
    hook = compile_permissions_block({"default": "deny"})
    assert hook.engine.default == PermissionDecision.DENY


def test_permissions_block_default_invalid() -> None:
    with pytest.raises(ValueError, match="permissions.default"):
        compile_permissions_block({"default": "maybe"})


def test_permissions_block_bare_string_rule() -> None:
    hook = compile_permissions_block({"deny": ["bash"]})
    assert len(hook.engine.rules) == 1
    rule = hook.engine.rules[0]
    assert rule.tool == "bash"
    assert rule.decision == PermissionDecision.DENY
    assert rule.arg_matcher is None


def test_permissions_block_contains_matcher() -> None:
    hook = compile_permissions_block(
        {
            "deny": [
                {"tool": "bash", "contains": {"command": "rm -rf"}, "reason": "destructive"},
            ]
        }
    )
    rule = hook.engine.rules[0]
    assert rule.tool == "bash"
    assert rule.reason == "destructive"
    assert rule.arg_matcher is not None
    assert rule.arg_matcher({"command": "rm -rf /"})
    assert not rule.arg_matcher({"command": "echo hi"})


def test_permissions_block_matches_matcher() -> None:
    hook = compile_permissions_block(
        {"allow": [{"tool": "write", "matches": {"file_path": "/tmp/ok"}}]}
    )
    rule = hook.engine.rules[0]
    assert rule.arg_matcher is not None
    assert rule.arg_matcher({"file_path": "/tmp/ok"})
    assert not rule.arg_matcher({"file_path": "/etc/passwd"})


def test_permissions_block_evaluation_order() -> None:
    hook = compile_permissions_block(
        {
            "allow": ["bash"],
            "deny": [{"tool": "bash", "contains": {"command": "rm -rf"}}],
        }
    )
    # Deny rule is appended first (per the documented order); the
    # PermissionEngine evaluates rules in order, so the deny rule
    # for the dangerous arg matches before the broad allow.
    rules = hook.engine.rules
    assert rules[0].decision == PermissionDecision.DENY
    assert rules[1].decision == PermissionDecision.ALLOW


def test_permissions_block_missing_tool_field() -> None:
    with pytest.raises(ValueError, match="missing 'tool'"):
        compile_permissions_block({"deny": [{"contains": {"x": "y"}}]})


def test_permissions_block_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="permissions block"):
        compile_permissions_block("nope")  # type: ignore[arg-type]


# ── unit tests: compile_model_block ────────────────────────────────


def test_model_block_passes_sampling_to_loopconfig() -> None:
    overrides = compile_model_block({"max_tokens": 8000, "temperature": 0.0}, existing_cfg={})
    assert overrides == {"max_tokens": 8000, "temperature": 0.0}


def test_model_block_metadata_carries_provider_and_name() -> None:
    overrides = compile_model_block(
        {
            "provider": "anthropic",
            "name": "claude-sonnet-4.6",
            "reasoning_effort": "high",
            "extra": {"cache_control": "ephemeral"},
        },
        existing_cfg={},
    )
    assert overrides["tool_metadata"]["model"] == {
        "provider": "anthropic",
        "name": "claude-sonnet-4.6",
        "reasoning_effort": "high",
        "extra": {"cache_control": "ephemeral"},
    }
    assert "max_tokens" not in overrides
    assert "temperature" not in overrides


def test_model_block_merges_with_existing_metadata() -> None:
    overrides = compile_model_block(
        {"provider": "openai"},
        existing_cfg={"tool_metadata": {"other": 1}},
    )
    assert overrides["tool_metadata"]["other"] == 1
    assert overrides["tool_metadata"]["model"]["provider"] == "openai"


def test_model_block_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="model block"):
        compile_model_block("anthropic", existing_cfg={})  # type: ignore[arg-type]


# ── unit tests: compile_output_schema ──────────────────────────────


def test_output_schema_basic() -> None:
    schema = compile_output_schema(
        {
            "type": "object",
            "required": ["summary", "pass"],
            "properties": {
                "summary": {"type": "string"},
                "pass": {"type": "boolean"},
            },
        }
    )
    assert isinstance(schema, OutputSchema)
    assert set(schema.fields) == {"summary", "pass"}
    assert schema.fields["summary"].required is True
    assert schema.fields["summary"].field_type == "str"
    assert schema.fields["pass"].field_type == "bool"

    ok = validate_args(schema, {"summary": "done", "pass": True})
    assert ok.valid

    missing = validate_args(schema, {"summary": "done"})
    assert not missing.valid

    wrong_type = validate_args(schema, {"summary": "done", "pass": "yes"})
    assert not wrong_type.valid


def test_output_schema_enum_to_allowed_values() -> None:
    schema = compile_output_schema(
        {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["ok", "fail"]},
            },
        }
    )
    spec = schema.fields["verdict"]
    assert spec.allowed_values == ["ok", "fail"]


def test_output_schema_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="output_schema.type"):
        compile_output_schema({"type": "string"})


def test_output_schema_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="output_schema"):
        compile_output_schema("nope")  # type: ignore[arg-type]


# ── default long_term path ─────────────────────────────────────────


def test_default_long_term_memory_path() -> None:
    assert default_long_term_memory_path() == "memory/long_term.md"


# ── integration tests via cartridge_to_preset ──────────────────────


def _write_minimal_workspace(root: Path, *, config_yaml: str, done_yaml: str | None = None) -> None:
    (root / "workspace.json").write_text(
        json.dumps({"name": "test_agent", "schema_version": 1}) + "\n"
    )
    (root / "config.yaml").write_text(config_yaml)
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "system.md").write_text("you are a test agent\n")
    tools = root / "tools" / "done"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "tool.yaml").write_text(
        done_yaml
        or textwrap.dedent(
            """\
            name: done
            description: Done.
            parameters:
              summary: { type: string }
            """
        )
    )
    (tools / "execute.py").write_text(
        "def execute(ctx, *, summary: str) -> dict:\n    return {'summary': summary}\n"
    )


def test_loader_installs_permissions_hook(tmp_path: Path) -> None:
    _write_minimal_workspace(
        tmp_path,
        config_yaml=textwrap.dedent(
            """\
            max_steps: 5
            permissions:
              default: allow
              deny:
                - tool: bash
                  contains:
                    command: "rm -rf"
                  reason: "destructive"
            """
        ),
    )
    preset = cartridge_to_preset(str(tmp_path))
    perm_hooks = [h for h in preset.hooks if isinstance(h, PermissionHook)]
    assert len(perm_hooks) == 1
    engine = perm_hooks[0].engine
    assert engine.default == PermissionDecision.ALLOW
    assert any(r.tool == "bash" and r.decision == PermissionDecision.DENY for r in engine.rules)


def test_loader_compiles_structured_model_block(tmp_path: Path) -> None:
    _write_minimal_workspace(
        tmp_path,
        config_yaml=textwrap.dedent(
            """\
            model:
              provider: anthropic
              name: claude-sonnet-4.6
              reasoning_effort: high
              max_tokens: 4096
              temperature: 0.0
            """
        ),
    )
    preset = cartridge_to_preset(str(tmp_path))
    assert preset.config.max_tokens == 4096
    assert preset.config.temperature == 0.0
    metadata = preset.config.tool_metadata.get("model", {})
    assert metadata.get("provider") == "anthropic"
    assert metadata.get("reasoning_effort") == "high"


def test_loader_auto_loads_long_term_memory(tmp_path: Path) -> None:
    _write_minimal_workspace(tmp_path, config_yaml="max_steps: 3\n")
    (tmp_path / "memory").mkdir(exist_ok=True)
    (tmp_path / "memory" / "long_term.md").write_text("REMEMBER: always test in tmp.\n")
    preset = cartridge_to_preset(str(tmp_path))
    rendered = "\n".join(s.text for s in preset.config.memory_sources if hasattr(s, "text"))
    assert "REMEMBER" in rendered


def test_loader_explicit_memory_long_term_path(tmp_path: Path) -> None:
    _write_minimal_workspace(
        tmp_path,
        config_yaml=textwrap.dedent(
            """\
            memory:
              long_term: notes/big.md
            """
        ),
    )
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "big.md").write_text("THE-BIG-NOTE\n")
    preset = cartridge_to_preset(str(tmp_path))
    rendered = "\n".join(s.text for s in preset.config.memory_sources if hasattr(s, "text"))
    assert "THE-BIG-NOTE" in rendered


def test_loader_installs_output_schema_from_done_tool(tmp_path: Path) -> None:
    _write_minimal_workspace(
        tmp_path,
        config_yaml="max_steps: 3\n",
        done_yaml=textwrap.dedent(
            """\
            name: done
            description: Done.
            parameters:
              summary: { type: string }
              passed:  { type: boolean }
            output_schema:
              type: object
              required: [summary, passed]
              properties:
                summary: { type: string }
                passed:  { type: boolean }
            """
        ),
    )
    preset = cartridge_to_preset(str(tmp_path))
    assert preset.config.output_schema is not None
    assert "summary" in preset.config.output_schema.fields
    assert "passed" in preset.config.output_schema.fields


def test_loader_strict_rejects_invalid_permissions(tmp_path: Path) -> None:
    _write_minimal_workspace(
        tmp_path,
        config_yaml=textwrap.dedent(
            """\
            permissions:
              default: maybe
            """
        ),
    )
    with pytest.raises(Exception, match="permissions"):
        cartridge_to_preset(str(tmp_path), strict=True)


def test_loader_strict_rejects_invalid_output_schema(tmp_path: Path) -> None:
    _write_minimal_workspace(
        tmp_path,
        config_yaml="max_steps: 3\n",
        done_yaml=textwrap.dedent(
            """\
            name: done
            description: Done.
            parameters:
              summary: { type: string }
            output_schema:
              type: string
            """
        ),
    )
    with pytest.raises(Exception, match="output_schema"):
        cartridge_to_preset(str(tmp_path), strict=True)
