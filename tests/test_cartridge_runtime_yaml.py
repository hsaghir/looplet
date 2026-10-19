"""Tests for cartridge-spec v2 runtime field tiering.

See ``paper/principled_cartridge_v2.md`` for design rationale.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from looplet import CartridgeLayout, cartridge_to_preset


def _write_minimal_cartridge(
    root: Path, *, config_text: str, runtime_text: str | None = None
) -> None:
    """Build a tiny but loadable cartridge skeleton at ``root``."""
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text(config_text)
    if runtime_text is not None:
        (root / "runtime.yaml").write_text(runtime_text)
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("you are a tester")
    (root / "tools" / "done").mkdir(parents=True)
    (root / "tools" / "done" / "tool.yaml").write_text(
        "name: done\ndescription: Finish.\nparameters:\n  summary:\n    type: string\n"
    )
    (root / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx, *, summary: str) -> dict:\n    return {'summary': summary}\n"
    )


def test_field_tiering_carves_loopconfig_into_three_disjoint_buckets() -> None:
    """RUNTIME, HOST, and CONTRACT tiers must be disjoint."""
    runtime = CartridgeLayout.RUNTIME_TIER_FIELDS
    host = CartridgeLayout.HOST_TIER_FIELDS
    contract = CartridgeLayout.contract_tier_fields()
    assert runtime.isdisjoint(host)
    assert runtime.isdisjoint(contract)
    # CONTRACT comes from SERIALIZABLE - RUNTIME, so HOST may not
    # appear there (HOST callables aren't serialisable). Sanity-check
    # the partition covers the documented serialisable surface.
    assert (runtime | contract) >= set(CartridgeLayout.SERIALIZABLE_CONFIG_FIELDS) - {
        "tool_metadata",
        "generate_kwargs",
    }


def test_runtime_yaml_loads_runtime_keys(tmp_path: Path) -> None:
    """Keys in ``runtime.yaml`` populate the same ``LoopConfig`` fields."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="max_tokens: 1234\ntemperature: 0.7\n",
    )
    preset = cartridge_to_preset(str(tmp_path))
    assert preset.config.max_tokens == 1234
    assert preset.config.temperature == 0.7
    assert preset.config.max_steps == 5


def test_text_completion_policy_round_trips_as_contract(tmp_path: Path) -> None:
    from looplet import preset_to_cartridge

    original = tmp_path / "original"
    original.mkdir()
    _write_minimal_cartridge(
        original,
        config_text="max_steps: 5\ndone_tool: done\nallow_text_completion: true\n",
        runtime_text="max_tokens: 100\nmax_turn_continuations: 1\n",
    )
    with cartridge_to_preset(original) as preset:
        assert preset.config.allow_text_completion is True
        exported = tmp_path / "exported"
        preset_to_cartridge(preset, exported)
    with cartridge_to_preset(exported) as restored:
        assert restored.config.allow_text_completion is True
        assert restored.config.max_turn_continuations == 1
        assert "allow_text_completion" in (exported / "config.yaml").read_text()
        assert "allow_text_completion" not in (exported / "runtime.yaml").read_text()


def test_runtime_yaml_loads_parallelism_cap(tmp_path: Path) -> None:
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="concurrent_dispatch: true\nmax_parallel_calls: 3\n",
    )
    preset = cartridge_to_preset(str(tmp_path))
    assert preset.config.concurrent_dispatch is True
    assert preset.config.max_parallel_calls == 3


def test_assistant_policy_export_requires_host_binding_and_versions_semantics(
    tmp_path, monkeypatch
):
    import json
    from dataclasses import replace

    from looplet import AssistantTurnDecision, preset_to_cartridge
    from looplet.blueprints import blueprint_from_preset
    from looplet.cartridge import _load
    from looplet.testing import MockLLMBackend

    original = tmp_path / "original"
    original.mkdir()
    _write_minimal_cartridge(
        original,
        config_text="max_steps: 2\ndone_tool: done\n",
        runtime_text="use_native_tools: false\n",
    )
    policy = lambda context: AssistantTurnDecision("continue")
    with cartridge_to_preset(original) as preset:
        preset.config = replace(
            preset.config, assistant_turn_policy=policy, max_model_calls=3, max_assistant_turns=2
        )
        blueprint = blueprint_from_preset(preset, name="assistant-policy")
        assert blueprint.config["requires_assistant_turn_policy"] is True
        assert "assistant_turn_policy" not in blueprint.config
        exported = tmp_path / "exported"
        preset_to_cartridge(preset, exported)
    manifest = json.loads((exported / "cartridge.json").read_text())
    assert "assistant_turns_v1" in manifest["compatibility"]["requires"]
    assert "requires_assistant_turn_policy: true" in (exported / "config.yaml").read_text()
    assert "max_model_calls" not in (exported / "config.yaml").read_text()
    assert "max_model_calls: 3" in (exported / "runtime.yaml").read_text()
    assert not (exported / "resources" / "assistant_turn_policy.py").exists()
    with cartridge_to_preset(exported) as restored:
        assert restored.config.assistant_turn_policy is None
        assert restored.config.max_assistant_turns == 2
        with pytest.raises(ValueError, match="explicit host"):
            list(restored.run(MockLLMBackend(["Do not call"])))
        restored.config = replace(restored.config, assistant_turn_policy=policy)
        steps = list(
            restored.run(
                MockLLMBackend(["Check evidence", '{"tool":"done","args":{"summary":"checked"}}'])
            )
        )
        assert [step.tool_call.tool for step in steps] == ["done"]
    monkeypatch.setattr(
        _load,
        "_SUPPORTED_COMPATIBILITY_FEATURES",
        _load._SUPPORTED_COMPATIBILITY_FEATURES - {"assistant_turns_v1"},
    )
    with pytest.raises(Exception, match="unsupported Looplet capabilities: assistant_turns_v1"):
        cartridge_to_preset(exported)


@pytest.mark.parametrize("field", ["max_model_calls", "max_assistant_turns"])
@pytest.mark.parametrize("value", ["true", "0", "-1", '"3"'])
def test_assistant_runtime_rejects_malformed_ceilings_on_load(tmp_path, field, value):
    _write_minimal_cartridge(
        tmp_path, config_text="done_tool: done\n", runtime_text=f"{field}: {value}\n"
    )
    with pytest.raises(ValueError, match=field):
        cartridge_to_preset(tmp_path)


def test_runtime_yaml_does_not_mask_runtime_keys_in_config_yaml(tmp_path: Path) -> None:
    """A duplicate runtime key does not make an invalid config valid."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\nmax_tokens: 100\n",
        runtime_text="max_tokens: 999\n",
    )
    with pytest.raises(Exception) as exc_info:
        cartridge_to_preset(str(tmp_path))
    assert "runtime-tier" in str(exc_info.value)
    assert "max_tokens" in str(exc_info.value)


def test_config_yaml_rejects_host_execution_policy(tmp_path: Path) -> None:
    _write_minimal_cartridge(
        tmp_path, config_text="name: host-policy\nschema_version: 2\nexecution_policy: {}"
    )

    from looplet import cartridge_to_preset

    with pytest.raises(Exception, match="host-only"):
        cartridge_to_preset(tmp_path, strict=True)


def test_runtime_yaml_present_silences_warning(tmp_path: Path) -> None:
    """When the runtime key is in runtime.yaml, no warning fires."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="max_tokens: 2000\ntemperature: 0.3\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        cartridge_to_preset(str(tmp_path))
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []


def test_runtime_yaml_rejects_contract_tier_keys(tmp_path: Path) -> None:
    """``runtime.yaml`` may only declare runtime-tier fields; contract
    keys (max_steps, system_prompt, done_tool, model, permissions) are
    rejected with a structured error in strict mode."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="max_steps: 99\n",  # contract key in wrong file
    )
    with pytest.raises(Exception) as exc_info:
        cartridge_to_preset(str(tmp_path), strict=True)
    assert "runtime.yaml" in str(exc_info.value)
    assert "max_steps" in str(exc_info.value)


@pytest.mark.parametrize("target", ["config", "runtime"])
def test_cartridge_files_reject_host_only_keys(tmp_path: Path, target: str) -> None:
    config = "max_steps: 5\ndone_tool: done\n"
    runtime = None
    if target == "config":
        config += "approval_handler: unsafe\n"
    else:
        runtime = "approval_handler: unsafe\n"
    _write_minimal_cartridge(tmp_path, config_text=config, runtime_text=runtime)

    with pytest.raises(Exception) as exc_info:
        cartridge_to_preset(tmp_path, strict=True)
    assert "approval_handler" in str(exc_info.value)
    assert "host-only" in str(exc_info.value) or "non-runtime" in str(exc_info.value)


def test_llm_gateway_requires_boolean(tmp_path: Path) -> None:
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\nllm_gateway: 1\n",
    )

    with pytest.raises(Exception, match="llm_gateway.*boolean"):
        cartridge_to_preset(tmp_path, strict=True)


def test_extends_inherits_parent_runtime_yaml(tmp_path: Path) -> None:
    """A child cartridge that ``extends:`` a parent picks up the
    parent's ``runtime.yaml`` defaults - without this, splitting
    runtime knobs out would silently drop them from descendants."""
    parent = tmp_path / "parent.cartridge"
    parent.mkdir()
    _write_minimal_cartridge(
        parent,
        config_text="max_steps: 10\ndone_tool: done\n",
        runtime_text="max_tokens: 7777\ntemperature: 0.42\n",
    )

    child = tmp_path / "child.cartridge"
    child.mkdir()
    _write_minimal_cartridge(
        child,
        config_text="max_steps: 20\ndone_tool: done\nextends: ../parent.cartridge\n",
    )
    # Remove child's own done tool so the extends picks up parent's.
    # (The minimal scaffold writes one for both; that's fine because
    # extends overlays the directories.)
    preset = cartridge_to_preset(str(child))
    assert preset.config.max_steps == 20  # child overrides parent
    assert preset.config.max_tokens == 7777  # inherited from parent runtime.yaml
    assert preset.config.temperature == 0.42


# ── Phase 2: approval handler contract ─────────────────────────────


def test_ask_rules_without_handler_raise_at_load(tmp_path: Path) -> None:
    """Cartridge spec v2 fail-loud: ``permissions: ask:`` rules require
    a host-supplied ``ask_handler``; without one, loading must raise so
    the human-in-the-loop contract isn't silently broken by ASK falling
    back to the engine default (typically ALLOW)."""
    from looplet import CartridgeSerializationError

    _write_minimal_cartridge(
        tmp_path,
        config_text=(
            "max_steps: 5\n"
            "done_tool: done\n"
            "permissions:\n"
            "  default: allow\n"
            "  ask:\n"
            "    - tool: done\n"
        ),
    )

    with pytest.raises(CartridgeSerializationError, match=r"permissions\.ask"):
        cartridge_to_preset(tmp_path)


def test_ask_rules_with_handler_load_cleanly(tmp_path: Path) -> None:
    """Supplying ``runtime={'ask_handler': callable}`` satisfies the
    contract; the handler is wired onto the compiled engine."""
    from looplet.permissions import PermissionDecision, PermissionHook

    _write_minimal_cartridge(
        tmp_path,
        config_text=(
            "max_steps: 5\n"
            "done_tool: done\n"
            "permissions:\n"
            "  default: allow\n"
            "  ask:\n"
            "    - tool: done\n"
        ),
    )

    sentinel = lambda _call, _rule: PermissionDecision.DENY  # noqa: E731
    preset = cartridge_to_preset(tmp_path, runtime={"ask_handler": sentinel})

    perm_hooks = [h for h in preset.hooks if isinstance(h, PermissionHook)]
    assert len(perm_hooks) == 1
    assert perm_hooks[0].engine.ask_handler is sentinel


def test_non_callable_ask_handler_raises(tmp_path: Path) -> None:
    """Defensive: loader rejects a non-callable ``ask_handler``."""
    from looplet import CartridgeSerializationError

    _write_minimal_cartridge(
        tmp_path,
        config_text=("max_steps: 5\ndone_tool: done\npermissions:\n  ask:\n    - tool: done\n"),
    )

    with pytest.raises(CartridgeSerializationError, match=r"must be callable"):
        cartridge_to_preset(tmp_path, runtime={"ask_handler": "not a function"})


def test_no_ask_rules_means_no_handler_required(tmp_path: Path) -> None:
    """Cartridges without ``ask:`` rules don't need an ``ask_handler``."""
    _write_minimal_cartridge(
        tmp_path,
        config_text=(
            "max_steps: 5\n"
            "done_tool: done\n"
            "permissions:\n"
            "  default: allow\n"
            "  deny:\n"
            "    - tool: done\n"
        ),
    )
    # Should not raise.
    cartridge_to_preset(tmp_path)
