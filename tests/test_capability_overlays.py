"""Tests for capability overlays, step-boundary reload, and origin-scoped registries."""

from __future__ import annotations

import json
from contextlib import aclosing, closing
from pathlib import Path
from typing import Any

import pytest

from looplet import (
    BaseToolRegistry,
    CachePolicy,
    CartridgeSerializationError,
    CheckpointHook,
    CheckpointParticipant,
    CheckpointParticipants,
    DefaultState,
    FileCheckpointStore,
    LoopConfig,
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    ProvenanceSink,
    ReloadHook,
    RunEnvelope,
    RunResult,
    Skill,
    SkillActivationHook,
    SkillManager,
    StaticMemorySource,
    ToolCall,
    ToolSpec,
    WorkspaceWatcher,
    async_composable_loop,
    async_replay_loop,
    cartridge_to_preset,
    composable_loop,
    preset_to_cartridge,
    replay_loop,
    scaffold_cartridge,
    serialize_harness,
    tool,
    tools_from,
)
from looplet.cache import CacheBreakDetector, CacheControl
from looplet.validation import FieldSpec, OutputSchema, ValidatingToolRegistry

pytestmark = pytest.mark.smoke


def _make_base_cartridge(path: Path) -> Path:
    scaffold_cartridge(path, name="base_agent", tools=["echo_base"])
    (path / "prompts" / "system.md").write_text("Base system prompt.", encoding="utf-8")
    (path / "tools" / "echo_base" / "tool.yaml").write_text(
        "name: echo_base\n"
        "description: Return text from the base cartridge.\n"
        "parameters:\n"
        "  text:\n"
        "    type: string\n"
        "    description: Text to echo.\n",
        encoding="utf-8",
    )
    (path / "tools" / "echo_base" / "execute.py").write_text(
        "def execute(ctx, *, text: str) -> dict:\n    return {'echo': text, 'source': 'base'}\n",
        encoding="utf-8",
    )
    return path


def _write_multi_file_tool(
    root: Path,
    name: str,
    body: str,
    *,
    yaml_extra: str = "",
    parameters_yaml: str = "  value:\n    type: integer\n    description: Input integer.\n",
) -> None:
    tool_dir = root / "tools" / name
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "tool.yaml").write_text(
        f"name: {name}\n"
        f"description: Overlay tool {name}.\n"
        f"parameters:\n{parameters_yaml}"
        f"{yaml_extra}",
        encoding="utf-8",
    )
    (tool_dir / "execute.py").write_text(body, encoding="utf-8")


# ── BaseToolRegistry & ValidatingToolRegistry origin operations ─────


def test_tool_spec_origin_validation_and_registry_sync_atomic_rollback():
    with pytest.raises(ValueError, match="non-empty string"):
        ToolSpec(name="bad", description="x", parameters={}, execute=lambda: {}, origin="  ")

    @tool(origin="base")
    def alpha(*, x: int) -> dict[str, int]:
        return {"x": x}

    @tool(origin="overlay")
    def beta(*, y: int) -> dict[str, int]:
        return {"y": y}

    registry = ValidatingToolRegistry()
    registry.register_typed(alpha)
    registry.register_typed(
        ToolSpec(
            name="done", description="finish", parameters={}, execute=lambda: {}, origin="base"
        )
    )

    assert registry.get("alpha") is alpha
    assert registry.get("missing") is None
    assert [s.name for s in registry.tools_for_origin("base")] == ["alpha", "done"]

    summary = registry.sync_origin(
        "overlay",
        [beta],
        protected={"done"},
        argument_schemas={"beta": "parameters"},
    )
    assert summary == {"added": ("beta",), "updated": (), "removed": ()}
    assert registry.get("beta") is not None
    assert registry.get("beta").origin == "overlay"
    # Validation schema applies to synced tool
    bad_dispatch = registry.dispatch(ToolCall(tool="beta", args={"y": "not_an_int"}))
    assert bad_dispatch.error is not None and "Validation failed" in bad_dispatch.error

    # Attempting to overwrite a base-owned or protected tool fails atomically without losing beta
    @tool(name="alpha")
    def sneaky_alpha(*, x: int) -> dict[str, int]:
        return {"x": x + 1}

    with pytest.raises(ValueError, match="owned by origin 'base'"):
        registry.sync_origin("overlay", [sneaky_alpha], protected={"done"})
    assert [s.name for s in registry.tools_for_origin("overlay")] == ["beta"]

    @tool(name="done")
    def sneaky_done() -> dict[str, str]:
        return {"ok": "bad"}

    with pytest.raises(ValueError, match="protected"):
        registry.sync_origin("overlay", [sneaky_done], protected={"done"})
    assert [s.name for s in registry.tools_for_origin("overlay")] == ["beta"]

    # Syncing empty overlay removes beta and cleans up its validation schema
    removed = registry.sync_origin("overlay", (), protected={"done"})
    assert removed == {"added": (), "updated": (), "removed": ("beta",)}
    assert registry.get("beta") is None
    assert "beta" not in registry._schemas


def test_serialize_harness_records_tool_origins_and_reload_hook_config(tmp_path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    _write_multi_file_tool(
        overlay,
        "double_it",
        "def execute(ctx, *, value: int) -> dict:\n    return {'doubled': value * 2}\n",
    )
    with WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        strict=True,
    ) as watcher:
        preset = watcher.preset()
        hook = ReloadHook(watcher, protected_tools={"echo_base"})
        snap = serialize_harness(config=preset.config, tools=preset.tools, hooks=[hook])

    by_name = {entry["name"]: entry for entry in snap["tools"]}
    assert by_name["echo_base"]["origin"] == "cartridge"
    assert by_name["done"]["origin"] == "cartridge"
    assert by_name["double_it"]["origin"] == "overlay"
    assert snap["hook_configurations"][0] == {
        "overlay_count": 1,
        "allowed_overlay_slots": ["tools"],
        "protect_initial_tools": True,
        "protected_tools": ["echo_base"],
        "notify_on_reload": True,
        "notify_on_error": True,
    }


# ── Cartridge overlays, slot allowlists, and security fences ────────


def test_cartridge_overlays_layer_order_helper_modules_and_base_immutability(tmp_path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    ov1 = tmp_path / "ov1"
    ov2 = tmp_path / "ov2"
    ov1.mkdir()
    ov2.mkdir()
    (ov1 / "math_helper.py").write_text("FACTOR = 3\n", encoding="utf-8")
    _write_multi_file_tool(
        ov1,
        "scale_value",
        "import math_helper\n"
        "def execute(ctx, *, value: int) -> dict:\n"
        "    return {'scaled': value * math_helper.FACTOR}\n",
    )
    (ov2 / "math_helper.py").write_text("FACTOR = 10\n", encoding="utf-8")
    (ov2 / "prompts").mkdir()
    (ov2 / "prompts" / "system.md").write_text("Overlay v2 prompt.", encoding="utf-8")

    with cartridge_to_preset(
        base,
        overlays=[ov1, ov2],
        allowed_overlay_slots={"tools", "helpers", "prompts"},
        strict=True,
    ) as preset:
        assert preset.config.system_prompt == "Overlay v2 prompt."
        res = preset.tools.dispatch(ToolCall(tool="scale_value", args={"value": 7}))
        assert res.error is None and res.data == {"scaled": 70}
        assert not (base / "tools" / "scale_value").exists()
        assert (base / "prompts" / "system.md").read_text(encoding="utf-8") == "Base system prompt."


@pytest.mark.parametrize(
    ("rel_path", "content", "match"),
    [
        ("config.yaml", "max_steps: 999\n", "disallowed slot 'config'"),
        ("runtime.yaml", "max_tokens: 9999\n", "disallowed slot 'runtime'"),
        ("hooks/00_Bad/hook.py", "class Bad: pass\n", "disallowed slot 'hooks'"),
        ("evals/eval_cheat.py", "def eval_ok(ctx): return True\n", "cannot modify evals/"),
        (".hidden/secret.py", "x = 1\n", "hidden path"),
        ("setup.py", "x = 1\n", "setup.py"),
        (
            "tools/done/tool.yaml",
            "name: done\ndescription: fake\nparameters: {}\n",
            "protected tool 'done'",
        ),
        ("tools/done.py", "def execute(): return {}\n", "protected tool 'done'"),
    ],
)
def test_cartridge_overlays_reject_disallowed_slots_and_protected_done(
    tmp_path, rel_path, content, match
):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    target = overlay / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    with pytest.raises(CartridgeSerializationError, match=match):
        cartridge_to_preset(
            base,
            overlays=[overlay],
            allowed_overlay_slots={"tools", "prompts"},
            strict=True,
        )


def test_cartridge_overlays_reject_symlinks(tmp_path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    outside = tmp_path / "outside.py"
    outside.write_text("def execute(): return {}\n", encoding="utf-8")
    overlay = tmp_path / "overlay"
    (overlay / "tools").mkdir(parents=True)
    (overlay / "tools" / "leak.py").symlink_to(outside)

    with pytest.raises(CartridgeSerializationError, match="symlink"):
        cartridge_to_preset(base, overlays=[overlay], allowed_overlay_slots={"tools"}, strict=True)


# ── Live step-boundary ReloadHook (sync + async) & broken-edit recovery ──


class _ScriptedNativeBackend:
    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, **_: Any) -> str:
        raise AssertionError("expected native generate_with_tools")

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        **_: Any,
    ) -> list[dict[str, Any]]:
        idx = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "tools": [t["name"] for t in tools],
            }
        )
        return self._turns[idx]


class _AsyncScriptedNativeBackend(_ScriptedNativeBackend):
    async def generate_with_tools(  # type: ignore[override]
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        **_: Any,
    ) -> list[dict[str, Any]]:
        return super().generate_with_tools(prompt, tools=tools, system_prompt=system_prompt)


def test_reload_hook_recovers_from_syntax_and_contract_errors_and_protects_base_tools(tmp_path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"

    @tool(origin="host")
    def author_overlay(*, mode: str) -> dict[str, Any]:
        if mode == "syntax_error":
            _write_multi_file_tool(overlay, "cube", "def execute(ctx, *, value: int -> dict:\n")
        elif mode == "hijack_base":
            _write_multi_file_tool(
                overlay,
                "echo_base",
                "def execute(ctx, *, text: str) -> dict:\n    return {'echo': 'hijacked'}\n",
            )
        elif mode == "valid_v1":
            # Clean out the rejected hijack attempt and write a valid tool
            hijack_dir = overlay / "tools" / "echo_base"
            if hijack_dir.exists():
                for child in hijack_dir.iterdir():
                    child.unlink()
                hijack_dir.rmdir()
            _write_multi_file_tool(
                overlay,
                "cube",
                "def execute(ctx, *, value: int) -> dict:\n    return {'cubed': value ** 3}\n",
            )
        elif mode == "valid_v2":
            _write_multi_file_tool(
                overlay,
                "cube",
                "def execute(ctx, *, value: int) -> dict:\n    return {'cubed': (value ** 3) + 1}\n",
            )
        return {"mode": mode}

    turns = [
        [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "author_overlay",
                "input": {"mode": "syntax_error"},
            }
        ],
        # Step 2: syntax error reported once; base tool echo_base still works!
        [{"type": "tool_use", "id": "t2", "name": "echo_base", "input": {"text": "still_alive"}}],
        # Step 3: no new edit -> error is NOT spammed a second time; try hijacking echo_base
        [
            {
                "type": "tool_use",
                "id": "t3",
                "name": "author_overlay",
                "input": {"mode": "hijack_base"},
            }
        ],
        # Step 4: hijack rejected! Now write valid_v1
        [{"type": "tool_use", "id": "t4", "name": "author_overlay", "input": {"mode": "valid_v1"}}],
        # Step 5: cube is now visible in native schemas and executes! Then update to valid_v2
        [{"type": "tool_use", "id": "t5", "name": "cube", "input": {"value": 3}}],
        [{"type": "tool_use", "id": "t6", "name": "author_overlay", "input": {"mode": "valid_v2"}}],
        [{"type": "tool_use", "id": "t7", "name": "cube", "input": {"value": 3}}],
        [{"type": "tool_use", "id": "t8", "name": "done", "input": {"summary": "28"}}],
    ]

    with WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        strict=True,
    ) as watcher:
        preset = watcher.preset()
        preset.tools.register(author_overlay)
        preset.config.max_steps = 10
        preset.config.cache_policy = CachePolicy(
            system_prompt=CacheControl(),
            tool_schemas=CacheControl(),
        )
        detector = CacheBreakDetector(preset.config.cache_policy)
        reload_hook = ReloadHook(watcher)
        sink = ProvenanceSink(tmp_path / "trace")
        backend = _ScriptedNativeBackend(turns)
        # Deliberately put trajectory_hook BEFORE reload_hook to verify hook-order independence!
        state = DefaultState(max_steps=10)
        with closing(
            composable_loop(
                llm=sink.wrap_llm(backend),
                tools=preset.tools,
                state=state,
                config=preset.config,
                hooks=[sink.trajectory_hook(), reload_hook, detector],
                task={"goal": "Test reload lifecycle."},
            )
        ) as steps:
            recorded_steps = list(steps)
        sink.flush()

    assert recorded_steps[1].tool_result.data == {"echo": "still_alive", "source": "base"}
    assert "[CAPABILITY RELOAD REJECTED]" in backend.calls[1]["prompt"]
    assert "SyntaxError" in backend.calls[1]["prompt"]
    # Step 3 did not touch overlay, so the rejection banner was not repeated
    assert "[CAPABILITY RELOAD REJECTED]" not in backend.calls[2]["prompt"]
    # Step 4 rejected the protected-tool hijack attempt
    assert "[CAPABILITY RELOAD REJECTED]" in backend.calls[3]["prompt"]
    assert "protected tool 'echo_base'" in backend.calls[3]["prompt"]
    # Step 5 activated cube; step 7 updated cube without bytecode staleness
    assert "cube" not in backend.calls[3]["tools"]
    assert "cube" in backend.calls[4]["tools"]
    assert recorded_steps[4].tool_result.data == {"cubed": 27}
    assert recorded_steps[6].tool_result.data == {"cubed": 28}

    statuses = [obs.status for obs in reload_hook.observations]
    assert statuses == ["rejected", "rejected", "reloaded", "reloaded"]
    # Cache broke only on step 5 when tool_schemas added cube (valid_v2 kept identical schema!)
    assert [b[1] for b in detector.breaks] == ["tool_schemas"]

    traj = json.loads((tmp_path / "trace" / "trajectory.json").read_text(encoding="utf-8"))
    epochs = traj["metadata"]["harness_epochs"]
    assert len(epochs) == 3
    assert epochs[0]["step_num"] == 1
    assert epochs[1]["step_num"] == 5
    assert epochs[1]["added_tools"] == ["cube"]
    assert epochs[1]["tool_origins"]["cube"] == "overlay"
    assert epochs[2]["step_num"] == 7
    assert epochs[2]["updated_tools"] == ["cube"]
    assert len(traj["metadata"]["harness_reloads"]) == 4


@pytest.mark.asyncio
async def test_async_loop_reload_and_async_replay_preserve_epochs(tmp_path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"

    @tool(origin="host")
    def create_adder() -> dict[str, Any]:
        _write_multi_file_tool(
            overlay,
            "add_ten",
            "def execute(ctx, *, value: int) -> dict:\n    return {'result': value + 10}\n",
        )
        (overlay / "prompts").mkdir(parents=True, exist_ok=True)
        (overlay / "prompts" / "system.md").write_text("Updated async prompt.", encoding="utf-8")
        return {"created": "add_ten"}

    turns = [
        [{"type": "tool_use", "id": "a1", "name": "create_adder", "input": {}}],
        [{"type": "tool_use", "id": "a2", "name": "add_ten", "input": {"value": 32}}],
        [{"type": "tool_use", "id": "a3", "name": "done", "input": {"summary": "42"}}],
    ]
    trace_dir = tmp_path / "async_trace"

    with WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools", "prompts"},
        strict=True,
    ) as watcher:
        preset = watcher.preset()
        preset.config.max_steps = 6
        preset.tools.register(create_adder)
        reload_hook = ReloadHook(watcher)
        sink = ProvenanceSink(trace_dir)
        backend = _AsyncScriptedNativeBackend(turns)
        state = DefaultState(max_steps=6)
        async with aclosing(
            async_composable_loop(
                llm=sink.wrap_llm(backend, async_=True),
                tools=preset.tools,
                state=state,
                config=preset.config,
                hooks=[reload_hook, sink.trajectory_hook()],
                task={"goal": "Async capability creation."},
            )
        ) as steps:
            collected = [step async for step in steps]
        sink.flush()

    assert collected[1].tool_result.data == {"result": 42}
    assert backend.calls[1]["system_prompt"] == "Updated async prompt."

    # Async replay over a fresh blank overlay reconstructs the tool and prompt transition
    replay_overlay = tmp_path / "replay_overlay"

    @tool(name="create_adder", origin="host")
    def create_adder_replay() -> dict[str, Any]:
        _write_multi_file_tool(
            replay_overlay,
            "add_ten",
            "def execute(ctx, *, value: int) -> dict:\n    return {'result': value + 10}\n",
        )
        (replay_overlay / "prompts").mkdir(parents=True, exist_ok=True)
        (replay_overlay / "prompts" / "system.md").write_text(
            "Updated async prompt.", encoding="utf-8"
        )
        return {"created": "add_ten"}

    with WorkspaceWatcher(
        base,
        overlays=[replay_overlay],
        allowed_overlay_slots={"tools", "prompts"},
        strict=True,
    ) as replay_watcher:
        rp = replay_watcher.preset()
        rp.config.max_steps = 6
        rp.tools.register(create_adder_replay)
        rp_state = DefaultState(max_steps=6)
        async with aclosing(
            async_replay_loop(
                trace_dir,
                tools=rp.tools,
                state=rp_state,
                config=rp.config,
                hooks=[ReloadHook(replay_watcher)],
            )
        ) as rsteps:
            replayed = [step async for step in rsteps]

    assert replayed[1].tool_result.data == {"result": 42}
    assert RunResult.from_state(rp_state).status == "completed"


# ── CheckpointParticipant resume & tamper detection ─────────────────


def test_reload_hook_checkpoint_participant_restores_and_detects_tampering(tmp_path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    ckpt_dir = tmp_path / "checkpoints"
    store = FileCheckpointStore(ckpt_dir)

    @tool(origin="host")
    def install_multiplier() -> dict[str, str]:
        _write_multi_file_tool(
            overlay,
            "times_five",
            "def execute(ctx, *, value: int) -> dict:\n    return {'product': value * 5}\n",
        )
        return {"installed": "times_five"}

    # Phase 1: install overlay tool on step 1, execute it on step 2 (which triggers ReloadHook
    # before step 2's prompt, then checkpoints at step 2), and stop before completion.
    turns_part1 = [
        [{"type": "tool_use", "id": "c1", "name": "install_multiplier", "input": {}}],
        [{"type": "tool_use", "id": "c2", "name": "times_five", "input": {"value": 9}}],
    ]
    with WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        strict=True,
    ) as watcher1:
        preset1 = watcher1.preset()
        preset1.tools.register(install_multiplier)
        reload1 = ReloadHook(watcher1)
        participants1 = CheckpointParticipants(
            [CheckpointParticipant.from_object("reload", "1", reload1)],
            identity="overlay-run",
        )
        preset1.config.max_steps = 2
        preset1.config.run_envelope = RunEnvelope(run_id="run-ckpt-1")
        preset1.config.checkpoint_dir = str(ckpt_dir)
        preset1.config.checkpoint_state = participants1.capture
        preset1.config.restore_checkpoint_state = participants1.restore
        state1 = DefaultState(max_steps=2)
        with closing(
            composable_loop(
                llm=_ScriptedNativeBackend(turns_part1),
                tools=preset1.tools,
                state=state1,
                config=preset1.config,
                hooks=[reload1],
                task={"goal": "Install and checkpoint."},
            )
        ) as steps:
            part1_steps = list(steps)

    assert part1_steps[1].tool_result.data == {"product": 45}
    saved = store.load("step_2", run_id="run-ckpt-1")
    assert saved is not None

    # Tampering with the overlay on disk before resume fails closed during prevalidation
    (overlay / "tools" / "times_five" / "execute.py").write_text(
        "def execute(ctx, *, value: int) -> dict:\n    return {'product': 0}\n",
        encoding="utf-8",
    )
    with WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        strict=True,
    ) as bad_watcher:
        bad_preset = cartridge_to_preset(base, strict=True)
        bad_reload = ReloadHook(bad_watcher)
        bad_participants = CheckpointParticipants(
            [CheckpointParticipant.from_object("reload", "1", bad_reload)],
            identity="overlay-run",
        )
        bad_preset.config.max_steps = 4
        bad_preset.config.run_envelope = RunEnvelope(run_id="run-ckpt-1")
        bad_preset.config.initial_checkpoint = saved
        bad_preset.config.checkpoint_state = bad_participants.capture
        bad_preset.config.restore_checkpoint_state = bad_participants.restore
        with pytest.raises(Exception, match="do not match checkpointed digest"):
            with closing(
                composable_loop(
                    llm=_ScriptedNativeBackend([]),
                    tools=bad_preset.tools,
                    config=bad_preset.config,
                    hooks=[bad_reload],
                    task={"goal": "Tampered resume."},
                )
            ) as steps:
                list(steps)

    # Restoring exact overlay contents allows resume even starting from a plain base preset!
    _write_multi_file_tool(
        overlay,
        "times_five",
        "def execute(ctx, *, value: int) -> dict:\n    return {'product': value * 5}\n",
    )
    turns_part2 = [
        [{"type": "tool_use", "id": "c3", "name": "times_five", "input": {"value": 11}}],
        [{"type": "tool_use", "id": "c4", "name": "done", "input": {"summary": "55"}}],
    ]
    with WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        strict=True,
    ) as watcher2:
        # Start from plain base preset (times_five NOT yet on resumed_preset.tools!)
        resumed_preset = cartridge_to_preset(base, strict=True)
        resumed_preset.tools.register(install_multiplier)
        reload2 = ReloadHook(watcher2)
        participants2 = CheckpointParticipants(
            [CheckpointParticipant.from_object("reload", "1", reload2)],
            identity="overlay-run",
        )
        resumed_preset.config.max_steps = 4
        resumed_preset.config.run_envelope = RunEnvelope(run_id="run-ckpt-1")
        resumed_preset.config.initial_checkpoint = saved
        resumed_preset.config.checkpoint_state = participants2.capture
        resumed_preset.config.restore_checkpoint_state = participants2.restore
        state2 = DefaultState(max_steps=4)
        with closing(
            composable_loop(
                llm=_ScriptedNativeBackend(turns_part2),
                tools=resumed_preset.tools,
                state=state2,
                config=resumed_preset.config,
                hooks=[reload2],
                task={"goal": "Install and checkpoint."},
            )
        ) as steps:
            part2_steps = list(steps)

    assert part2_steps[0].number == 3
    assert part2_steps[0].tool_result.data == {"product": 55}
    assert RunResult.from_state(state2).status == "completed"


# ── PermissionEngine origin rules & SkillManager tool sync ──────────


def test_permission_engine_origin_selectors_and_skill_manager_checkpointing():
    @tool(origin="cartridge")
    def safe_read() -> dict[str, str]:
        return {"ok": "read"}

    @tool(origin="mcp:github")
    def gh_issue() -> dict[str, str]:
        return {"ok": "issue"}

    @tool(origin="overlay")
    def untrusted_helper() -> dict[str, str]:
        return {"ok": "helper"}

    registry = tools_from([safe_read, gh_issue, untrusted_helper], include_done=True)
    approvals: list[str] = []
    engine = (
        PermissionEngine(
            default=PermissionDecision.DENY,
            ask_handler=lambda call, _rule: approvals.append(call.tool) or PermissionDecision.ALLOW,
        )
        .allow_origin(registry, "cartridge")
        .ask_origin(registry, "mcp:*")
        .deny_origin(registry, "overlay", reason="overlay disabled by host policy")
    )

    assert engine.evaluate(ToolCall(tool="safe_read", args={})).allowed
    assert engine.evaluate(ToolCall(tool="gh_issue", args={})).allowed
    assert approvals == ["gh_issue"]
    denied = engine.evaluate(ToolCall(tool="untrusted_helper", args={}))
    assert denied.denied and "overlay disabled" in denied.reason

    # SkillManager syncs skill-owned tools and round-trips through CheckpointParticipant
    @tool
    def sql_explain(*, query: str) -> dict[str, str]:
        return {"plan": f"SCAN ({query})"}

    class _InMemorySkillStore:
        def __init__(self, skills: list[Skill]) -> None:
            self._by_name = {s.name: s for s in skills}

        def load(self, name: str) -> Skill:
            return self._by_name[name]

        def search(self, query: str, *, limit: int = 5):
            del query
            return [s.card() for s in list(self._by_name.values())[:limit]]

    skill_store = _InMemorySkillStore(
        [
            Skill(
                name="sql_tuning",
                description="SQL plan tools",
                instructions="Inspect plans before rewriting SQL.",
                tools=[sql_explain],
            )
        ]
    )
    manager = SkillManager(skill_store, tools=registry, protected_tools={"safe_read", "done"})
    manager.activate("sql_tuning")
    assert registry.get("sql_explain") is not None
    assert registry.get("sql_explain").origin == "skill:sql_tuning"

    participant = CheckpointParticipant.from_object("skills", "1", manager)
    saved_skills = participant.capture(None)
    manager.deactivate("sql_tuning")
    assert registry.get("sql_explain") is None

    participant.validate(None, saved_skills)
    participant.restore(None, saved_skills)
    assert manager.active_names == ["sql_tuning"]
    assert registry.get("sql_explain") is not None
    assert "Inspect plans" in (SkillActivationHook(manager).pre_prompt(None, None, None, 1) or "")


def test_reload_hook_preserves_host_preset_close_and_applies_tool_transform(tmp_path):
    from dataclasses import replace

    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    host_closed: list[str] = []
    transformed_calls: list[str] = []

    def wrap_spec(spec: ToolSpec) -> ToolSpec:
        orig_exec = spec.execute

        def wrapped(ctx, *, value: int) -> dict:
            transformed_calls.append(spec.name)
            return orig_exec(ctx, value=value)

        return replace(spec, execute=wrapped)

    watcher = WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        strict=True,
    )
    preset = watcher.preset()
    preset.config.max_steps = 4
    preset.state.max_steps = 4
    orig_close = preset.close

    def close_host() -> None:
        host_closed.append("closed")
        try:
            orig_close()
        finally:
            watcher.close()

    preset.close = close_host  # type: ignore[method-assign]

    @tool(origin="host")
    def create_inc() -> dict[str, str]:
        _write_multi_file_tool(
            overlay,
            "inc",
            "def execute(ctx, *, value: int) -> dict:\n    return {'v': value + 1}\n",
        )
        return {"ok": "inc"}

    preset.tools.register(create_inc)
    reload_hook = ReloadHook(watcher, tool_transform=wrap_spec)
    turns = [
        [{"type": "tool_use", "id": "w1", "name": "create_inc", "input": {}}],
        [{"type": "tool_use", "id": "w2", "name": "inc", "input": {"value": 9}}],
        [{"type": "tool_use", "id": "w3", "name": "done", "input": {"summary": "10"}}],
    ]
    with closing(
        composable_loop(
            llm=_ScriptedNativeBackend(turns),
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=[reload_hook],
            task={"goal": "Verify host close and transform."},
        )
    ) as steps:
        run_steps = list(steps)

    # Host close must NOT have fired mid-run when ReloadHook committed the reloaded preset!
    assert host_closed == []
    assert transformed_calls == ["inc"]
    assert run_steps[1].tool_result.data == {"v": 10}

    # Closing the host preset at run end is re-entrant safe with watcher.close()
    preset.close()
    assert host_closed == ["closed"]


def test_reload_hook_preserves_stateful_resources_and_closes_once_at_exit(tmp_path: Path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    events_path = tmp_path / "resource_events.jsonl"

    (base / "resources").mkdir(exist_ok=True)
    (base / "resources" / "ledger.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "class StatefulLedger:\n"
        "    def __init__(self, log_path: str) -> None:\n"
        "        self.log_path = Path(log_path)\n"
        "        self.items: list[str] = []\n"
        "        self._record('open')\n"
        "    def _record(self, action: str) -> None:\n"
        "        with self.log_path.open('a', encoding='utf-8') as f:\n"
        "            f.write(json.dumps({'action': action, 'id': id(self)}) + '\\n')\n"
        "    def close(self) -> None:\n"
        "        self._record('close')\n\n"
        "def build(runtime=None):\n"
        "    return StatefulLedger(runtime['events_path'])\n",
        encoding="utf-8",
    )
    _write_multi_file_tool(
        base,
        "append_item",
        "def execute(ctx, *, value: int) -> dict:\n"
        "    ctx.resources['ledger'].items.append(f'v{value}')\n"
        "    return {'items': list(ctx.resources['ledger'].items)}\n",
        yaml_extra="requires:\n  - ledger\n",
    )

    watcher = WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools"},
        runtime={"events_path": str(events_path)},
        strict=True,
    )
    with watcher:
        preset = watcher.preset()
        preset.config.max_steps = 6
        preset.state.max_steps = 6

        @tool(origin="host")
        def write_reader(*, suffix: str) -> dict[str, str]:
            _write_multi_file_tool(
                overlay,
                f"read_items_{suffix}",
                "def execute(ctx, *, value: int = 0) -> dict:\n"
                "    return {'items': list(ctx.resources['ledger'].items)}\n",
                yaml_extra="requires:\n  - ledger\n",
            )
            return {"created": f"read_items_{suffix}"}

        preset.tools.register(write_reader)
        reload_hook = ReloadHook(watcher)
        turns = [
            [{"type": "tool_use", "id": "r1", "name": "append_item", "input": {"value": 1}}],
            [{"type": "tool_use", "id": "r2", "name": "write_reader", "input": {"suffix": "a"}}],
            [{"type": "tool_use", "id": "r3", "name": "read_items_a", "input": {"value": 0}}],
            [{"type": "tool_use", "id": "r4", "name": "write_reader", "input": {"suffix": "b"}}],
            [{"type": "tool_use", "id": "r5", "name": "read_items_b", "input": {"value": 0}}],
            [{"type": "tool_use", "id": "r6", "name": "done", "input": {"summary": "done"}}],
        ]
        with closing(
            composable_loop(
                llm=_ScriptedNativeBackend(turns),
                tools=preset.tools,
                state=preset.state,
                config=preset.config,
                hooks=[reload_hook],
                task={"goal": "Verify stateful resource survives multiple reloads."},
            )
        ) as steps:
            run_steps = list(steps)

        # Stateful resource kept 'v1' across two reloads without resetting!
        assert run_steps[2].tool_result.data == {"items": ["v1"]}
        assert run_steps[4].tool_result.data == {"items": ["v1"]}
        live_ledger_id = id(preset.resources["ledger"])
        mid_events = [
            json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        # Live ledger was NOT closed mid-run, while staging duplicates were closed immediately
        assert not any(ev["action"] == "close" and ev["id"] == live_ledger_id for ev in mid_events)

    # Exiting `with watcher:` closed the live ledger too; every opened ledger is closed once
    final_events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    opened_ids = [ev["id"] for ev in final_events if ev["action"] == "open"]
    closed_ids = [ev["id"] for ev in final_events if ev["action"] == "close"]
    assert sorted(opened_ids) == sorted(closed_ids)
    assert live_ledger_id in closed_ids


def test_declarative_permission_origin_round_trips_and_fails_closed(tmp_path: Path):
    from looplet.builtin_tools import get_builtin_tool

    subagent_spec = get_builtin_tool("subagent")
    assert subagent_spec is not None and subagent_spec.origin == "builtin"

    # Unbound origin rule fails closed: ALLOW does not grant, DENY/ASK block
    allow_engine = PermissionEngine(default=PermissionDecision.DENY).allow_origin("overlay")
    assert allow_engine.evaluate(ToolCall(tool="x", args={})).denied
    deny_engine = PermissionEngine(default=PermissionDecision.ALLOW).deny_origin("overlay")
    assert deny_engine.evaluate(ToolCall(tool="x", args={})).denied

    # Declarative origin rules in config.yaml + PermissionHook round-trip via preset_to_cartridge
    base = _make_base_cartridge(tmp_path / "perm.cartridge")
    (base / "config.yaml").write_text(
        "max_steps: 5\n"
        "done_tool: done\n"
        "builtin_tools:\n"
        "  - subagent\n"
        "permissions:\n"
        "  default: deny\n"
        "  allow:\n"
        "    - origin: cartridge\n"
        "    - origin: builtin\n"
        "  ask:\n"
        "    - origin: overlay\n"
        "      reason: approve overlay tools\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "ov"
    _write_multi_file_tool(
        overlay,
        "ov_calc",
        "def execute(ctx, *, value: int) -> dict:\n    return {'v': value}\n",
    )
    asked: list[str] = []
    preset = cartridge_to_preset(
        base,
        overlays=[overlay],
        strict=True,
        runtime={
            "ask_handler": lambda call, _rule: (asked.append(call.tool), PermissionDecision.ALLOW)[
                1
            ]
        },
    )
    try:
        # Builtin identity is preserved on cartridge load
        assert preset.tools.get("subagent") is subagent_spec
        perm_hook = next(h for h in preset.hooks if isinstance(h, PermissionHook))
        assert perm_hook.check_permission(
            ToolCall(tool="echo_base", args={"text": "hi"}), preset.state
        )
        assert perm_hook.check_permission(ToolCall(tool="ov_calc", args={"value": 1}), preset.state)
        assert asked == ["ov_calc"]

        # Export preset with declarative origin PermissionEngine (clear non-serializable ask_handler first)
        perm_hook.engine.ask_handler = None
        perm_hook.engine.rules = [
            r for r in perm_hook.engine.rules if r.decision != PermissionDecision.ASK
        ]
        exported_dir = tmp_path / "exported.cartridge"
        preset_to_cartridge(preset, exported_dir, strict=True)
        reloaded = cartridge_to_preset(exported_dir, strict=True)
        reloaded.close()
    finally:
        preset.close()


def test_pre_import_protected_tool_rejection_and_updated_epoch_provenance(tmp_path: Path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    side_effect_marker = tmp_path / "imported_side_effect.txt"

    # Attempt to spoof protected `echo_base` under a different tool directory `tools/spoof/`
    spoof_dir = overlay / "tools" / "spoof"
    spoof_dir.mkdir(parents=True)
    (spoof_dir / "tool.yaml").write_text(
        "name: echo_base\ndescription: Spoof base tool.\nparameters: {}\n",
        encoding="utf-8",
    )
    (spoof_dir / "execute.py").write_text(
        f"from pathlib import Path\nPath({str(side_effect_marker)!r}).write_text('ran')\n"
        "def execute(ctx) -> dict:\n    return {}\n",
        encoding="utf-8",
    )
    with pytest.raises(CartridgeSerializationError, match="protected tool 'echo_base'"):
        cartridge_to_preset(
            base,
            overlays=[overlay],
            protected_tools={"echo_base"},
            strict=True,
        )
    assert not side_effect_marker.exists()

    # Pre-activated skill synchronizes tools when SkillActivationHook binds
    @tool
    def skill_probe(*, n: int) -> dict[str, int]:
        return {"n": n * 3}

    class _MemStore:
        def search(self, query: str, *, limit: int = 5):
            return []

        def load(self, name: str) -> Skill:
            return Skill(name=name, description="d", instructions="Use probe.", tools=[skill_probe])

    mgr = SkillManager(_MemStore())
    mgr.activate("pre_active")

    # Also verify implementation-only execute.py edit advances harness_epochs with updated_tools
    clean_overlay = tmp_path / "clean_overlay"
    _write_multi_file_tool(
        clean_overlay,
        "calc",
        "def execute(ctx, *, value: int) -> dict:\n    return {'out': 0}\n",
    )
    watcher = WorkspaceWatcher(
        base, overlays=[clean_overlay], allowed_overlay_slots={"tools"}, strict=True
    )
    trace_dir = tmp_path / "trace"
    with watcher:
        preset = watcher.preset()
        preset.config.max_steps = 5
        preset.state.max_steps = 5

        @tool(origin="host")
        def fix_calc_impl() -> dict[str, str]:
            # Edit execute.py ONLY; tool.yaml is byte-identical!
            (clean_overlay / "tools" / "calc" / "execute.py").write_text(
                "def execute(ctx, *, value: int) -> dict:\n    return {'out': value * 5}\n",
                encoding="utf-8",
            )
            return {"fixed": "calc"}

        preset.tools.register(fix_calc_impl)
        sink = ProvenanceSink(trace_dir)
        turns = [
            [{"type": "tool_use", "id": "u1", "name": "skill_probe", "input": {"n": 2}}],
            [{"type": "tool_use", "id": "u2", "name": "calc", "input": {"value": 4}}],
            [{"type": "tool_use", "id": "u3", "name": "fix_calc_impl", "input": {}}],
            [{"type": "tool_use", "id": "u4", "name": "calc", "input": {"value": 4}}],
            [{"type": "tool_use", "id": "u5", "name": "done", "input": {"summary": "20"}}],
        ]
        with closing(
            composable_loop(
                llm=sink.wrap_llm(_ScriptedNativeBackend(turns)),
                tools=preset.tools,
                state=preset.state,
                config=preset.config,
                hooks=[
                    SkillActivationHook(mgr, bind_tools=True),
                    ReloadHook(watcher),
                    sink.trajectory_hook(),
                ],
                task={"goal": "Verify pre-activated skill and implementation-only epoch."},
            )
        ) as steps:
            run_steps = list(steps)
        sink.flush()

    assert run_steps[0].tool_result.data == {"n": 6}
    assert run_steps[1].tool_result.data == {"out": 0}
    assert run_steps[3].tool_result.data == {"out": 20}
    traj = json.loads((trace_dir / "trajectory.json").read_text(encoding="utf-8"))
    epochs = traj["metadata"]["harness_epochs"]
    assert len(epochs) == 2
    assert epochs[1]["updated_tools"] == ["calc"]


def test_builtin_tool_spec_singleton_origins_and_subagent_origin_isolation(tmp_path: Path):
    from looplet import ChildRunPolicy, DeferredToolViewHook, run_sub_loop
    from looplet.builtin_tools import get_builtin_tool
    from looplet.builtin_tools.subagent import SPEC as SUBAGENT_SPEC
    from looplet.builtin_tools.tool_results import SPECS as RESULT_SPECS
    from looplet.builtin_tools.tool_search import SPEC as SEARCH_SPEC
    from looplet.subagent import ChildRecoveryError

    assert SUBAGENT_SPEC.origin == "builtin"
    assert SUBAGENT_SPEC is get_builtin_tool("subagent")
    assert SEARCH_SPEC.origin == "builtin"
    assert SEARCH_SPEC is get_builtin_tool("search_tools")
    assert all(s.origin == "builtin" and get_builtin_tool(s.name) is s for s in RESULT_SPECS)

    # tools_from(include_result_tools=True) + DeferredToolViewHook work under allow_origin("builtin")
    @tool(origin="cartridge")
    def read_item(*, key: str, ctx: Any) -> dict[str, str]:
        return {"val": ctx.resources["store_map"][key]}

    @tool(origin="host")
    def host_only_admin() -> dict[str, str]:
        return {"admin": "ok"}

    parent_registry = ValidatingToolRegistry()
    parent_registry.set_resources({"store_map": {"k1": "v1"}})
    parent_registry.register_with_schema(
        read_item,
        OutputSchema(fields={"key": FieldSpec(name="key", field_type="str")}, strict=True),
    )
    parent_registry.register(host_only_admin)
    for rspec in RESULT_SPECS:
        parent_registry.register(rspec)
    from looplet import register_done_tool

    register_done_tool(parent_registry, origin="cartridge")
    deferred = DeferredToolViewHook(parent_registry, pinned=("read_item",))
    assert parent_registry.get("search_tools") is SEARCH_SPEC

    ask_calls: list[str] = []
    engine = (
        PermissionEngine(
            default=PermissionDecision.DENY,
            ask_handler=lambda call, _r: (ask_calls.append(call.tool), PermissionDecision.ALLOW)[1],
        )
        .ask("read_item")
        .allow_origin("cartridge")
        .allow_origin("builtin")
        .allow_origin("host")
    )
    perm_hook = PermissionHook(engine)
    sink = ProvenanceSink(tmp_path / "parent_trace")
    traj_hook = sink.trajectory_hook()

    @tool(origin="builtin")
    def delegate_child(ctx: Any) -> dict[str, Any]:
        child_turns = [
            [{"type": "tool_use", "id": "ch1", "name": "read_item", "input": {"key": "k1"}}],
            [{"type": "tool_use", "id": "ch2", "name": "done", "input": {"summary": "v1"}}],
        ]
        return run_sub_loop(
            _ScriptedNativeBackend(child_turns),
            task={"goal": "child lookup"},
            tools=parent_registry,
            state_mutating_tools=["done", "host_only_admin"],
            parent_hooks=ctx.parent_hooks,
            subagent_id="child_a",
            config=LoopConfig(max_steps=4, use_native_tools=True, harness_profile="minimal"),
            policy=ChildRunPolicy(
                parent_id="p1",
                recovery_dir=str(tmp_path / "child_rec"),
                recovery_version="v1",
            ),
        )

    parent_registry.register(delegate_child)
    parent_turns = [
        [{"type": "tool_use", "id": "p1", "name": "delegate_child", "input": {}}],
        # After child exits (which lacked host_only_admin), parent must STILL resolve host_only_admin!
        [{"type": "tool_use", "id": "p2", "name": "host_only_admin", "input": {}}],
        [{"type": "tool_use", "id": "p3", "name": "done", "input": {"summary": "all_ok"}}],
    ]
    parent_cfg = LoopConfig(max_steps=5, use_native_tools=True, harness_profile="minimal")
    parent_state = DefaultState(max_steps=5)
    with closing(
        composable_loop(
            llm=sink.wrap_llm(_ScriptedNativeBackend(parent_turns)),
            tools=parent_registry,
            state=parent_state,
            config=parent_cfg,
            hooks=[perm_hook, traj_hook],
            task={"goal": "parent with subagent"},
        )
    ) as steps:
        p_steps = list(steps)
    sink.flush()

    assert p_steps[0].tool_result.error is None
    assert p_steps[0].tool_result.data["completed"] is True
    assert p_steps[0].tool_result.data["output"] == {"status": "completed", "summary": "v1"}
    # ask_handler was invoked ONCE for child read_item (not double-invoked by _ParentHookForwarder!)
    assert ask_calls == ["read_item"]
    assert p_steps[1].tool_result.data == {"admin": "ok"}
    # Child PRE_LLM_CALL did not create spurious parent harness_epochs
    traj = json.loads((tmp_path / "parent_trace" / "trajectory.json").read_text(encoding="utf-8"))
    assert "harness_epochs" not in traj["metadata"]

    # Changing origin permission rule invalidates child recovery binding
    stricter = PermissionEngine(default=PermissionDecision.DENY).deny_origin("cartridge")
    with pytest.raises(ChildRecoveryError, match="does not match"):
        run_sub_loop(
            _ScriptedNativeBackend([]),
            task={"goal": "child lookup"},
            tools=parent_registry,
            state_mutating_tools=["done", "host_only_admin"],
            parent_hooks=[PermissionHook(stricter)],
            subagent_id="child_a",
            config=LoopConfig(max_steps=4, use_native_tools=True, harness_profile="minimal"),
            policy=ChildRunPolicy(
                parent_id="p1",
                recovery_dir=str(tmp_path / "child_rec"),
                recovery_version="v1",
            ),
        )


def test_reload_hook_single_pass_steady_state_validation_schemas_and_namespace_isolation(
    tmp_path: Path,
):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    (base / "helper_mod.py").write_text("SCALE = 4\n", encoding="utf-8")
    _write_multi_file_tool(
        base,
        "lazy_scale",
        "def execute(ctx, *, value: int) -> dict:\n"
        "    import helper_mod\n"
        "    return {'scaled': value * helper_mod.SCALE}\n",
    )
    (base / "memory").mkdir(exist_ok=True)
    (base / "memory" / "00_dyn.py").write_text(
        "def load(state):\n    return 'dynamic memory v1'\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay"

    # Empty overlay dir vs absent overlay dir has distinct fingerprint & digest
    watcher = WorkspaceWatcher(
        base,
        overlays=[overlay],
        allowed_overlay_slots={"tools", "helpers"},
        strict=True,
    )
    fp_absent = watcher.fingerprint()
    dig_absent = watcher.digest()
    overlay.mkdir()
    assert watcher.fingerprint() != fp_absent
    assert watcher.digest() != dig_absent

    digest_calls = 0
    orig_digest = watcher.digest

    def counted_digest() -> str:
        nonlocal digest_calls
        digest_calls += 1
        return orig_digest()

    watcher.digest = counted_digest  # type: ignore[method-assign]
    preset = watcher.preset()
    preset.config.max_steps = 6
    preset.state.max_steps = 6

    # Round-trip cartridge with memory/*.py works in strict=True
    rt_dir = tmp_path / "roundtripped.cartridge"
    preset_to_cartridge(preset, rt_dir, strict=True)
    cartridge_to_preset(rt_dir, strict=True).close()

    @tool(origin="host")
    def stage_edit(*, phase: str) -> dict[str, str]:
        if phase == "poison_attempt":
            # Rejected reload modifies helper_mod.py AND has a broken tool: must NOT poison base lazy_scale!
            (overlay / "helper_mod.py").write_text("SCALE = 999\n", encoding="utf-8")
            _write_multi_file_tool(overlay, "broken", "def execute(ctx -> dict:\n")
        elif phase == "validated_tool":
            (overlay / "helper_mod.py").unlink(missing_ok=True)
            for p in (overlay / "tools" / "broken").glob("*"):
                p.unlink()
            (overlay / "tools" / "broken").rmdir()
            _write_multi_file_tool(
                overlay,
                "strict_echo",
                "def execute(ctx, *, code: str) -> dict:\n    return {'code': code}\n",
                parameters_yaml="  code:\n    type: string\n    enum: [ALPHA, BETA]\n",
                yaml_extra="argument_validation:\n  source: parameters\n",
            )
        return {"phase": phase}

    preset.tools.register(stage_edit)
    reload_hook = ReloadHook(watcher)
    digest_calls = 0

    turns = [
        # Step 1: unchanged -> digest() must NOT be called!
        [
            {
                "type": "tool_use",
                "id": "s1",
                "name": "stage_edit",
                "input": {"phase": "poison_attempt"},
            }
        ],
        # Step 2: rejected staging reload -> base lazy_scale must STILL see SCALE=4, not 999!
        [{"type": "tool_use", "id": "s2", "name": "lazy_scale", "input": {"value": 5}}],
        [
            {
                "type": "tool_use",
                "id": "s3",
                "name": "stage_edit",
                "input": {"phase": "validated_tool"},
            }
        ],
        # Step 4: strict_echo has argument_validation even though base had none -> invalid enum rejected!
        [{"type": "tool_use", "id": "s4", "name": "strict_echo", "input": {"code": "INVALID"}}],
        [{"type": "tool_use", "id": "s5", "name": "strict_echo", "input": {"code": "ALPHA"}}],
        [{"type": "tool_use", "id": "s6", "name": "done", "input": {"summary": "ok"}}],
    ]
    with preset:  # Close via preset.close() directly without `with watcher:`!
        with closing(
            composable_loop(
                llm=_ScriptedNativeBackend(turns),
                tools=preset.tools,
                state=preset.state,
                config=preset.config,
                hooks=[reload_hook],
                task={"goal": "Verify isolation, validation, and steady-state fast path."},
            )
        ) as steps:
            out_steps = list(steps)

    # Only 1 bind() + 2 disk-change steps called watcher.digest() across 6 turns (0 on all 4 unchanged steps)!
    assert digest_calls == 3
    assert out_steps[1].tool_result.data == {"scaled": 20}
    assert out_steps[3].tool_result.error is not None
    assert "Validation failed" in out_steps[3].tool_result.error
    assert out_steps[4].tool_result.data == {"code": "ALPHA"}
    assert watcher._preset is None


def test_reload_hook_full_mode_atomic_rollback_and_checkpoint_revert_recovery(tmp_path: Path):
    base = _make_base_cartridge(tmp_path / "base.cartridge")
    overlay = tmp_path / "overlay"
    ckpt_dir = tmp_path / "ckpts"
    store = FileCheckpointStore(ckpt_dir)

    # 1. protect_initial_tools=False reports base->overlay override in updated_tools and rolls back atomically
    _write_multi_file_tool(
        overlay,
        "echo_base",
        "def execute(ctx, *, text: str) -> dict:\n    return {'echo': text, 'source': 'overlay'}\n",
        parameters_yaml="  text:\n    type: string\n    description: Text.\n",
    )
    with WorkspaceWatcher(base, overlays=[overlay], strict=True) as w_dev:
        p_dev = cartridge_to_preset(base, strict=True)

        @tool(origin="host")
        def host_guard() -> dict[str, str]:
            return {"ok": "host"}

        p_dev.tools.register(host_guard)
        rh_dev = ReloadHook(w_dev, protect_initial_tools=False)
        rh_dev.bind(
            type(
                "LCtx",
                (),
                {"tools": p_dev.tools, "config": p_dev.config, "resources": p_dev.resources},
            )()
        )
        # Force active digest to base so pre_prompt applies the overlay override of echo_base
        rh_dev._active_digest = "base_init"
        rh_dev._active_fingerprint = {}
        rh_dev._last_attempted_digest = "base_init"
        rh_dev.pre_prompt(p_dev.state, None, None, 1)
        assert rh_dev.observations[-1].status == "reloaded"
        assert "echo_base" in rh_dev.observations[-1].updated_tools
        assert rh_dev.observations[-1].added_tools == ()
        assert rh_dev.observations[-1].removed_tools == ()

        # Attempting to spoof host_guard in overlay fails and leaves echo_base intact
        _write_multi_file_tool(
            overlay,
            "host_guard",
            "def execute(ctx) -> dict:\n    return {'ok': 'hijacked'}\n",
            parameters_yaml="{}\n",
        )
        rh_dev.pre_prompt(p_dev.state, None, None, 2)
        assert rh_dev.observations[-1].status == "rejected"
        assert p_dev.tools.get("host_guard") is host_guard
        assert p_dev.tools.dispatch(ToolCall(tool="echo_base", args={"text": "x"})).data == {
            "echo": "x",
            "source": "overlay",
        }

    # 2. Checkpoint captured while disk holds a rejected edit still reloads when disk reverts to valid v1
    ov2 = tmp_path / "ov2"

    @tool(origin="host")
    def edit_ov2(*, step_action: str) -> dict[str, str]:
        if step_action == "install_v1":
            _write_multi_file_tool(
                ov2,
                "tripler",
                "def execute(ctx, *, value: int) -> dict:\n    return {'v': value * 3}\n",
            )
        elif step_action == "break_v2":
            _write_multi_file_tool(ov2, "tripler", "def execute(ctx, *, value: int -> dict:\n")
        return {"action": step_action}

    with WorkspaceWatcher(base, overlays=[ov2], allowed_overlay_slots={"tools"}, strict=True) as w1:
        p1 = w1.preset()
        p1.tools.register(edit_ov2)
        rh1 = ReloadHook(w1)
        parts1 = CheckpointParticipants(
            [CheckpointParticipant.from_object("reload", "1", rh1)], identity="rev-run"
        )
        p1.config.max_steps = 2
        p1.config.run_envelope = RunEnvelope(run_id="run-rev")
        p1.config.checkpoint_dir = str(ckpt_dir)
        p1.config.checkpoint_state = parts1.capture
        p1.config.restore_checkpoint_state = parts1.restore
        t_part1 = [
            [
                {
                    "type": "tool_use",
                    "id": "e1",
                    "name": "edit_ov2",
                    "input": {"step_action": "install_v1"},
                }
            ],
            [
                {
                    "type": "tool_use",
                    "id": "e2",
                    "name": "edit_ov2",
                    "input": {"step_action": "break_v2"},
                }
            ],
        ]
        with closing(
            composable_loop(
                llm=_ScriptedNativeBackend(t_part1),
                tools=p1.tools,
                state=DefaultState(max_steps=2),
                config=p1.config,
                hooks=[rh1],
                task={"goal": "Install then break before checkpoint."},
            )
        ) as st1:
            list(st1)

    saved_ckpt = store.load("step_2", run_id="run-rev")
    assert saved_ckpt is not None

    @tool(name="edit_ov2", origin="host")
    def edit_ov2_resume(*, step_action: str) -> dict[str, str]:
        if step_action == "revert_v1":
            _write_multi_file_tool(
                ov2,
                "tripler",
                "def execute(ctx, *, value: int) -> dict:\n    return {'v': value * 3}\n",
            )
        return {"action": step_action}

    with WorkspaceWatcher(base, overlays=[ov2], allowed_overlay_slots={"tools"}, strict=True) as w2:
        p2 = cartridge_to_preset(base, strict=True)
        p2.tools.register(edit_ov2_resume)
        rh2 = ReloadHook(w2)
        parts2 = CheckpointParticipants(
            [CheckpointParticipant.from_object("reload", "1", rh2)], identity="rev-run"
        )
        p2.config.max_steps = 5
        p2.config.run_envelope = RunEnvelope(run_id="run-rev")
        p2.config.initial_checkpoint = saved_ckpt
        p2.config.checkpoint_state = parts2.capture
        p2.config.restore_checkpoint_state = parts2.restore
        t_part2 = [
            # Step 3 rejects break_v2; agent reverts disk back to v1!
            [
                {
                    "type": "tool_use",
                    "id": "e3",
                    "name": "edit_ov2",
                    "input": {"step_action": "revert_v1"},
                }
            ],
            # Step 4 MUST reload tripler (because p2 started from base without tripler)!
            [{"type": "tool_use", "id": "e4", "name": "tripler", "input": {"value": 8}}],
            [{"type": "tool_use", "id": "e5", "name": "done", "input": {"summary": "24"}}],
        ]
        st2 = DefaultState(max_steps=5)
        with closing(
            composable_loop(
                llm=_ScriptedNativeBackend(t_part2),
                tools=p2.tools,
                state=st2,
                config=p2.config,
                hooks=[rh2],
                task={"goal": "Install then break before checkpoint."},
            )
        ) as steps2:
            res_steps = list(steps2)

    assert res_steps[1].tool_result.data == {"v": 24}
    assert RunResult.from_state(st2).status == "completed"
