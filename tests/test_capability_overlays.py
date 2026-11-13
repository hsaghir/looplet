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
    assert len(epochs) == 2
    assert epochs[0]["step_num"] == 1
    assert epochs[1]["step_num"] == 5
    assert epochs[1]["added_tools"] == ["cube"]
    assert epochs[1]["tool_origins"]["cube"] == "overlay"
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
