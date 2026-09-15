from __future__ import annotations

from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    CapabilityDeniedError,
    ExecutionPolicy,
    LoopConfig,
    ToolCall,
    ToolContext,
    ToolSpec,
)
from looplet.types import ErrorKind


def test_execution_policy_checks_capabilities_and_paths(tmp_path: Path) -> None:
    policy = ExecutionPolicy(
        workspace_root=str(tmp_path),
        capabilities=frozenset({"workspace.read"}),
    )

    assert policy.allows(["workspace.read"])
    assert policy.missing(["workspace.read", "workspace.write"]) == ("workspace.write",)
    assert policy.resolve_path("nested/file.txt") == (tmp_path / "nested/file.txt").resolve()

    with pytest.raises(CapabilityDeniedError, match="absolute paths"):
        policy.resolve_path(tmp_path / "file.txt")
    with pytest.raises(CapabilityDeniedError, match="escapes workspace"):
        policy.resolve_path("../outside.txt")

    absolute_policy = ExecutionPolicy(
        workspace_root=str(tmp_path),
        capabilities=frozenset({"host.absolute_path"}),
    )
    assert absolute_policy.resolve_path("/tmp/outside.txt") == Path("/tmp/outside.txt")
    assert ExecutionPolicy(environment=frozenset({"SAFE_FLAG"})).allows_environment("SAFE_FLAG")


def test_declared_tool_capability_denial_is_structured() -> None:
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="write_file",
            description="write",
            parameters={},
            execute=lambda: {"ok": True},
            capabilities=["workspace.write"],
        )
    )

    result = registry.dispatch(
        ToolCall(tool="write_file"),
        ctx=ToolContext(
            execution_policy=ExecutionPolicy(capabilities=frozenset({"workspace.read"}))
        ),
    )

    assert result.error_kind == ErrorKind.PERMISSION_DENIED
    assert result.error_detail is not None
    assert result.error_detail.context["missing_capabilities"] == ["workspace.write"]


def test_declared_tool_capability_is_granted() -> None:
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="fetch",
            description="fetch",
            parameters={},
            execute=lambda: {"ok": True},
            capabilities=["network"],
        )
    )

    result = registry.dispatch(
        ToolCall(tool="fetch"),
        ctx=ToolContext(execution_policy=ExecutionPolicy(capabilities=frozenset({"network"}))),
    )

    assert result.error is None
    assert result.data == {"ok": True}


def test_undeclared_tool_preserves_legacy_behavior() -> None:
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(name="legacy", description="legacy", parameters={}, execute=lambda: {"ok": True})
    )

    result = registry.dispatch(
        ToolCall(tool="legacy"),
        ctx=ToolContext(execution_policy=ExecutionPolicy()),
    )

    assert result.error is None
    assert result.data == {"ok": True}


def test_loop_config_policy_reaches_tool_context() -> None:
    received: list[ExecutionPolicy | None] = []
    registry = BaseToolRegistry()

    def inspect(*, ctx: ToolContext) -> dict[str, bool]:
        received.append(ctx.execution_policy)
        return {"present": ctx.execution_policy is not None}

    registry.register(ToolSpec("inspect", "inspect", {}, inspect))
    policy = ExecutionPolicy(capabilities=frozenset({"network"}))
    from looplet.loop import _build_tool_ctx

    ctx = _build_tool_ctx(LoopConfig(execution_policy=policy), tools=registry)
    result = registry.dispatch(ToolCall(tool="inspect"), ctx=ctx)

    assert result.data == {"present": True}
    assert received == [policy]


def test_policy_workspace_is_propagated_to_tool_context(tmp_path: Path) -> None:
    from looplet.loop import _build_tool_ctx

    policy = ExecutionPolicy(workspace_root=str(tmp_path))
    ctx = _build_tool_ctx(LoopConfig(execution_policy=policy))

    assert ctx.workspace_root == str(tmp_path)
    assert ctx.cwd == str(tmp_path)


def test_tool_capabilities_round_trip_through_cartridge(tmp_path) -> None:
    from looplet import (
        AgentPreset,
        DefaultState,
        LoopConfig,
        cartridge_to_preset,
        preset_to_cartridge,
    )

    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="fetch",
            description="fetch",
            parameters={},
            execute=lambda: {"ok": True},
            capabilities=["network", "workspace.read"],
        )
    )
    registry.register(ToolSpec("done", "done", {}, lambda: {"ok": True}))
    preset_to_cartridge(
        AgentPreset(
            config=LoopConfig(max_steps=1),
            hooks=[],
            tools=registry,
            state=DefaultState(max_steps=1),
        ),
        tmp_path / "cartridge",
        strict=False,
    )

    reloaded = cartridge_to_preset(tmp_path / "cartridge", strict=False)
    assert reloaded.tools._tools["fetch"].capabilities == ["network", "workspace.read"]
