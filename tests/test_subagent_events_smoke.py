"""Smoke tests for SUBAGENT_START / SUBAGENT_STOP lifecycle events."""

from __future__ import annotations

import asyncio
import time

import pytest

from looplet import (
    BaseToolRegistry,
    LifecycleEvent,
    LoopConfig,
)
from looplet.checkpoint import Checkpoint
from looplet.subagent import run_sub_loop
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def test_shared_admission_rechecks_deadline_after_lock_acquisition(monkeypatch):
    from looplet.subagent import ChildRunLimitExceeded, SharedModelBudget

    clock = [0.0]

    class AdvancingLock:
        def __enter__(self):
            clock[0] = 2.0

        def __exit__(self, *args):
            return None

    monkeypatch.setattr("looplet.subagent.time.monotonic", lambda: clock[0])
    budget = SharedModelBudget(2, _lock=AdvancingLock())

    with pytest.raises(ChildRunLimitExceeded, match="deadline"):
        budget.charge(deadline=1.0)

    assert budget.remaining == 2


@pytest.mark.parametrize("async_mode", [False, True])
async def test_shared_admission_nested_wrappers_charge_each_allowance_once(async_mode):
    from types import SimpleNamespace

    from looplet.backend_contract import BackendWrapper, single_model_request_active
    from looplet.subagent import SharedModelBudget

    observed = []

    def generate(prompt):
        observed.append(single_model_request_active())
        return prompt

    async def generate_async(prompt):
        await asyncio.sleep(0)
        return generate(prompt)

    budget = SharedModelBudget(3)
    other = SharedModelBudget(3)
    backend = SimpleNamespace(generate=generate_async if async_mode else generate)
    wrapped = budget.wrap(BackendWrapper(other.wrap(budget.wrap(backend))))
    if async_mode:
        assert await asyncio.gather(wrapped.generate("first"), wrapped.generate("second")) == [
            "first",
            "second",
        ]
    else:
        assert wrapped.generate("first") == "first"
        assert wrapped.generate("second") == "second"
    assert observed == [True, True]
    assert budget.remaining == other.remaining == 1


async def test_shared_admission_preserves_scope_for_deferred_awaitable():
    from types import SimpleNamespace

    from looplet.backend_contract import single_model_request_active
    from looplet.subagent import SharedModelBudget

    observed = []

    async def generate_async(prompt):
        await asyncio.sleep(0)
        observed.append(single_model_request_active())
        return prompt

    budget = SharedModelBudget(2)
    wrapped = budget.wrap(SimpleNamespace(generate=lambda prompt: generate_async(prompt)))
    assert await wrapped.generate("deferred") == "deferred"
    assert observed == [True]
    assert budget.remaining == 1
    assert not single_model_request_active()


def test_shared_admission_owns_one_provider_attempt():
    from types import SimpleNamespace

    from looplet.resilient import ResilientBackend
    from looplet.subagent import SharedModelBudget

    calls = []

    def generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise ConnectionError("transient")
        return "unexpected hidden retry"

    budget = SharedModelBudget(2)
    wrapped = budget.wrap(
        ResilientBackend(SimpleNamespace(generate=generate), sleep=lambda delay: None)
    )
    with pytest.raises(ConnectionError, match="transient"):
        wrapped.generate("one admitted attempt")
    assert calls == ["one admitted attempt"]
    assert budget.remaining == 1


def test_shared_admission_checkpoint_cannot_increase_allowance():
    from looplet.subagent import SharedModelBudget

    budget = SharedModelBudget(5)
    with budget.admit():
        pass
    saved = budget.checkpoint_state()
    restored = SharedModelBudget(5)
    restored.restore_checkpoint_state(saved)
    assert restored.remaining == 4
    restored.charge()
    restored.restore_checkpoint_state(saved)
    assert restored.remaining == 3
    with pytest.raises(ValueError, match="increase"):
        SharedModelBudget(6).restore_checkpoint_state(saved)
    assert saved["remaining"] == 4


async def test_async_child_uses_async_loop_and_shared_budget():
    from looplet import run_sub_loop_async
    from looplet.subagent import ChildRunPolicy, SharedModelBudget
    from looplet.testing import AsyncMockLLMBackend

    budget = SharedModelBudget(remaining=2)
    backend = AsyncMockLLMBackend(
        responses=[
            '{"tool":"add","args":{"a":1,"b":2}}',
            '{"tool":"done","args":{"summary":"sum is 3"}}',
        ]
    )
    result = await run_sub_loop_async(
        llm=budget.wrap(backend),
        tools=_tools(),
        config=LoopConfig(max_steps=3, use_native_tools=False),
        policy=ChildRunPolicy(parent_id="parent-async", model_budget=budget),
    )
    assert result["steps"][0]["tool_result"]["data"] == {"sum": 3}
    assert result["stop_reason"] == "done"
    assert result["completed"]
    assert result["llm_calls"] == 2
    assert result["parent_id"] == "parent-async"
    assert budget.remaining == 0


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("terminal_name", ["done", "finish"])
async def test_default_child_completion_is_isolated(async_mode, native, terminal_name):
    import json

    from looplet import run_sub_loop_async
    from looplet.testing import AsyncMockLLMBackend

    parent_calls = []
    parent = BaseToolRegistry()
    parent_terminal = ToolSpec(
        name=terminal_name,
        description="Complete the parent.",
        parameters={"answer": "str"},
        execute=lambda **args: parent_calls.append(args),
    )
    parent.register(parent_terminal)
    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    options = {
        "llm": backend_type(
            [json.dumps({"tool": terminal_name, "args": {"summary": "child finished"}})]
        ),
        "tools": parent,
        "state_mutating_tools": [terminal_name],
        "config": LoopConfig(max_steps=1, use_native_tools=native, done_tool=terminal_name),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert result["completed"]
    assert result["output"] == {"status": "completed", "summary": "child finished"}
    assert result["steps"][0]["tool_result"].get("error") is None
    assert parent_calls == []
    assert parent._tools[terminal_name] is parent_terminal


@pytest.mark.parametrize("async_mode", [False, True])
async def test_routed_child_spends_shared_model_allowance(async_mode):
    from looplet import run_sub_loop_async
    from looplet.router import ModelProfile, SimpleRouter
    from looplet.subagent import ChildRunPolicy, SharedModelBudget
    from looplet.testing import AsyncMockLLMBackend

    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    selected = backend_type(responses=['{"tool":"add","args":{"a":1,"b":2}}'])
    unused = backend_type(responses=[])
    profile = ModelProfile("selected", selected)
    router = SimpleRouter({"reasoning": profile}, default_profile=profile)
    config = LoopConfig(max_steps=3, use_native_tools=False, router=router)
    budget = SharedModelBudget(1)
    options = {
        "llm": unused,
        "sub_tools": _tools(),
        "config": config,
        "policy": ChildRunPolicy(parent_id="parent", model_budget=budget),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert selected.calls == 1 and unused.calls == 0
    assert budget.remaining == 0
    assert result["steps"][0]["tool_result"]["data"] == {"sum": 3}
    assert not result["completed"] and result["stop_reason"] == "model_budget_exhausted"
    assert config.router is router


async def test_async_child_awaits_parent_events():
    from looplet import run_sub_loop_async
    from looplet.testing import AsyncMockLLMBackend

    received = []

    class Observer:
        async def on_event(self, payload):
            received.append(payload.event)

    result = await run_sub_loop_async(
        llm=AsyncMockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2}}',
                '{"tool":"done","args":{"summary":"ok"}}',
            ]
        ),
        tools=_tools(),
        config=LoopConfig(max_steps=3, use_native_tools=False),
        parent_hooks=[Observer()],
    )
    assert result["completed"]
    assert LifecycleEvent.PRE_TOOL_USE in received
    assert received.count(LifecycleEvent.SUBAGENT_START) == 1
    assert received.count(LifecycleEvent.SUBAGENT_STOP) == 1


async def test_parallel_children_are_bounded_and_ordered():
    from looplet import run_sub_loops_async
    from looplet.testing import AsyncMockLLMBackend

    active = 0
    peak = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    class Backend(AsyncMockLLMBackend):
        async def generate(self, prompt, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
            try:
                await release.wait()
                return await super().generate(prompt, **kwargs)
            finally:
                active -= 1

    requests = [
        {
            "llm": Backend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
            "tools": _tools(),
            "config": LoopConfig(max_steps=1, use_native_tools=False),
            "subagent_id": f"child-{index}",
        }
        for index in range(4)
    ]
    task = asyncio.create_task(run_sub_loops_async(requests, max_concurrency=2))
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert peak == 2
    release.set()
    results = await task
    assert peak == 2
    assert all(result["completed"] for result in results)
    assert [result["subagent_id"] for result in results] == [
        "child-0",
        "child-1",
        "child-2",
        "child-3",
    ]


@pytest.mark.parametrize("native", [False, True])
async def test_parallel_recording_keeps_child_and_parent_scopes(native):
    import json

    from looplet import run_sub_loops_async, tools_from
    from looplet.loop import _scope_llm_for_tool
    from looplet.provenance import AsyncRecordingLLMBackend
    from looplet.subagent import ChildRunPolicy, SharedModelBudget
    from looplet.types import ToolCall

    fast_finished = asyncio.Event()
    release_slow = asyncio.Event()

    class Backend:
        async def generate(self, prompt, **kwargs):
            label = next(
                name
                for name in ("slow-request", "fast-request", "parent-request")
                if name in prompt
            )
            if label == "slow-request":
                await release_slow.wait()
            elif label == "fast-request":
                fast_finished.set()
            return json.dumps({"tool": "done", "args": {"summary": label}})

        async def generate_with_tools(self, prompt, **kwargs):
            response = json.loads(await self.generate(prompt, **kwargs))
            return [
                {
                    "type": "tool_use",
                    "id": "finish",
                    "name": response["tool"],
                    "input": response["args"],
                }
            ]

    recording = AsyncRecordingLLMBackend(Backend())
    budget = SharedModelBudget(remaining=3)
    parent_backend = budget.wrap(recording)
    requests = [
        {
            "llm": _scope_llm_for_tool(parent_backend, ToolCall(tool=f"{name}-delegate")),
            "task": {"description": f"{name}-request"},
            "sub_tools": tools_from([], include_done=True),
            "config": LoopConfig(max_steps=1, use_native_tools=native),
            "policy": ChildRunPolicy(parent_id="scope-parent", model_budget=budget),
            "subagent_id": name,
        }
        for name in ("slow", "fast")
    ]
    children = asyncio.create_task(run_sub_loops_async(requests, max_concurrency=2))
    try:
        await asyncio.wait_for(fast_finished.wait(), timeout=2)
        await parent_backend.generate("parent-request")
    finally:
        release_slow.set()
        results = await children
    assert [result["output"]["summary"] for result in results] == ["slow-request", "fast-request"]
    assert [call.scope for call in recording.calls] == [
        "tool:fast-delegate",
        None,
        "tool:slow-delegate",
    ]
    assert budget.remaining == 0


async def test_parallel_failure_cancels_and_closes_sibling():
    from looplet import run_sub_loops_async

    started = asyncio.Event()
    closed = asyncio.Event()
    never = asyncio.Event()

    class WaitingBackend:
        async def generate(self, prompt, **kwargs):
            started.set()
            try:
                await never.wait()
            finally:
                closed.set()

    class FailingBackend:
        async def generate(self, prompt, **kwargs):
            await started.wait()
            raise ValueError("child failed")

    requests = [
        {"llm": backend, "tools": _tools(), "config": LoopConfig(use_native_tools=False)}
        for backend in [WaitingBackend(), FailingBackend()]
    ]
    with pytest.raises(ExceptionGroup) as failure:
        await run_sub_loops_async(requests, max_concurrency=2)
    assert closed.is_set()
    assert failure.value.exceptions[0].result["stop_reason"] == "llm_error"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_parallel_children_reject_invalid_limit(limit):
    from looplet import run_sub_loops_async

    with pytest.raises(ValueError, match="positive integer"):
        await run_sub_loops_async([], max_concurrency=limit)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("custom_parent", [False, True])
async def test_child_cannot_override_parent_permission(async_mode, custom_parent):
    from looplet import PermissionEngine, PermissionHook, run_sub_loop_async
    from looplet.testing import AsyncMockLLMBackend

    effects = []
    tools = _tools()
    tools.register(
        ToolSpec(
            name="write",
            description="Write",
            parameters={},
            execute=lambda: effects.append("write"),
        )
    )

    class CustomPermissionHook(PermissionHook):
        def check_permission(self, tool_call, state):
            return tool_call.tool != "write"

        def pre_loop(self, *args, **kwargs):
            raise AssertionError("Parent flow control must not run on child state")

    parent = (
        CustomPermissionHook(PermissionEngine().allow("write"))
        if custom_parent
        else PermissionHook(PermissionEngine().deny("write"))
    )
    child = PermissionHook(PermissionEngine().allow("write"))
    responses = [
        '{"tool":"write","args":{}}',
        '{"tool":"add","args":{"a":1,"b":2}}',
        '{"tool":"done","args":{"summary":"ok"}}',
    ]
    options = {
        "llm": AsyncMockLLMBackend(responses=responses)
        if async_mode
        else MockLLMBackend(responses=responses),
        "tools": tools,
        "hooks": [child],
        "parent_hooks": [parent],
        "config": LoopConfig(max_steps=4, use_native_tools=False),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert effects == []
    assert result["steps"][0]["tool_result"]["error"]
    assert result["steps"][1]["tool_result"]["data"] == {"sum": 3}
    assert result["completed"]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_child_cannot_widen_execution_capabilities(async_mode):
    from looplet import ExecutionPolicy, run_sub_loop_async
    from looplet.subagent import ChildRunPolicy
    from looplet.testing import AsyncMockLLMBackend

    effects = []
    tools = _tools()
    tools.register(
        ToolSpec(
            name="shell",
            description="Shell",
            parameters={},
            execute=lambda: effects.append("shell"),
            capabilities=["shell"],
        )
    )
    responses = ['{"tool":"shell","args":{}}', '{"tool":"done","args":{"summary":"ok"}}']
    config = LoopConfig(
        max_steps=3,
        use_native_tools=False,
        execution_policy=ExecutionPolicy(capabilities=frozenset({"shell", "network"})),
    )
    options = {
        "llm": AsyncMockLLMBackend(responses=responses)
        if async_mode
        else MockLLMBackend(responses=responses),
        "tools": tools,
        "config": config,
        "policy": ChildRunPolicy(
            parent_id="parent",
            execution_policy=ExecutionPolicy(capabilities=frozenset({"network"})),
        ),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert effects == []
    assert result["steps"][0]["tool_result"]["error"]
    assert config.execution_policy.capabilities == frozenset({"shell", "network"})
    assert result["completed"]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_child_recovery_reuses_completed_output(tmp_path, async_mode):
    from looplet import run_sub_loop_async
    from looplet.subagent import ChildRunPolicy, SharedModelBudget
    from looplet.testing import AsyncMockLLMBackend

    artifact = tmp_path / "writes.txt"

    def write():
        with artifact.open("a") as handle:
            handle.write("one effect\n")
        return {"path": str(artifact)}

    tools = _tools()
    tools.register(ToolSpec(name="write", description="Write", parameters={}, execute=write))
    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    first = backend_type(
        responses=['{"tool":"write","args":{}}', '{"tool":"done","args":{"answer":"saved"}}']
    )
    budget = SharedModelBudget(2)
    options = {
        "llm": first,
        "sub_tools": tools,
        "task": {"goal": "write once"},
        "subagent_id": "write-child",
        "config": LoopConfig(max_steps=3, use_native_tools=False),
        "policy": ChildRunPolicy(
            parent_id="parent",
            model_budget=budget,
            recovery_dir=str(tmp_path / "children"),
            recovery_version="agent-v1/policy-v1",
        ),
    }
    run = run_sub_loop_async if async_mode else run_sub_loop
    result = await run(**options) if async_mode else run(**options)
    assert result["completed"]
    assert result["output"] == {"answer": "saved"}
    assert budget.remaining == 0
    second = backend_type(responses=[])
    options["llm"] = second
    recovered = await run(**options) if async_mode else run(**options)
    assert recovered["completed"] and recovered["recovered"]
    assert recovered["output"] == result["output"]
    assert recovered["subagent_id"] == result["subagent_id"]
    assert recovered["new_llm_calls"] == 0
    assert second.calls == 0
    assert artifact.read_text().splitlines() == ["one effect"]


@pytest.mark.parametrize("change", ["task", "version", "permission"])
def test_child_recovery_refuses_changed_request(tmp_path, change):
    from dataclasses import replace

    from looplet import PermissionEngine
    from looplet.subagent import ChildRecoveryError, ChildRunPolicy

    options = {
        "llm": MockLLMBackend(responses=['{"tool":"done","args":{"answer":"saved"}}']),
        "sub_tools": _tools(),
        "task": {"goal": "original"},
        "subagent_id": "stable-child",
        "config": LoopConfig(max_steps=3, use_native_tools=False),
        "policy": ChildRunPolicy(
            parent_id="parent", recovery_dir=str(tmp_path), recovery_version="v1"
        ),
    }
    assert run_sub_loop(**options)["completed"]
    if change == "task":
        options["task"] = {"goal": "different"}
    elif change == "version":
        options["policy"] = replace(options["policy"], recovery_version="v2")
    else:
        options["policy"] = replace(
            options["policy"], permission_engines=(PermissionEngine().deny("add"),)
        )
    with pytest.raises(ChildRecoveryError, match="does not match"):
        run_sub_loop(**options)


def test_child_recovery_refuses_uncertain_effect(tmp_path):
    from looplet.checkpoint import UncertainToolEffect
    from looplet.subagent import ChildRunPolicy

    class ProcessInterrupted(BaseException):
        pass

    effects = []

    def write():
        effects.append("written")
        raise ProcessInterrupted()

    tools = _tools()
    tools.register(ToolSpec(name="write", description="Write", parameters={}, execute=write))
    options = {
        "llm": MockLLMBackend(responses=['{"tool":"write","args":{}}']),
        "sub_tools": tools,
        "subagent_id": "interrupted-child",
        "config": LoopConfig(max_steps=3, use_native_tools=False),
        "policy": ChildRunPolicy(
            parent_id="parent", recovery_dir=str(tmp_path), recovery_version="v1"
        ),
    }
    with pytest.raises(ProcessInterrupted):
        run_sub_loop(**options)
    options["llm"] = MockLLMBackend(responses=['{"tool":"write","args":{}}'])
    with pytest.raises(UncertainToolEffect):
        run_sub_loop(**options)
    assert effects == ["written"]
    assert options["llm"].calls == 0


@pytest.mark.parametrize("async_mode", [False, True])
async def test_child_recovery_resumes_committed_step(tmp_path, async_mode):
    import os
    import subprocess
    import sys

    from looplet import DefaultState, run_sub_loop_async
    from looplet.provenance import AsyncRecordingLLMBackend, RecordingLLMBackend
    from looplet.subagent import ChildRunPolicy
    from looplet.testing import AsyncMockLLMBackend

    artifact = tmp_path / "committed.txt"
    records = tmp_path / "records"
    program = """
import os
from pathlib import Path
from runpy import run_path
from looplet import BaseToolRegistry, DefaultState, LoopConfig, ToolSpec, run_sub_loop
from looplet.subagent import ChildRunPolicy
artifact = Path(os.environ["CHILD_ARTIFACT"])
tools = run_path(os.environ["CHILD_TEST_MODULE"])["_tools"]()
tools._tools.pop("add")
def write():
    with artifact.open("a") as handle:
        handle.write("committed\\n")
    return {"path": str(artifact)}
tools.register(ToolSpec(name="write", description="Write", parameters={}, execute=write))
class InterruptedBackend:
    calls = 0
    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            os._exit(0)
        return '{"tool":"write","args":{}}'
run_sub_loop(llm=InterruptedBackend(), sub_tools=tools, state=DefaultState(max_steps=3),
             task={"goal": "write once"}, subagent_id="committed-child",
             config=LoopConfig(max_steps=3, use_native_tools=False),
             policy=ChildRunPolicy(parent_id="parent", recovery_dir=os.environ["CHILD_RECORDS"], recovery_version="v1"))
"""
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        timeout=15,
        env={
            **os.environ,
            "CHILD_ARTIFACT": str(artifact),
            "CHILD_RECORDS": str(records),
            "CHILD_TEST_MODULE": __file__,
        },
    )
    assert artifact.read_text() == "committed\n"

    def write():
        with artifact.open("a") as handle:
            handle.write("duplicate\n")
        return {"path": str(artifact)}

    tools = _tools()
    tools._tools.pop("add")
    tools.register(ToolSpec(name="write", description="Write", parameters={}, execute=write))
    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    backend = backend_type(responses=['{"tool":"done","args":{"answer":"resumed"}}'], cycle=False)
    recording = (AsyncRecordingLLMBackend if async_mode else RecordingLLMBackend)(backend)
    options = {
        "llm": recording,
        "sub_tools": tools,
        "state": DefaultState(max_steps=3),
        "task": {"goal": "write once"},
        "subagent_id": "committed-child",
        "config": LoopConfig(max_steps=3, use_native_tools=False),
        "policy": ChildRunPolicy(
            parent_id="parent", recovery_dir=str(records), recovery_version="v1"
        ),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert result["completed"]
    assert backend.calls == 1
    assert artifact.read_text() == "committed\n"
    assert options["state"].steps[0].number == 2
    assert len(result["steps"]) == 1
    assert len(recording.calls) == 1
    assert f'"path": "{artifact}"' in recording.calls[0].prompt
    assert "do not repeat their effects" in recording.calls[0].prompt


def test_child_recovery_receipts_preserve_domain_callbacks():
    from types import SimpleNamespace

    from looplet.subagent import _ChildRecoveryEvidence

    restored = []
    evidence = _ChildRecoveryEvidence(
        lambda context: {"domain_value": 7},
        lambda context, domain: restored.append(dict(domain)),
    )
    receipt = {"step": 1, "tool_call": {"tool": "write"}, "tool_result": {"data": "saved"}}
    domain = {"domain_value": 7, "_looplet_child_receipts": [receipt]}
    context = SimpleNamespace(state=SimpleNamespace(steps=[]))
    evidence.restore_checkpoint_state(context, domain)
    assert restored == [{"domain_value": 7}]
    assert evidence.checkpoint_state(context) == domain
    assert evidence.pre_prompt(None, None, None, 2)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_child_inherits_custom_permission_without_parent_flow(async_mode):
    from looplet import run_sub_loop_async
    from looplet.testing import AsyncMockLLMBackend

    class ParentGuard:
        def check_permission(self, tool_call, state):
            return tool_call.tool != "write"

        def should_stop(self, *args):
            raise AssertionError("parent flow must not run on child state")

    effects = []
    tools = _tools()
    tools.register(
        ToolSpec(
            name="write",
            description="Write",
            parameters={},
            execute=lambda: effects.append("write"),
        )
    )
    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    options = {
        "llm": backend_type(
            responses=[
                '{"tool":"write","args":{}}',
                '{"tool":"add","args":{"a":2,"b":3}}',
                '{"tool":"done","args":{"answer":"ok"}}',
            ],
            cycle=False,
        ),
        "sub_tools": tools,
        "parent_hooks": [ParentGuard()],
        "config": LoopConfig(max_steps=4, use_native_tools=False),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert result["completed"]
    assert effects == []
    assert result["steps"][0]["tool_result"]["error"]
    assert result["steps"][1]["tool_result"]["data"] == {"sum": 5}


async def test_parallel_children_keep_approval_requests_separate():
    import json

    from looplet import ApprovalHook, run_sub_loops_async
    from looplet.testing import AsyncMockLLMBackend

    parent = ApprovalHook()
    tools = _tools()
    tools.register(
        ToolSpec(
            name="request",
            description="Request",
            parameters={"label": "str"},
            execute=lambda label: {"needs_approval": True, "approval_description": label},
        )
    )
    requests = [
        {
            "llm": AsyncMockLLMBackend(
                responses=[json.dumps({"tool": "request", "args": {"label": label}})], cycle=False
            ),
            "tools": tools,
            "parent_hooks": [parent],
            "config": LoopConfig(max_steps=3, use_native_tools=False),
        }
        for label in ("first action", "second action")
    ]
    results = await run_sub_loops_async(requests, max_concurrency=2)
    assert [result["approval"]["description"] for result in results] == [
        "first action",
        "second action",
    ]
    assert all(result["stop_reason"] == "waiting_for_approval" for result in results)
    assert all(not result["completed"] for result in results)
    assert parent.pending is None


def test_child_summary_uses_accepted_terminal_output():
    tools = _tools()
    tools._tools["done"] = ToolSpec(
        name="done",
        description="Done",
        parameters={"summary": "str"},
        execute=lambda summary: {"summary": summary},
    )
    result = run_sub_loop(
        llm=MockLLMBackend(
            responses=['{"tool":"done","args":{"summary":"Greeted Alice"}}'], cycle=False
        ),
        sub_tools=tools,
        config=LoopConfig(max_steps=2, use_native_tools=False),
    )
    assert result["completed"]
    assert result["summary"] == "Greeted Alice"
    assert result["output"] == {"summary": "Greeted Alice"}


@pytest.mark.parametrize("async_mode", [False, True])
async def test_child_config_overrides_default_state_limit(async_mode):
    from looplet import run_sub_loop_async
    from looplet.testing import AsyncMockLLMBackend

    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    backend = backend_type(responses=['{"tool":"add","args":{"a":2,"b":3}}'])
    options = {
        "llm": backend,
        "sub_tools": _tools(),
        "config": LoopConfig(max_steps=7, use_native_tools=False),
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert len(result["steps"]) == backend.calls == 7
    assert result["stop_reason"] == "budget_exhausted"
    assert not result["completed"]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("with_policy", [False, True])
async def test_child_config_bounds_supplied_default_state(async_mode, with_policy):
    from looplet import DefaultState, run_sub_loop_async
    from looplet.subagent import ChildRunPolicy
    from looplet.testing import AsyncMockLLMBackend

    backend_type = AsyncMockLLMBackend if async_mode else MockLLMBackend
    backend = backend_type(responses=['{"tool":"add","args":{"a":2,"b":3}}'])
    state = DefaultState(max_steps=9)
    config = LoopConfig(max_steps=4, use_native_tools=False)
    options = {
        "llm": backend,
        "sub_tools": _tools(),
        "state": state,
        "config": config,
        "policy": ChildRunPolicy(parent_id="parent") if with_policy else None,
    }
    result = await run_sub_loop_async(**options) if async_mode else run_sub_loop(**options)
    assert len(result["steps"]) == backend.calls == 4
    assert result["stop_reason"] == "budget_exhausted"
    assert not result["completed"]
    assert config.max_steps == state.max_steps == 4


def test_child_does_not_load_parent_checkpoint_without_policy(tmp_path):
    from looplet import RunEnvelope
    from looplet.checkpoint import Checkpoint, FileCheckpointStore

    parent = Checkpoint(
        step_number=8,
        session_log_data={},
        conversation_data=None,
        config_snapshot={},
        tool_results_store={},
        metadata={"pending_calls": [{"call_id": "parent-effect"}]},
        run_envelope={"run_id": "parent"},
    )
    FileCheckpointStore(tmp_path).save(parent, "step_8_pending")
    config = LoopConfig(
        max_steps=2,
        use_native_tools=False,
        checkpoint_dir=str(tmp_path),
        initial_checkpoint=parent,
        run_envelope=RunEnvelope(run_id="parent"),
    )
    backend = MockLLMBackend(['{"tool":"done","args":{"answer":"child output"}}'], cycle=False)
    result = run_sub_loop(llm=backend, sub_tools=_tools(), config=config, subagent_id="child")
    assert result["completed"] and result["output"] == {"answer": "child output"}
    assert result["run_envelope"]["run_id"] == "child"
    assert result["parent_id"] == "parent"
    assert config.initial_checkpoint is parent
    assert config.checkpoint_dir == str(tmp_path)


def test_child_recovery_refuses_corrupt_checkpoint(tmp_path):
    from looplet.subagent import ChildRecoveryError, ChildRunPolicy

    options = {
        "llm": MockLLMBackend(['{"tool":"done","args":{"answer":"saved"}}'], cycle=False),
        "sub_tools": _tools(),
        "subagent_id": "child",
        "config": LoopConfig(max_steps=2, use_native_tools=False),
        "policy": ChildRunPolicy(
            parent_id="parent", recovery_dir=str(tmp_path), recovery_version="v1"
        ),
    }
    assert run_sub_loop(**options)["completed"]
    checkpoint = next(tmp_path.rglob("*__delegation.json"))
    checkpoint.write_text("{corrupt")
    backend = MockLLMBackend(['{"tool":"done","args":{"answer":"repeat"}}'], cycle=False)
    options["llm"] = backend
    with pytest.raises(ChildRecoveryError, match="corrupt"):
        run_sub_loop(**options)
    assert backend.calls == 0


async def test_async_child_start_cancellation_emits_one_stop():
    from looplet import run_sub_loop_async
    from looplet.testing import AsyncMockLLMBackend

    entered = asyncio.Event()
    never = asyncio.Event()
    events = []
    stops = []

    class Observer:
        async def on_event(self, payload):
            events.append(payload.event)
            if payload.event == LifecycleEvent.SUBAGENT_STOP:
                stops.append(payload.termination_reason)
            if payload.event == LifecycleEvent.SUBAGENT_START:
                entered.set()
                await never.wait()

    backend = AsyncMockLLMBackend(['{"tool":"done","args":{"answer":"ok"}}'], cycle=False)
    child = asyncio.create_task(
        run_sub_loop_async(
            llm=backend,
            sub_tools=_tools(),
            parent_hooks=[Observer()],
            config=LoopConfig(use_native_tools=False),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    child.cancel()
    with pytest.raises(asyncio.CancelledError):
        await child
    assert events.count(LifecycleEvent.SUBAGENT_START) == 1
    assert events.count(LifecycleEvent.SUBAGENT_STOP) == 1
    assert stops == ["cancelled"]
    assert backend.calls == 0


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="add",
            description="Add",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="Finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


class _Recorder:
    def __init__(self):
        self.events: list[LifecycleEvent] = []
        self.payloads = []

    def on_event(self, payload):
        self.events.append(payload.event)
        self.payloads.append(payload)
        return None


class TestSubagentLifecycleEvents:
    def test_shared_budget_preserves_text_backend_and_limits_calls(self):
        from looplet.subagent import ChildRunLimitExceeded, SharedModelBudget

        class TextOnlyBackend:
            def generate(self, prompt):
                return "ok"

        budget = SharedModelBudget(remaining=1)
        backend = budget.wrap(TextOnlyBackend())
        assert not hasattr(backend, "generate_with_tools")
        assert backend.generate("parent") == "ok"
        with pytest.raises(ChildRunLimitExceeded, match="model call budget exhausted"):
            backend.generate("again")
        assert budget.remaining == 0

    def test_expired_child_deadline_stops_before_model_call(self):
        from looplet.subagent import ChildRunLimitExceeded, ChildRunPolicy

        backend = MockLLMBackend(responses=['{"tool":"add","args":{"a":1,"b":2}}'])
        with pytest.raises(ChildRunLimitExceeded, match="deadline exceeded"):
            run_sub_loop(
                llm=backend,
                tools=_tools(),
                policy=ChildRunPolicy(parent_id="parent-1", deadline=time.monotonic() - 1),
            )
        assert backend.calls == 0

    def test_child_policy_does_not_resume_parent_checkpoint(self):
        from looplet.subagent import ChildRunPolicy

        parent_checkpoint = Checkpoint(
            step_number=3,
            session_log_data={"entries": []},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={},
        )
        config = LoopConfig(
            max_steps=4, initial_checkpoint=parent_checkpoint, use_native_tools=False
        )
        result = run_sub_loop(
            llm=MockLLMBackend(responses=['{"tool":"add","args":{"a":1,"b":2}}']),
            tools=_tools(),
            config=config,
            policy=ChildRunPolicy(parent_id="parent-1", max_steps=1),
        )
        assert [step["step"] for step in result["steps"]] == [1]
        assert config.initial_checkpoint is parent_checkpoint

    def test_child_policy_limits_tools_and_shared_model_calls(self):
        from looplet.subagent import ChildRunPolicy, SharedModelBudget

        effects = []
        tools = _tools()
        tools.register(
            ToolSpec(
                name="write", description="Write", parameters={}, execute=lambda: effects.append(1)
            )
        )
        budget = SharedModelBudget(remaining=2)
        backend = budget.wrap(
            MockLLMBackend(
                responses=[
                    '{"tool":"write","args":{},"call_id":"w"}',
                    '{"tool":"add","args":{"a":1,"b":2},"call_id":"a"}',
                ],
                cycle=False,
            )
        )
        result = run_sub_loop(
            llm=backend,
            tools=tools,
            max_steps=5,
            policy=ChildRunPolicy(
                parent_id="parent-1",
                allowed_tools=frozenset({"add"}),
                max_steps=4,
                model_budget=budget,
            ),
        )
        assert effects == []
        assert [step["tool_call"]["tool"] for step in result["steps"]] == ["write", "add"]
        assert result["steps"][0]["tool_result"]["error"]
        assert result["steps"][1]["tool_result"]["data"] == {"sum": 3}
        assert budget.remaining == 0
        assert result["llm_calls"] == 2
        assert result["stop_reason"] == "model_budget_exhausted"
        assert result["parent_id"] == "parent-1"

    def test_subagent_start_and_stop_fire(self):
        r = _Recorder()
        run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
        )
        assert LifecycleEvent.SUBAGENT_START in r.events
        assert LifecycleEvent.SUBAGENT_STOP in r.events

    def test_subagent_id_correlates_events(self):
        r = _Recorder()
        run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
            subagent_id="custom-id",
        )
        ids = [
            p.subagent_id
            for p in r.payloads
            if p.event in (LifecycleEvent.SUBAGENT_START, LifecycleEvent.SUBAGENT_STOP)
        ]
        assert ids == ["custom-id", "custom-id"]

    def test_result_carries_subagent_id(self):
        result = run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            subagent_id="correlation-7",
        )
        assert result["subagent_id"] == "correlation-7"

    def test_auto_generated_id_when_not_supplied(self):
        r = _Recorder()
        result = run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
        )
        assert result["subagent_id"]  # non-empty
        assert len(result["subagent_id"]) == 12

    def test_subagent_stop_fires_once_when_summary_fails(self):
        recorder = _Recorder()

        with pytest.raises(RuntimeError, match="summary failed"):
            run_sub_loop(
                llm=MockLLMBackend(
                    responses=['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']
                ),
                tools=_tools(),
                max_steps=3,
                hooks=[recorder],
                build_summary=lambda *_args: (_ for _ in ()).throw(RuntimeError("summary failed")),
            )

        assert recorder.events.count(LifecycleEvent.SUBAGENT_START) == 1
        assert recorder.events.count(LifecycleEvent.SUBAGENT_STOP) == 1
        stop = [
            payload
            for payload in recorder.payloads
            if payload.event is LifecycleEvent.SUBAGENT_STOP
        ][0]
        assert stop.termination_reason == "error"


# ── parent_hooks=... forwards sub-loop events to parent observers ──


def test_parent_hooks_receive_sub_loop_events() -> None:
    """``run_sub_loop(parent_hooks=...)`` forwards every lifecycle
    event the sub-loop emits onto the parent's hooks via their
    ``on_event`` method, tagged with ``subagent_id`` in
    ``payload.extra``. Lets parent observability stacks (MetricsHook,
    StreamingHook, TrajectoryRecorder) see sub-activity without the
    user manually plumbing it."""
    import json

    from looplet import register_done_tool
    from looplet.subagent import run_sub_loop
    from looplet.testing import MockLLMBackend
    from looplet.tools import BaseToolRegistry

    received: list[tuple[str, dict | None]] = []

    class _Spy:
        def on_event(self, payload) -> None:
            received.append((str(payload.event), dict(payload.extra or {})))

    parent_hook = _Spy()

    tools = BaseToolRegistry()
    register_done_tool(tools)

    sub_llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "done", "args": {"summary": "sub-done"}, "reasoning": "r"}),
        ]
    )

    result = run_sub_loop(
        llm=sub_llm,
        task={"goal": "do nothing"},
        tools=tools,
        max_steps=2,
        parent_hooks=[parent_hook],
    )
    assert result["subagent_id"]
    # The parent's spy received SUBAGENT_START + SUBAGENT_STOP at minimum,
    # both tagged with the same subagent_id.
    event_names = [name for name, _ in received]
    assert any("SUBAGENT_START" in n for n in event_names), event_names
    assert any("SUBAGENT_STOP" in n for n in event_names), event_names
    # And every event payload carries the subagent_id.
    sub_ids = {extra.get("subagent_id") for _, extra in received if extra}
    sub_ids.discard(None)
    assert sub_ids == {result["subagent_id"]}, sub_ids


def test_no_parent_hooks_means_no_forwarding() -> None:
    """Without ``parent_hooks``, the sub-loop's events stay isolated
    (default opt-in behaviour)."""
    import json

    from looplet import register_done_tool
    from looplet.subagent import run_sub_loop
    from looplet.testing import MockLLMBackend
    from looplet.tools import BaseToolRegistry

    received: list[str] = []

    class _Spy:
        def on_event(self, payload) -> None:
            received.append(str(payload.event))

    parent_hook = _Spy()

    tools = BaseToolRegistry()
    register_done_tool(tools)
    sub_llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "done", "args": {"summary": "x"}, "reasoning": "r"}),
        ]
    )

    # parent_hooks omitted → spy must see nothing.
    run_sub_loop(
        llm=sub_llm,
        task={"goal": "x"},
        tools=tools,
        max_steps=2,
        # hooks= is the SUB-loop's hooks - does not implicitly include
        # the parent observer.
    )
    assert received == []
    # And: even if you pass [parent_hook] as hooks=, it gets the
    # sub-loop's full LoopHook interface (which it doesn't implement),
    # not just on_event forwarding. The on_event method is still
    # called for events, so we'd see them - that's the OLD behaviour.
    # The new behaviour is that you can have parent observers see
    # events WITHOUT sharing the sub-loop's full hook interface.
