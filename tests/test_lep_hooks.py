"""Dogfood tests for out-of-process hooks (``looplet.lep``).

These tests witness the two hazards the cross-runtime design must
survive (HOOK_CARTRIDGE_DESIGN.md):

* **H4 composition** - an in-process hook and an out-of-process LEP hook
  run in the *same* loop and their authority composes (AND semantics on
  permission; both denials are honoured).
* **H2 wide view** - an LEP hook that declares a ``transcript`` view can
  read conversation history across the process boundary and decide on it.

It also pins the failure-policy contract (fail_closed denies on a broken
server; fail_open allows) and the basic authority slots
(``check_permission`` / ``pre_prompt``).
"""

from __future__ import annotations

import io
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

import looplet
from looplet import (
    BaseToolRegistry,
    DefaultState,
    Deny,
    LoopConfig,
    composable_loop,
)
from looplet.hook_view import ViewSpec
from looplet.lep import LEPHookAdapter, server_argv
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import ToolCall

_SRC = str(Path(looplet.__file__).resolve().parent.parent)


def _write_server(tmp_path: Path, body: str, name: str = "server.py") -> Path:
    """Write a LEPServerBase policy server that can import looplet.

    ``body`` is the source of a ``decide`` method (with a 4-space class-body
    indent already applied by the caller via dedent here).
    """
    method_src = textwrap.indent(textwrap.dedent(body).strip("\n"), "    ")
    src = (
        "import sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "from looplet.lep import LEPServerBase\n"
        "\n"
        "class PolicyServer(LEPServerBase):\n"
        f"{method_src}\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(PolicyServer().serve())\n"
    )
    path = tmp_path / name
    path.write_text(src, encoding="utf-8")
    return path


def _adapter(tmp_path: Path, body: str, **kwargs) -> LEPHookAdapter:
    server = _write_server(tmp_path, body)
    return LEPHookAdapter(server_argv(str(server)), **kwargs)


class TestLEPAuthoritySlots:
    def test_check_permission_denies_out_of_process(self, tmp_path):
        adapter = _adapter(
            tmp_path,
            """
            def decide(self, slot, view):
                if slot == "check_permission" and view.get("tool") == "rm":
                    return {"kind": "Deny", "block": "rm denied by policy"}
                return {"kind": "Continue"}
            """,
            view=ViewSpec(fields=frozenset({"tool", "args"})),
        )
        adapter.pre_loop(None, None, None)
        try:
            assert adapter.check_permission(ToolCall(tool="rm", args={}), None) is False
            assert adapter.check_permission(ToolCall(tool="ls", args={}), None) is True
        finally:
            adapter.close()

    def test_pre_prompt_injects_context(self, tmp_path):
        adapter = _adapter(
            tmp_path,
            """
            def decide(self, slot, view):
                if slot == "pre_prompt":
                    return {"kind": "InjectContext", "text": "[audited]"}
                return {"kind": "Continue"}
            """,
        )
        adapter.pre_loop(None, None, None)
        try:
            assert adapter.pre_prompt(None, None, None, 0) == "[audited]"
        finally:
            adapter.close()


class TestFailurePolicy:
    def test_legacy_block_permission_remains_a_denial_in_the_loop(self, tmp_path):
        adapter = _adapter(
            tmp_path,
            """
            def decide(self, slot, view):
                if slot == "check_permission" and view.get("tool") == "add":
                    return {"kind": "Block", "reason": "legacy denial"}
                return {"kind": "Continue"}
        """,
        )
        try:
            steps = list(
                composable_loop(
                    MockLLMBackend(
                        [
                            '{"tool":"add","args":{"a":1,"b":2}}',
                            '{"tool":"done","args":{"answer":"ok"}}',
                        ]
                    ),
                    tools=_tools(),
                    hooks=[adapter],
                    config=LoopConfig(max_steps=2, use_native_tools=False),
                )
            )
            assert "legacy denial" in steps[0].tool_result.error
        finally:
            adapter.close()

    def test_subclass_named_method_override_is_not_bypassed(self):
        from looplet import HookDecision
        from looplet.hook_decision import _invoke_hook

        class Adapter(LEPHookAdapter):
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                return HookDecision(updated_args={"a": 4, "b": 6})

        adapter = Adapter(["unused"])
        result = _invoke_hook(adapter, "pre_dispatch", None, None, ToolCall("add"), 1)
        assert result.updated_args == {"a": 4, "b": 6}

    def test_fail_closed_denies_on_broken_server(self, tmp_path):
        # Server exits immediately → every RPC fails. fail_closed must deny.
        bad = tmp_path / "bad.py"
        bad.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
        adapter = LEPHookAdapter(server_argv(str(bad)), on_failure="fail_closed")
        adapter.pre_loop(None, None, None)
        try:
            assert adapter.check_permission(ToolCall(tool="x", args={}), None) is False
        finally:
            adapter.close()

    def test_close_reaps_stubborn_process_without_waiting_for_shutdown_response(self):
        class StubbornProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.stdout = io.StringIO()
                self.killed = False
                self.reaped = False

            def wait(self, timeout=None):
                if not self.killed:
                    raise subprocess.TimeoutExpired(["stubborn"], timeout)
                self.reaped = True

            def kill(self):
                self.killed = True

        adapter = LEPHookAdapter(["unused"])
        process = StubbornProcess()
        adapter._proc = process

        adapter.close()

        assert process.killed
        assert process.reaped
        assert adapter._proc is None

    @pytest.mark.parametrize(
        "response",
        [
            "[]\n",
            "1\n",
            '{"id":99,"result":{}}\n',
            '{"id":1,"error":"bad"}\n',
            '{"id":1,"result":[]}\n',
        ],
    )
    def test_malformed_response_applies_failure_policy(self, response):
        class Process:
            stdin = io.StringIO()
            stdout = io.StringIO(response)

        adapter = LEPHookAdapter(["unused"], on_failure="fail_closed")
        adapter._proc = Process()

        assert adapter.check_permission(ToolCall(tool="x", args={}), None) is False

    def test_fail_open_allows_on_broken_server(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
        adapter = LEPHookAdapter(server_argv(str(bad)), on_failure="fail_open")
        adapter.pre_loop(None, None, None)
        try:
            assert adapter.check_permission(ToolCall(tool="x", args={}), None) is True
        finally:
            adapter.close()


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="add",
            description="add",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    reg.register(
        ToolSpec(
            name="rm",
            description="rm",
            parameters={"path": "str"},
            execute=lambda *, path: {"removed": path},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="done",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("transport", ["python", "lep"])
@pytest.mark.parametrize(
    "scenario",
    [
        "args",
        "pre_context",
        "post_result",
        "post_stop",
        "should_stop",
        "deny_reason",
        "completion_stop",
        "block_only",
    ],
)
async def test_named_effects_conform_across_drivers_and_transports(
    tmp_path, async_mode, transport, scenario
):
    from looplet import Block, HookDecision, Stop, ToolResult, async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    cases = {
        "args": ("pre_dispatch", HookDecision(updated_args={"a": 4, "b": 6})),
        "pre_context": ("pre_dispatch", HookDecision(additional_context="context from hook")),
        "post_result": (
            "post_dispatch",
            HookDecision(updated_result=ToolResult("add", "", {"sum": 42})),
        ),
        "post_stop": ("post_dispatch", Stop("specific_stop")),
        "should_stop": ("should_stop", Stop("specific_stop")),
        "deny_reason": ("check_permission", Deny("specific denial")),
        "completion_stop": ("check_done", Stop("unsupported_completion_stop")),
        "block_only": ("pre_dispatch", Block("unsupported_tool_block")),
    }
    target_slot, effect = cases[scenario]

    class Policy:
        def decide(self, slot, call=None):
            if slot == target_slot and (
                slot in {"should_stop", "check_done"} or getattr(call, "tool", None) == "add"
            ):
                return effect
            return None

        def pre_dispatch(self, state, session_log, tool_call, step_num):
            return self.decide("pre_dispatch", tool_call)

        def check_permission(self, tool_call, state):
            return self.decide("check_permission", tool_call)

        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            return self.decide("post_dispatch", tool_call)

        def check_done(self, state, session_log, context, step_num):
            return self.decide("check_done")

        def should_stop(self, state, step_num, new_entities):
            return self.decide("should_stop")

    hook = (
        Policy()
        if transport == "python"
        else _adapter(
            tmp_path,
            f"""
        def decide(self, slot, view):
            if slot == {target_slot!r} and (slot in {{'should_stop', 'check_done'}} or view.get('tool') == 'add'):
                return {effect.to_wire()!r}
            return {{'kind': 'Continue'}}
        """,
            view=ViewSpec(fields=frozenset({"tool", "args"})),
        )
    )
    backend = (AsyncMockLLMBackend if async_mode else MockLLMBackend)(
        [
            '{"tool":"add","args":{"a":1,"b":2}}',
            '{"tool":"done","args":{"answer":"ok"}}',
        ]
    )
    state = DefaultState(max_steps=2)
    kwargs = dict(
        llm=backend,
        tools=_tools(),
        state=state,
        hooks=[hook],
        config=LoopConfig(max_steps=2, use_native_tools=False),
    )
    try:
        steps = (
            [step async for step in async_composable_loop(**kwargs)]
            if async_mode
            else list(composable_loop(**kwargs))
        )
        assert steps[0].tool_call.tool == "add"
        if scenario == "deny_reason":
            assert "specific denial" in steps[0].tool_result.error
        else:
            assert steps[0].tool_result.data == {
                "sum": 10 if scenario == "args" else 42 if scenario == "post_result" else 3
            }
        if scenario in {"post_stop", "should_stop"}:
            assert len(steps) == 1
            assert state._stop_reason == "specific_stop"
        else:
            assert steps[-1].tool_call.tool == "done"
            assert state._stop_reason == "done"
        if scenario == "pre_context":
            assert "context from hook" in backend.last_prompt
    finally:
        close = getattr(hook, "close", None)
        if close is not None:
            close()


class TestH4Composition:
    def test_inprocess_and_lep_hook_compose(self, tmp_path):
        """An in-process hook denying ``add`` and an LEP hook denying ``rm``
        compose: both calls are blocked in the same loop run."""
        lep = _adapter(
            tmp_path,
            """
            def decide(self, slot, view):
                if slot == "check_permission" and view.get("tool") == "rm":
                    return {"kind": "Deny", "block": "rm denied by lep"}
                return {"kind": "Continue"}
            """,
            view=ViewSpec(fields=frozenset({"tool", "args"})),
        )

        class DenyAdd:
            def check_permission(self, tool_call, state):
                return tool_call.tool != "add"

        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"rm","args":{"path":"/x"},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )
        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools(),
                state=DefaultState(max_steps=6),
                hooks=[DenyAdd(), lep],
                config=LoopConfig(max_steps=6),
            )
        )
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        rm_step = next(s for s in steps if s.tool_call.tool == "rm")
        # In-process hook blocked add; LEP hook blocked rm. Both honoured.
        assert add_step.tool_result.error is not None
        assert rm_step.tool_result.error is not None


class TestH2WideView:
    def test_lep_hook_reads_transcript_view(self, tmp_path):
        """A hook declaring a ``transcript`` view receives conversation
        history across the process boundary and can decide on it."""
        adapter = _adapter(
            tmp_path,
            """
            def decide(self, slot, view):
                if slot == "pre_prompt":
                    t = view.get("transcript") or []
                    return {"kind": "InjectContext",
                            "text": f"[transcript_len={len(t)}]"}
                return {"kind": "Continue"}
            """,
            view=ViewSpec(fields=frozenset({"transcript"}), fidelity="full"),
        )
        adapter.pre_loop(None, None, None)
        try:
            out = adapter.pre_prompt(None, None, None, 0)
            assert out is not None
            assert out.startswith("[transcript_len=")
        finally:
            adapter.close()
