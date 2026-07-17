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

import os
import textwrap
from pathlib import Path

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
