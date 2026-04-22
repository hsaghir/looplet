"""Tests for the approval() protocol on ToolContext.

Tools can opt-in to request caller input mid-execution by accepting
``ctx`` and calling ``ctx.approve(prompt, options)``. The handler
is installed via ``LoopConfig.approval_handler``; in headless runs the
method returns ``None`` so tools can proceed unattended.
"""

from __future__ import annotations

from looplet.loop import LoopConfig, composable_loop
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import DefaultState, LLMBackend, ToolContext


class TestToolContextApproval:
    def test_approve_returns_none_without_handler(self):
        ctx = ToolContext()
        assert ctx.approve("pick one", ["a", "b"]) is None

    def test_approve_calls_handler(self):
        captured = {}

        def handler(prompt, options):
            captured["prompt"] = prompt
            captured["options"] = options
            return "answer"

        ctx = ToolContext(request_approval=handler)
        out = ctx.approve("question?", ["a", "b"])
        assert out == "answer"
        assert captured == {"prompt": "question?", "options": ["a", "b"]}


class _LLM(LLMBackend):
    def __init__(self, *scripts):
        self.s = list(scripts)
        self.n = 0

    def generate(self, prompt: str, **kw) -> str:
        s = self.s[self.n]
        self.n += 1
        return s


class TestLoopPlumbsApproval:
    def test_tool_receives_approval_handler(self):
        seen: dict = {}

        def tool(ctx: ToolContext, **kw):
            reply = ctx.approve("confirm?", ["y", "n"])
            seen["reply"] = reply
            return {"ok": True, "reply": reply}

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="confirm",
                description="",
                parameters={},
                execute=tool,
                concurrent_safe=False,
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="",
                parameters={"summary": "s"},
                execute=lambda summary="": {"done": True, "summary": summary},
            )
        )

        handler = lambda prompt, options: "y"
        cfg = LoopConfig(max_steps=3, approval_handler=handler)
        llm = _LLM(
            '```json\n{"tool": "confirm", "args": {}}\n```',
            '```json\n{"tool": "done", "args": {"summary": "x"}}\n```',
        )
        list(
            composable_loop(
                llm=llm,
                task={"id": "T"},
                tools=reg,
                config=cfg,
                state=DefaultState(max_steps=3),
            )
        )
        assert seen == {"reply": "y"}
