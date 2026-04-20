"""Smoke tests for ApprovalHook — async approval pattern."""
from __future__ import annotations

import tempfile

import pytest

from openharness import (
    ApprovalHook,
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from openharness.approval import ApprovalRequest
from openharness.testing import MockLLMBackend
from openharness.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools():
    reg = BaseToolRegistry()

    def risky_action(*, target: str) -> dict:
        """A tool that requires approval for dangerous targets."""
        if "production" in target.lower():
            return {
                "needs_approval": True,
                "approval_description": f"Deploy to {target}?",
                "approval_options": ["approve", "deny"],
            }
        return {"deployed": target}

    reg.register(ToolSpec(
        name="deploy", description="Deploy to a target",
        parameters={"target": "str"}, execute=risky_action,
    ))
    reg.register(ToolSpec(
        name="done", description="d",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


class TestApprovalHook:
    def test_stops_loop_on_needs_approval(self):
        hook = ApprovalHook()
        steps = list(composable_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"deploy","args":{"target":"production-us-east"},"reasoning":"r"}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(), state=DefaultState(max_steps=5),
            hooks=[hook], config=LoopConfig(max_steps=5),
        ))
        # Loop should stop after the deploy step
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "deploy"
        # Pending approval recorded
        assert hook.pending is not None
        assert hook.pending.tool == "deploy"
        assert hook.pending.description == "Deploy to production-us-east?"
        assert "approve" in hook.pending.options

    def test_no_stop_when_no_approval_needed(self):
        hook = ApprovalHook()
        steps = list(composable_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"deploy","args":{"target":"staging"},"reasoning":"r"}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(), state=DefaultState(max_steps=5),
            hooks=[hook], config=LoopConfig(max_steps=5),
        ))
        # Non-production deploy doesn't need approval → loop continues
        assert len(steps) == 2
        assert hook.pending is None

    def test_checkpoint_dir_enables_resume(self):
        """checkpoint_dir + ApprovalHook = crash-safe async approval."""
        hook = ApprovalHook()
        with tempfile.TemporaryDirectory() as tmpdir:
            steps = list(composable_loop(
                llm=MockLLMBackend(responses=[
                    '{"tool":"deploy","args":{"target":"production"},"reasoning":"r"}',
                ]),
                tools=_tools(), state=DefaultState(max_steps=5),
                hooks=[hook],
                config=LoopConfig(max_steps=5, checkpoint_dir=tmpdir),
            ))
            assert len(steps) == 1
            assert hook.pending is not None
            # Checkpoint was saved
            import os
            ckpt_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            assert len(ckpt_files) >= 1

    def test_approval_request_dataclass(self):
        req = ApprovalRequest(step=3, tool="deploy", description="Deploy?")
        assert req.step == 3
        assert req.options == ["approve", "deny"]
