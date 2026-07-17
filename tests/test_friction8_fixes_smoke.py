"""Round-8 friction fixes: MCP leak + state.conversation + BudgetTelemetry."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from looplet import BaseToolRegistry, DefaultState, LoopConfig, composable_loop
from looplet.budget import BudgetTelemetry, ContextBudget
from looplet.conversation import Conversation, Message
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


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


class TestStateStashesConversation:
    def test_conversation_visible_on_state(self):
        captured: dict = {}

        class _Spy:
            def pre_loop(self, state, session_log, context):
                captured["conv"] = getattr(state, "conversation", None)

        conv = Conversation()
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']
                ),
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
                task={},
                conversation=conv,
                hooks=[_Spy()],
            )
        )
        assert captured["conv"] is conv


class TestBudgetTelemetryUsesConversation:
    def test_non_zero_estimate_with_conversation(self):
        budget = ContextBudget(context_window=1000, warning_at=600, error_at=800, compact_buffer=50)
        telem = BudgetTelemetry(budget)
        conv = Conversation()
        # Seed with enough content to register a measurable estimate.
        conv.append(Message(role="user", content="x" * 2000))
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']
                ),
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
                task={},
                conversation=conv,
                hooks=[telem],
            )
        )
        # Without the fix, estimate would be 1 (session_log empty).
        # With conversation access: ~500 tokens from 2000 chars.
        assert telem.samples, "expected at least one sample"
        assert telem.samples[0][2] > 100, (
            f"expected >100 tokens from seeded conv, got {telem.samples[0][2]}"
        )


class TestMCPStartupCleanup:
    def test_bad_command_does_not_leak_subprocess(self):
        from looplet.mcp import MCPToolAdapter

        # Spawn a shell that exits immediately - init will fail since
        # the server never replies. Repeat to cover both EOF and the
        # early-exit BrokenPipe race; neither may leak a process/stream.
        for _ in range(10):
            adapter = MCPToolAdapter("true", timeout=1.0)
            with pytest.raises(RuntimeError, match="failed to initialize"):
                adapter._ensure_started()
            assert adapter._proc is None

    @pytest.mark.parametrize("output", ["", "not-json\n"])
    def test_live_unresponsive_server_honors_timeout(
        self,
        output: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ):
        from looplet import mcp
        from looplet.mcp import MCPToolAdapter

        server = tmp_path / "unresponsive.py"
        pid_file = tmp_path / "child.pid"
        server.write_text(
            f"import os, sys, time\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            f"sys.stdout.write({output!r})\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        spawned: list[subprocess.Popen] = []
        real_popen = subprocess.Popen

        def capture_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(mcp.subprocess, "Popen", capture_popen)
        adapter = MCPToolAdapter(f'"{sys.executable}" "{server}"', timeout=0.1)
        started = time.monotonic()

        with pytest.raises(RuntimeError, match="failed to initialize"):
            adapter._ensure_started()

        assert time.monotonic() - started < 2.0
        assert adapter._proc is None
        assert len(spawned) == 1
        assert spawned[0].poll() is not None
        if os.name == "posix":
            child_pid = int(pid_file.read_text())
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                pytest.fail(f"MCP child process {child_pid} survived timeout cleanup")
