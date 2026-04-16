"""Code review agent example — demonstrates cadence with a simulated code review agent.

Shows:
  - QualityGateHook: custom LoopHook that blocks done() until all files are reviewed
  - StreamingHook with CallbackEmitter for real-time event printing
  - acceptance_criteria in LoopConfig
  - Multi-hook composition (both hooks active simultaneously)
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from openharness.loop import LoopConfig, LoopHook, composable_loop
from openharness.session import SessionLog
from openharness.streaming import CallbackEmitter, StreamingHook
from openharness.tools import BaseToolRegistry, ToolSpec
from openharness.types import ToolCall, ToolResult


# ── Mock LLM ─────────────────────────────────────────────────────


class MockReviewLLM:
    """Scripted LLM that simulates a code review workflow.

    Scripted flow:
      1. list_files('src/')
      2. read_file('src/main.py')
      3. add_comment(file='src/main.py', line='10', comment='Missing type hint')
      4. read_file('src/utils.py')
      5. add_comment(file='src/utils.py', line='5', comment='Unused import')
      6. done(summary='Found 2 issues: missing type hints and unused imports')
    """

    _RESPONSES = [
        '{"tool": "list_files", "args": {"path": "src/"}}',
        '{"tool": "read_file", "args": {"path": "src/main.py"}}',
        '{"tool": "add_comment", "args": {"file": "src/main.py", "line": "10", "comment": "Missing type hint on return value"}}',
        '{"tool": "read_file", "args": {"path": "src/utils.py"}}',
        '{"tool": "add_comment", "args": {"file": "src/utils.py", "line": "5", "comment": "Unused import: os"}}',
        '{"tool": "done", "args": {"summary": "Found 2 issues: missing type hints and unused imports"}}',
    ]

    def __init__(self) -> None:
        self._idx = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        response = self._RESPONSES[min(self._idx, len(self._RESPONSES) - 1)]
        self._idx += 1
        return response


# ── State ─────────────────────────────────────────────────────────


@dataclass
class ReviewState:
    """Agent state for the code review agent — tracks files reviewed and comments."""

    steps: list = field(default_factory=list)
    queries_used: int = 0
    files_listed: list[str] = field(default_factory=list)
    files_reviewed: set[str] = field(default_factory=set)
    comments: list[dict[str, str]] = field(default_factory=list)
    _max_steps: int = 20

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_steps - len(self.steps))

    def context_summary(self) -> str:
        return (
            f"Review: files={len(self.files_listed)}, "
            f"reviewed={len(self.files_reviewed)}, "
            f"comments={len(self.comments)}"
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": len(self.steps),
            "files_listed": self.files_listed[:],
            "files_reviewed": list(self.files_reviewed),
            "comments": len(self.comments),
        }


# ── Quality Gate Hook ─────────────────────────────────────────────


class QualityGateHook:
    """Custom LoopHook that enforces review quality before allowing done().

    Blocks done() if any listed file has not been reviewed yet.
    Returns None (allow) only when all listed files have been read.
    """

    def __init__(self, state: ReviewState) -> None:
        self._state = state

    def pre_prompt(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        return None

    def pre_dispatch(
        self,
        state: Any,
        session_log: SessionLog,
        tool_call: ToolCall,
        step_num: int,
    ) -> ToolResult | None:
        return None

    def post_dispatch(
        self,
        state: Any,
        session_log: SessionLog,
        tool_call: ToolCall,
        tool_result: ToolResult,
        step_num: int,
    ) -> str | None:
        return None

    def check_done(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Block done() if not all listed files have been reviewed."""
        unreviewed = set(self._state.files_listed) - self._state.files_reviewed
        if unreviewed:
            return f"QualityGate: must review all files first. Unreviewed: {sorted(unreviewed)}"
        return None

    def should_stop(
        self,
        state: Any,
        step_num: int,
        new_entities: int,
    ) -> bool:
        return False

    def on_loop_end(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
        llm: Any,
    ) -> int:
        return 0


# ── Tool Registry ─────────────────────────────────────────────────


_MOCK_FILES = {
    "src/": ["src/main.py", "src/utils.py"],
}

_MOCK_CODE = {
    "src/main.py": (
        "def greet(name):\n"
        "    return f'Hello, {name}'\n"
        "\n"
        "# line 10: missing return type\n"
        "def compute(x, y):\n"
        "    return x + y\n"
    ),
    "src/utils.py": (
        "import os\n"
        "import sys\n"
        "\n"
        "# line 5: os is not used below\n"
        "def format_path(p: str) -> str:\n"
        "    return sys.path[0] + '/' + p\n"
    ),
}


def _make_review_registry(state: ReviewState) -> BaseToolRegistry:
    reg = BaseToolRegistry()

    def _list_files(path: str = "") -> dict[str, Any]:
        files = _MOCK_FILES.get(path, [f"{path}example.py"])
        state.files_listed.extend(files)
        return {"path": path, "files": files}

    def _read_file(path: str = "") -> dict[str, Any]:
        content = _MOCK_CODE.get(path, f"# empty file: {path}\n")
        state.files_reviewed.add(path)
        return {"path": path, "content": content}

    def _add_comment(file: str = "", line: str = "", comment: str = "") -> dict[str, Any]:
        entry = {"file": file, "line": line, "comment": comment}
        state.comments.append(entry)
        return {"recorded": entry, "total_comments": len(state.comments)}

    def _done(summary: str = "") -> dict[str, Any]:
        return {"summary": summary, "total_comments": len(state.comments)}

    reg.register(ToolSpec(
        name="list_files",
        description="List files in a directory",
        parameters={"path": "directory path to list"},
        execute=_list_files,
    ))
    reg.register(ToolSpec(
        name="read_file",
        description="Read the content of a source file",
        parameters={"path": "file path to read"},
        execute=_read_file,
    ))
    reg.register(ToolSpec(
        name="add_comment",
        description="Add a review comment on a file at a specific line",
        parameters={
            "file": "file path",
            "line": "line number",
            "comment": "review comment text",
        },
        execute=_add_comment,
        free=True,
    ))
    reg.register(ToolSpec(
        name="done",
        description="Finish the review with a summary",
        parameters={"summary": "overall review summary"},
        execute=_done,
    ))
    return reg


# ── Entry point ───────────────────────────────────────────────────


def run() -> None:
    """Run the code review agent with streaming events and quality gate."""
    print("=== Code Review Agent ===")

    state = ReviewState()
    session_log = SessionLog()
    reg = _make_review_registry(state)
    llm = MockReviewLLM()

    # StreamingHook with CallbackEmitter — prints each event
    emitter = CallbackEmitter(callback=lambda evt: print(f"  [event] {evt.event_type}"))
    streaming_hook = StreamingHook(emitter=emitter)

    # QualityGateHook — blocks done() until all files reviewed
    quality_hook = QualityGateHook(state=state)

    config = LoopConfig(
        max_steps=15,
        done_tool="done",
        acceptance_criteria=["All listed files must be reviewed before submitting"],
    )

    for step in composable_loop(
        llm,
        tools=reg,
        config=config,
        state=state,
        hooks=[quality_hook, streaming_hook],
        session_log=session_log,
    ):
        print(f"Step {step.number}: {step.tool_call.tool}({step.tool_call.args})")
        if step.tool_result.error:
            print(f"  ERROR: {step.tool_result.error}")
        else:
            print(f"  → {step.tool_result.data}")
        if step.tool_call.tool == "done":
            summary = step.tool_call.args.get("summary", "")
            print(f"\nReview complete: {summary}")

    print(f"\nComments added: {len(state.comments)}")
    for c in state.comments:
        print(f"  {c['file']}:{c['line']} — {c['comment']}")
    print("=== Done ===")


if __name__ == "__main__":
    run()
