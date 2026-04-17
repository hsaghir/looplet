"""Research agent example — demonstrates openharness with a simulated web research agent.

Shows:
  - SessionLog for entity/memory tracking
  - Theory tracking (agent updates theory as it learns)
  - Multi-step research workflow: search → read → notes → done
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openharness.loop import LoopConfig, composable_loop
from openharness.session import SessionLog
from openharness.tools import BaseToolRegistry, ToolSpec

# ── Mock LLM ─────────────────────────────────────────────────────


class MockResearchLLM:
    """Scripted LLM that simulates a research workflow.

    Scripted flow:
      1. search('python async best practices')
      2. read_page('https://docs.python.org/asyncio')
      3. take_notes('Use asyncio.gather for concurrent I/O')
      4. done(report='Async best practice: use asyncio.gather')
    """

    _RESPONSES = [
        '{"tool": "search", "args": {"query": "python async best practices"}}',
        '{"tool": "read_page", "args": {"url": "https://docs.python.org/asyncio"}}',
        '{"tool": "take_notes", "args": {"text": "Use asyncio.gather for concurrent I/O"}}',
        '{"tool": "done", "args": {"report": "Async best practice: use asyncio.gather for concurrent I/O operations"}}',
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
class ResearchState:
    """Agent state for the research agent — tracks notes and theory."""

    steps: list = field(default_factory=list)
    queries_used: int = 0
    notes: list[str] = field(default_factory=list)
    theory: str = "unknown"
    _max_steps: int = 15

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_steps - len(self.steps))

    def context_summary(self) -> str:
        return f"Research: theory='{self.theory}', notes={len(self.notes)}, steps={len(self.steps)}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": len(self.steps),
            "theory": self.theory,
            "notes": self.notes[:],
        }


# ── Tool Registry ─────────────────────────────────────────────────


_MOCK_SEARCH_RESULTS = [
    {"title": "Python asyncio docs", "url": "https://docs.python.org/asyncio"},
    {"title": "Real Python — Async IO", "url": "https://realpython.com/async-io-python"},
]

_MOCK_PAGE_CONTENT = {
    "https://docs.python.org/asyncio": (
        "asyncio is a library to write concurrent code using the async/await syntax. "
        "Use asyncio.gather() to run multiple coroutines concurrently."
    ),
    "https://realpython.com/async-io-python": (
        "asyncio.gather() is the recommended way to run many coroutines concurrently. "
        "Avoid blocking calls inside async functions."
    ),
}


def _make_research_registry(state: ResearchState) -> BaseToolRegistry:
    reg = BaseToolRegistry()
    step_counter: list[int] = [0]

    def _search(query: str = "") -> dict[str, Any]:
        results = _MOCK_SEARCH_RESULTS
        step_counter[0] += 1
        return {"query": query, "results": results}

    def _read_page(url: str = "") -> dict[str, Any]:
        content = _MOCK_PAGE_CONTENT.get(url, f"[no content found for {url}]")
        # Update theory based on reading
        state.theory = "asyncio.gather is the best practice for concurrent I/O"
        step_counter[0] += 1
        return {"url": url, "content": content}

    def _take_notes(text: str = "") -> dict[str, Any]:
        state.notes.append(text)
        step_counter[0] += 1
        return {"recorded": text, "total_notes": len(state.notes)}

    def _done(report: str = "") -> dict[str, Any]:
        return {"report": report, "notes_used": len(state.notes)}

    reg.register(ToolSpec(
        name="search",
        description="Search for pages on a topic",
        parameters={"query": "search query string"},
        execute=_search,
    ))
    reg.register(ToolSpec(
        name="read_page",
        description="Fetch and read the content of a web page",
        parameters={"url": "URL to read"},
        execute=_read_page,
    ))
    reg.register(ToolSpec(
        name="take_notes",
        description="Record an important finding to memory",
        parameters={"text": "note text to record"},
        execute=_take_notes,
        free=True,
    ))
    reg.register(ToolSpec(
        name="done",
        description="Finish and produce a research report",
        parameters={"report": "final research summary report"},
        execute=_done,
    ))
    return reg


# ── Entry point ───────────────────────────────────────────────────


def run() -> None:
    """Run the research agent and print each step including session log."""
    print("=== Research Agent ===")
    state = ResearchState(theory="unknown")
    session_log = SessionLog()
    reg = _make_research_registry(state)
    llm = MockResearchLLM()
    config = LoopConfig(max_steps=10, done_tool="done")

    for step in composable_loop(llm, tools=reg, config=config, state=state, session_log=session_log):
        print(f"Step {step.number}: {step.tool_call.tool}({step.tool_call.args})")
        if step.tool_result.error:
            print(f"  ERROR: {step.tool_result.error}")
        else:
            print(f"  → {step.tool_result.data}")
        print(f"  theory: {state.theory}")
        if step.tool_call.tool == "done":
            report = step.tool_call.args.get("report", "")
            print(f"\nFinal report: {report}")

    print(f"\nNotes collected: {state.notes}")
    print(f"Session log entries: {len(session_log.entries)}")
    print("=== Done ===")


if __name__ == "__main__":
    run()
