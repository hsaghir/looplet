"""Calculator agent example — demonstrates cadence with a math problem-solving agent.

Uses a mock LLM that returns scripted tool calls. Shows the basic pattern:
  1. Create a ToolRegistry with domain tools
  2. Create an LLMBackend (mock or real)
  3. Run composable_loop, consuming yielded Steps
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openharness.loop import LoopConfig, composable_loop
from openharness.tools import BaseToolRegistry, ToolSpec
from openharness.types import LLMBackend


# ── Mock LLM ─────────────────────────────────────────────────────


class MockMathLLM:
    """Scripted LLM that returns tool calls for a simple math problem.

    Scripted flow:
      step 1 → calculate('2+3')   # result: 5
      step 2 → calculate('5*4')   # result: 20; but we demonstrate using get_result
      step 3 → done(answer=25)    # 5 + 20 = 25
    """

    _RESPONSES = [
        '{"tool": "calculate", "args": {"expression": "2+3"}}',
        '{"tool": "calculate", "args": {"expression": "5*4"}}',
        '{"tool": "done", "args": {"answer": "25"}}',
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
class CalcState:
    """Minimal AgentState for the calculator agent."""

    steps: list = field(default_factory=list)
    queries_used: int = 0
    _last_result: float = 0.0
    _max_steps: int = 10

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_steps - len(self.steps))

    def context_summary(self) -> str:
        return f"Calculator: last_result={self._last_result}, steps={len(self.steps)}"

    def snapshot(self) -> dict[str, Any]:
        return {"steps": len(self.steps), "last_result": self._last_result}


# ── Tool Registry ─────────────────────────────────────────────────


def _make_calc_registry(state: CalcState) -> BaseToolRegistry:
    reg = BaseToolRegistry()

    def _calculate(expression: str = "") -> dict[str, Any]:
        try:
            # Safe evaluation of simple arithmetic only
            allowed = set("0123456789+-*/()., ")
            if not all(c in allowed for c in expression):
                return {"error": "unsafe expression"}
            result = eval(expression)  # noqa: S307 — constrained above
            state._last_result = float(result)
            return {"result": result, "expression": expression}
        except Exception as exc:
            return {"error": str(exc)}

    def _get_result() -> dict[str, Any]:
        return {"last_result": state._last_result}

    def _done(answer: str = "") -> dict[str, Any]:
        return {"final_answer": answer, "status": "complete"}

    reg.register(ToolSpec(
        name="calculate",
        description="Evaluate a simple arithmetic expression",
        parameters={"expression": "arithmetic expression like '2+3'"},
        execute=_calculate,
    ))
    reg.register(ToolSpec(
        name="get_result",
        description="Return the last calculation result",
        parameters={},
        execute=_get_result,
        free=True,
    ))
    reg.register(ToolSpec(
        name="done",
        description="Finish with a final answer",
        parameters={"answer": "the final numeric answer"},
        execute=_done,
    ))
    return reg


# ── Entry point ───────────────────────────────────────────────────


def run() -> None:
    """Run the calculator agent and print each step."""
    print("=== Calculator Agent ===")
    state = CalcState()
    llm = MockMathLLM()
    reg = _make_calc_registry(state)
    config = LoopConfig(max_steps=10, done_tool="done")

    for step in composable_loop(llm, tools=reg, config=config, state=state):
        print(f"Step {step.number}: {step.tool_call.tool}({step.tool_call.args})")
        if step.tool_result.error:
            print(f"  ERROR: {step.tool_result.error}")
        else:
            print(f"  → {step.tool_result.data}")
        if step.tool_call.tool == "done":
            answer = step.tool_call.args.get("answer", "?")
            print(f"\nFinal answer: {answer}")
    print("=== Done ===")


if __name__ == "__main__":
    run()
