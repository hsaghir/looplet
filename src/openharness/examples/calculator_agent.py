"""Calculator agent example — demonstrates openharness with a math problem-solving agent.

Uses a mock LLM that returns scripted tool calls. Shows the basic pattern:
  1. Create a ToolRegistry with domain tools
  2. Create an LLMBackend (mock or real)
  3. Run composable_loop, consuming yielded Steps
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

from openharness.loop import LoopConfig, composable_loop
from openharness.tools import BaseToolRegistry, ToolSpec

# ── Safe arithmetic evaluator (no eval()) ────────────────────────

_OPS: dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a simple binary arithmetic expression without eval().

    Supports expressions like ``"2+3"``, ``"5 * 4"``, ``"10 / 2"``.
    Only one operator is supported per call (no chaining).
    """
    expr = expression.strip()
    for op_char, op_fn in _OPS.items():
        if op_char in expr:
            parts = expr.split(op_char, 1)
            if len(parts) == 2:
                left = float(parts[0].strip())
                right = float(parts[1].strip())
                return float(op_fn(left, right))
    # Bare number
    return float(expr)

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
    last_result: float = 0.0
    max_steps: int = 10

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - len(self.steps))

    def context_summary(self) -> str:
        return f"Calculator: last_result={self.last_result}, steps={len(self.steps)}"

    def snapshot(self) -> dict[str, Any]:
        return {"steps": len(self.steps), "last_result": self.last_result}


# ── Tool Registry ─────────────────────────────────────────────────


def _make_calc_registry(state: CalcState) -> BaseToolRegistry:
    reg = BaseToolRegistry()

    def _calculate(expression: str = "") -> dict[str, Any]:
        try:
            result = _safe_eval_arithmetic(expression)
            state.last_result = result
            return {"result": result, "expression": expression}
        except (ValueError, ZeroDivisionError) as exc:
            return {"error": str(exc)}

    def _get_result() -> dict[str, Any]:
        return {"last_result": state.last_result}

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
