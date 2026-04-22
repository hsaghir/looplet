"""Integration: LoopConfig.memory_sources flow through to default prompt."""

from __future__ import annotations

from looplet.loop import LoopConfig, composable_loop
from looplet.memory import CallableMemorySource, StaticMemorySource
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import DefaultState, LLMBackend


class _CapturingLLM(LLMBackend):
    def __init__(self) -> None:
        self.last_prompt: str = ""

    def generate(self, prompt: str, *, max_tokens: int = 2000,
                 system_prompt: str = "", temperature: float = 0.2) -> str:
        self.last_prompt = prompt
        return '```json\n{"tool": "done", "args": {"summary": "x"}}\n```'


def _reg() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="done", description="finish", parameters={"summary": "s"},
        execute=lambda summary="": {"done": True, "summary": summary},
    ))
    return reg


class TestLoopRendersMemory:
    def test_static_memory_source_reaches_prompt(self):
        llm = _CapturingLLM()
        cfg = LoopConfig(
            max_steps=2,
            memory_sources=[StaticMemorySource("Always use UTC timestamps.")],
        )
        list(composable_loop(
            llm=llm, task={"id": "T-1"}, tools=_reg(),
            config=cfg, state=DefaultState(max_steps=2),
        ))
        assert "═══ MEMORY ═══" in llm.last_prompt
        assert "Always use UTC timestamps." in llm.last_prompt

    def test_callable_memory_receives_state(self):
        llm = _CapturingLLM()
        captured = []
        cfg = LoopConfig(
            max_steps=2,
            memory_sources=[
                CallableMemorySource(lambda s: f"step={s.step_count}; budget={s.budget_remaining}"),
            ],
        )
        state = DefaultState(max_steps=2)
        list(composable_loop(
            llm=llm, task={"id": "T-1"}, tools=_reg(),
            config=cfg, state=state,
        ))
        # First turn: step_count starts at 0
        assert "step=0" in llm.last_prompt
        assert "budget=" in llm.last_prompt
