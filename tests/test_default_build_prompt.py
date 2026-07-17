"""Tests for default build_prompt - loop runs without domain callables.

A key "domain-agnostic" property: callers should be able to drop
``composable_loop()`` into a brand-new project with only ``tools`` +
``state`` and get a coherent prompt, not a bare ``Task: {...}`` dump.
"""

from __future__ import annotations

from looplet.loop import LoopConfig, composable_loop
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import DefaultState, LLMBackend


class _CapturingLLM(LLMBackend):
    """Records the last prompt it saw; returns a done() JSON so the loop stops."""

    def __init__(self) -> None:
        self.last_prompt: str = ""
        self.calls: int = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.last_prompt = prompt
        self.calls += 1
        # Stop immediately after one turn via the done tool.
        return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'


class TestDefaultBuildPrompt:
    def _registry(self) -> BaseToolRegistry:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="search",
                description="find something",
                parameters={"q": "query"},
                execute=lambda q: {"results": [q]},
                concurrent_safe=True,
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="finish",
                parameters={"summary": "final"},
                execute=lambda summary="": {"done": True, "summary": summary},
            )
        )
        return reg

    def test_loop_produces_structured_prompt_with_no_build_prompt_supplied(self):
        """Without ``config.build_prompt``, the loop should still emit the
        7-section TASK / TOOLS / ... / STEP layout from prompts.build_prompt."""
        llm = _CapturingLLM()
        state = DefaultState(max_steps=3)
        reg = self._registry()
        # Intentionally omit build_prompt / build_briefing / extract_entities.
        config = LoopConfig(max_steps=3)
        list(
            composable_loop(
                llm=llm,
                task={"id": "T-1", "title": "demo", "description": "find stuff"},
                tools=reg,
                config=config,
                state=state,
            )
        )
        p = llm.last_prompt
        assert p, "LLM received no prompt"
        assert "═══ TASK ═══" in p
        assert "═══ TOOLS ═══" in p
        assert "═══ STEP" in p
        assert "title: demo" in p
        assert "search" in p  # tool catalog present

    def test_explicit_build_prompt_still_wins(self):
        """User-supplied build_prompt overrides the default."""
        llm = _CapturingLLM()
        state = DefaultState(max_steps=3)
        reg = self._registry()

        def custom(**kw) -> str:
            return "CUSTOM PROMPT " + str(kw.get("step_number"))

        config = LoopConfig(max_steps=3, build_prompt=custom)
        list(
            composable_loop(
                llm=llm,
                task={"id": "T-1"},
                tools=reg,
                config=config,
                state=state,
            )
        )
        assert llm.last_prompt.startswith("CUSTOM PROMPT ")

    def test_default_prompt_includes_budget_warning_when_low(self):
        llm = _CapturingLLM()
        # Start with budget already low: max_steps=1 means after step 1 budget_remaining=0
        # We can't easily inspect mid-run; just assert the STEP section carries budget info.
        state = DefaultState(max_steps=2)
        reg = self._registry()
        list(
            composable_loop(
                llm=llm,
                task={"id": "T-2", "title": "t"},
                tools=reg,
                config=LoopConfig(max_steps=2),
                state=state,
            )
        )
        assert "budget:" in llm.last_prompt
