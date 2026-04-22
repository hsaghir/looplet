"""Tests for looplet.loop and looplet.parse."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.smoke


# ── Minimal test helpers ─────────────────────────────────────────


class SimpleState:
    """Minimal AgentState-compatible state for tests."""

    def __init__(self, max_steps: int = 15) -> None:
        self.steps: list = []
        self.queries_used: int = 0
        self._max_steps = max_steps

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_steps - len(self.steps))

    def context_summary(self) -> str:
        return f"steps={len(self.steps)}"

    def snapshot(self) -> dict:
        return {"steps": len(self.steps)}


def _make_done_llm(pre_responses: list[str], done_summary: str = "all done") -> Any:
    """LLM that returns pre_responses then done() JSON."""
    from tests.conftest import MockLLMBackend
    done_json = f'{{"tool": "done", "args": {{"summary": "{done_summary}"}}}}'
    return MockLLMBackend(pre_responses + [done_json])


def _make_registry_with_done(*extra_tools):
    """Build a registry with a done tool and optional extras."""
    from looplet.tools import BaseToolRegistry, ToolSpec

    reg = BaseToolRegistry()

    def _done_fn(**kwargs):
        return {"ok": True, "summary": kwargs.get("summary", "")}

    reg.register(ToolSpec(
        name="done",
        description="Signal task completion.",
        parameters={"summary": "Summary of what was accomplished."},
        execute=_done_fn,
    ))
    for spec in extra_tools:
        reg.register(spec)
    return reg


# ══════════════════════════════════════════════════════════════════
# parse.py tests
# ══════════════════════════════════════════════════════════════════


class TestParseToolCall:
    def test_simple_json(self):
        from looplet.parse import parse_tool_call
        tc = parse_tool_call('{"tool": "search", "args": {"q": "hello"}}')
        assert tc is not None
        assert tc.tool == "search"
        assert tc.args["q"] == "hello"

    def test_markdown_fence(self):
        from looplet.parse import parse_tool_call
        raw = '```json\n{"tool": "search", "args": {}}\n```'
        tc = parse_tool_call(raw)
        assert tc is not None
        assert tc.tool == "search"

    def test_invalid_returns_none(self):
        from looplet.parse import parse_tool_call
        assert parse_tool_call("not json at all") is None

    def test_no_tool_key(self):
        from looplet.parse import parse_tool_call
        assert parse_tool_call('{"foo": "bar"}') is None

    def test_reasoning_captured(self):
        from looplet.parse import parse_tool_call
        tc = parse_tool_call('{"tool": "think", "args": {}, "reasoning": "why not"}')
        assert tc is not None
        assert tc.reasoning == "why not"


class TestParseMultiToolCalls:
    def test_single_tool(self):
        from looplet.parse import parse_multi_tool_calls
        calls = parse_multi_tool_calls('{"tool": "search", "args": {"q": "x"}}')
        assert len(calls) == 1
        assert calls[0].tool == "search"

    def test_multi_tool_format(self):
        from looplet.parse import parse_multi_tool_calls
        raw = '{"tools": [{"tool": "search", "args": {}}, {"tool": "think", "args": {}}]}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0].tool == "search"
        assert calls[1].tool == "think"

    def test_empty_on_invalid(self):
        from looplet.parse import parse_multi_tool_calls
        assert parse_multi_tool_calls("bad text") == []

    def test_theory_propagated_to_args(self):
        from looplet.parse import parse_multi_tool_calls
        raw = '{"theory": "my-theory", "tools": [{"tool": "search", "args": {}}]}'
        calls = parse_multi_tool_calls(raw)
        assert calls[0].args.get("__theory__") == "my-theory"

    def test_markdown_fenced(self):
        from looplet.parse import parse_multi_tool_calls
        raw = '```json\n{"tool": "done", "args": {}}\n```'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "done"


class TestParseNativeToolUse:
    def test_parses_tool_use_blocks(self):
        from looplet.parse import parse_native_tool_use
        blocks = [
            {"type": "tool_use", "id": "abc", "name": "search", "input": {"q": "x"}},
        ]
        calls = parse_native_tool_use(blocks)
        assert len(calls) == 1
        assert calls[0].tool == "search"
        assert calls[0].args == {"q": "x"}

    def test_skips_non_tool_use(self):
        from looplet.parse import parse_native_tool_use
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "x", "name": "think", "input": {}},
        ]
        calls = parse_native_tool_use(blocks)
        assert len(calls) == 1

    def test_empty_on_no_blocks(self):
        from looplet.parse import parse_native_tool_use
        assert parse_native_tool_use([]) == []

    def test_no_domain_specific_imports(self):
        import inspect

        import looplet.parse as m
        assert "primal_security" not in inspect.getsource(m)


# ══════════════════════════════════════════════════════════════════
# LoopConfig tests
# ══════════════════════════════════════════════════════════════════


class TestLoopConfig:
    def test_defaults(self):
        from looplet.loop import LoopConfig
        c = LoopConfig()
        assert c.max_steps == 15
        assert c.max_tokens == 2000
        assert c.temperature == 0.2
        assert c.recovery_temperature == 0.1
        assert c.done_tool == "done"
        assert c.use_native_tools is False
        assert c.acceptance_criteria is None
        assert c.max_briefing_tokens is None
        assert c.build_briefing is None
        assert c.build_trace is None
        assert c.extract_step_metadata is None
        assert c.extract_entities is None
        assert c.build_prompt is None

    def test_no_alert_exploration_params(self):
        """Backward-compat params must not exist."""
        import inspect

        from looplet.loop import composable_loop
        sig = inspect.signature(composable_loop)
        assert "alert" not in sig.parameters
        assert "exploration" not in sig.parameters


# ══════════════════════════════════════════════════════════════════
# LoopHook protocol tests
# ══════════════════════════════════════════════════════════════════


class TestLoopHookProtocol:
    def test_protocol_runtime_checkable(self):
        from looplet.loop import LoopHook

        class MinimalHook:
            def pre_loop(self, state, session_log, context):
                pass

            def pre_prompt(self, state, session_log, context, step_num):
                return None

            def pre_dispatch(self, state, session_log, tool_call, step_num):
                return None

            def check_permission(self, tool_call, state):
                return True

            def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                return None

            def check_done(self, state, session_log, context, step_num):
                return None

            def should_stop(self, state, step_num, new_entities):
                return False

            def should_compact(self, state, session_log, conversation, step_num):
                return False

            def build_briefing(self, state, session_log, context):
                return None

            def build_prompt(self, **kwargs):
                return None

            def on_loop_end(self, state, session_log, context, llm):
                return 0

            def on_event(self, payload):
                return None

        h = MinimalHook()
        assert isinstance(h, LoopHook)

    def test_protocol_has_six_methods(self):
        import inspect

        from looplet.loop import LoopHook
        members = {
            name for name, _ in inspect.getmembers(LoopHook, predicate=callable)
            if not name.startswith("_")
        }
        required = {"pre_prompt", "pre_dispatch", "post_dispatch", "check_done", "should_stop", "on_loop_end"}
        assert required.issubset(members)

    def test_no_domain_specific_imports(self):
        import inspect

        import looplet.loop as m
        assert "primal_security" not in inspect.getsource(m)


# ══════════════════════════════════════════════════════════════════
# composable_loop integration tests
# ══════════════════════════════════════════════════════════════════


class TestComposableLoopBasic:
    def test_tools_required(self):
        from looplet.loop import composable_loop
        state = SimpleState()
        from tests.conftest import MockLLMBackend
        gen = composable_loop(MockLLMBackend(), state=state)
        with pytest.raises((ValueError, TypeError)):
            next(gen)

    def test_basic_loop_yields_step_and_terminates(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.types import Step

        state = SimpleState()
        # LLM immediately returns done()
        llm = _make_done_llm([], done_summary="finished")
        reg = _make_registry_with_done()

        steps = list(composable_loop(llm, state=state, tools=reg, config=LoopConfig(max_steps=5)))
        assert len(steps) >= 1
        assert all(isinstance(s, Step) for s in steps)
        # last step should be the done tool
        assert steps[-1].tool_call.tool == "done"

    def test_loop_after_two_tool_steps(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec

        state = SimpleState()
        search_json = '{"tool": "search", "args": {"q": "test"}}'
        llm = _make_done_llm([search_json], done_summary="done after search")

        def _search(**kwargs):
            return {"results": ["r1"]}

        reg = _make_registry_with_done(
            ToolSpec(
                name="search",
                description="Search",
                parameters={"q": "Search query."},
                execute=_search,
            )
        )

        steps = list(composable_loop(llm, state=state, tools=reg, config=LoopConfig(max_steps=10)))
        tools_called = [s.tool_call.tool for s in steps]
        assert "search" in tools_called
        assert "done" in tools_called

    def test_max_steps_enforced(self):
        from looplet.loop import LoopConfig, composable_loop

        state = SimpleState(max_steps=2)
        # LLM never calls done — just calls search forever
        from tests.conftest import MockLLMBackend
        llm = MockLLMBackend(['{"tool": "search", "args": {}}'] * 10)

        from looplet.tools import ToolSpec
        def _search(**kwargs):
            return {}

        reg = _make_registry_with_done(
            ToolSpec(
                name="search",
                description="search",
                parameters={},
                execute=_search,
            )
        )

        steps = list(composable_loop(llm, state=state, tools=reg, config=LoopConfig(max_steps=2)))
        # Should stop when budget runs out
        assert len(steps) <= 4  # budget_remaining=0 breaks loop

    def test_parse_error_yields_parse_error_step(self):
        from looplet.loop import LoopConfig, composable_loop

        state = SimpleState(max_steps=3)
        from tests.conftest import MockLLMBackend
        # First response: bad JSON; second: done (for recovery attempt too)
        done_json = '{"tool": "done", "args": {}}'
        llm = MockLLMBackend(["not json at all", "not json either", done_json])
        reg = _make_registry_with_done()

        steps = list(composable_loop(llm, state=state, tools=reg, config=LoopConfig(max_steps=5)))
        tool_names = [s.tool_call.tool for s in steps]
        assert "__parse_error__" in tool_names

    def test_return_value_is_trace(self):
        from looplet.loop import LoopConfig, composable_loop

        state = SimpleState()
        llm = _make_done_llm([])
        reg = _make_registry_with_done()

        gen = composable_loop(llm, state=state, tools=reg, config=LoopConfig(max_steps=5))
        steps = []
        trace = None
        try:
            while True:
                steps.append(next(gen))
        except StopIteration as e:
            trace = e.value

        assert trace is not None
        assert isinstance(trace, dict)
        assert "llm_calls" in trace


class TestComposableLoopHooks:
    def _make_hook_spy(self):
        """Return a hook that records call order."""
        events = []

        class SpyHook:
            def pre_prompt(self, state, session_log, context, step_num):
                events.append(("pre_prompt", step_num))
                return None

            def pre_dispatch(self, state, session_log, tool_call, step_num):
                events.append(("pre_dispatch", tool_call.tool))
                return None

            def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                events.append(("post_dispatch", tool_call.tool))
                return None

            def check_done(self, state, session_log, context, step_num):
                events.append(("check_done", step_num))
                return None

            def should_stop(self, state, step_num, new_entities):
                events.append(("should_stop", step_num))
                return False

            def on_loop_end(self, state, session_log, context, llm):
                events.append("on_loop_end")
                return 0

        return SpyHook(), events

    def test_hook_firing_order(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec

        state = SimpleState()
        search_json = '{"tool": "search", "args": {}}'
        done_json = '{"tool": "done", "args": {}}'
        from tests.conftest import MockLLMBackend
        llm = MockLLMBackend([search_json, done_json])

        def _search(**kwargs):
            return {"r": 1}

        reg = _make_registry_with_done(
            ToolSpec(
                name="search", description="s",
                parameters={},
                execute=_search,
            )
        )
        hook, events = self._make_hook_spy()

        list(composable_loop(
            llm, state=state, tools=reg, hooks=[hook],
            config=LoopConfig(max_steps=5),
        ))

        # pre_prompt fires before dispatch
        pre_prompt_idx = next(i for i, e in enumerate(events) if e[0] == "pre_prompt")
        post_dispatch_idx = next(i for i, e in enumerate(events) if e[0] == "post_dispatch")
        assert pre_prompt_idx < post_dispatch_idx

        # on_loop_end fires last
        assert events[-1] == "on_loop_end"

    def test_quality_gate_rejection(self):
        from looplet.loop import LoopConfig, composable_loop

        state = SimpleState()
        done_json = '{"tool": "done", "args": {}}'
        real_done_json = '{"tool": "done", "args": {}}'
        from tests.conftest import MockLLMBackend
        llm = MockLLMBackend([done_json, real_done_json])
        reg = _make_registry_with_done()

        reject_count = [0]

        class RejectOnce:
            def check_done(self, state, session_log, context, step_num):
                if reject_count[0] < 1:
                    reject_count[0] += 1
                    return "not enough evidence yet"
                return None

        steps = list(composable_loop(
            llm, state=state, tools=reg, hooks=[RejectOnce()],
            config=LoopConfig(max_steps=5),
        ))
        # Should see a rejected done step and then a real done step
        rejected = [s for s in steps if s.tool_call.tool == "done" and
                    isinstance(s.tool_result.data, dict) and s.tool_result.data.get("rejected")]
        assert len(rejected) >= 1

    def test_pre_dispatch_interception(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec
        from looplet.types import ToolResult

        state = SimpleState()
        from tests.conftest import MockLLMBackend
        llm = MockLLMBackend(['{"tool": "search", "args": {}}',
                               '{"tool": "done", "args": {}}'])

        dispatched = []

        def _search(**kwargs):
            dispatched.append("real_search")
            return {}

        reg = _make_registry_with_done(
            ToolSpec(
                name="search", description="s",
                parameters={},
                execute=_search,
            )
        )

        class InterceptHook:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "search":
                    return ToolResult(tool="search", args_summary="intercepted",
                                      data={"cached": True})
                return None

        list(composable_loop(
            llm, state=state, tools=reg, hooks=[InterceptHook()],
            config=LoopConfig(max_steps=5),
        ))
        # Real search should NOT have been called — intercepted
        assert "real_search" not in dispatched

    def test_multi_tool_in_one_response(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec

        state = SimpleState()
        multi_json = '{"tools": [{"tool": "search", "args": {}}, {"tool": "think", "args": {}}]}'
        done_json = '{"tool": "done", "args": {}}'
        from tests.conftest import MockLLMBackend
        llm = MockLLMBackend([multi_json, done_json])

        def _search(**kwargs):
            return {"r": 1}
        def _think(**kwargs):
            return {"thought": "ok"}

        reg = _make_registry_with_done(
            ToolSpec(name="search", description="s",
                     parameters={}, execute=_search),
            ToolSpec(name="think", description="t", free=True,
                     parameters={}, execute=_think),
        )

        steps = list(composable_loop(
            llm, state=state, tools=reg,
            config=LoopConfig(max_steps=10),
        ))
        tool_names = [s.tool_call.tool for s in steps]
        assert "search" in tool_names
        assert "think" in tool_names

    def test_should_stop_hook(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec

        state = SimpleState(max_steps=20)
        from tests.conftest import MockLLMBackend
        llm = MockLLMBackend(['{"tool": "search", "args": {}}'] * 20)

        def _search(**kwargs):
            return {}

        reg = _make_registry_with_done(
            ToolSpec(name="search", description="s",
                     parameters={}, execute=_search)
        )

        class StopAfterOneHook:
            def should_stop(self, state, step_num, new_entities):
                return step_num >= 1

        steps = list(composable_loop(
            llm, state=state, tools=reg, hooks=[StopAfterOneHook()],
            config=LoopConfig(max_steps=20),
        ))
        # Should stop very early
        assert len(steps) <= 3
