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

    reg.register(
        ToolSpec(
            name="done",
            description="Signal task completion.",
            parameters={"summary": "Summary of what was accomplished."},
            execute=_done_fn,
        )
    )
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
            name
            for name, _ in inspect.getmembers(LoopHook, predicate=callable)
            if not name.startswith("_")
        }
        required = {
            "pre_prompt",
            "pre_dispatch",
            "post_dispatch",
            "check_done",
            "should_stop",
            "on_loop_end",
        }
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

    def test_concurrent_dispatch_uses_per_tool_progress_context(self):
        from looplet.events import LifecycleEvent
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec
        from looplet.types import ToolContext

        progress_tools = []

        class ProgressHook:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.TOOL_PROGRESS:
                    progress_tools.append(payload.tool_call.tool)

        def first(*, ctx: ToolContext):
            ctx.report_progress("working", {})
            return {"tool": "first"}

        def second(*, ctx: ToolContext):
            ctx.report_progress("working", {})
            return {"tool": "second"}

        llm = _make_done_llm(
            [
                '{"tools": ['
                '{"tool": "first", "args": {}, "reasoning": "r"},'
                '{"tool": "second", "args": {}, "reasoning": "r"}'
                "]}"
            ]
        )
        reg = _make_registry_with_done(
            ToolSpec(
                name="first",
                description="first",
                parameters={},
                execute=first,
                concurrent_safe=True,
            ),
            ToolSpec(
                name="second",
                description="second",
                parameters={},
                execute=second,
                concurrent_safe=True,
            ),
        )

        list(
            composable_loop(
                llm,
                state=SimpleState(max_steps=5),
                tools=reg,
                config=LoopConfig(max_steps=5, concurrent_dispatch=True),
                hooks=[ProgressHook()],
            )
        )

        assert sorted(progress_tools) == ["first", "second"]

    def test_regular_checkpoint_includes_recorded_session_log_entry(self, tmp_path):
        from looplet.checkpoint import FileCheckpointStore
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec
        from tests.conftest import MockLLMBackend

        reg = _make_registry_with_done(
            ToolSpec(
                name="ping",
                description="ping",
                parameters={},
                execute=lambda: {"pong": True},
            )
        )
        llm = MockLLMBackend(
            [
                '{"tool": "ping", "args": {}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "done"}, "reasoning": "r"}',
            ]
        )

        list(
            composable_loop(
                llm=llm,
                tools=reg,
                state=SimpleState(max_steps=3),
                config=LoopConfig(max_steps=3, checkpoint_dir=str(tmp_path)),
            )
        )

        checkpoint = FileCheckpointStore(tmp_path).load("step_1")

        assert checkpoint is not None
        assert len(checkpoint.session_log_data["entries"]) == 1

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
                name="search",
                description="s",
                parameters={},
                execute=_search,
            )
        )
        hook, events = self._make_hook_spy()

        list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[hook],
                config=LoopConfig(max_steps=5),
            )
        )

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

        steps = list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[RejectOnce()],
                config=LoopConfig(max_steps=5),
            )
        )
        # Should see a rejected done step and then a real done step
        rejected = [
            s
            for s in steps
            if s.tool_call.tool == "done"
            and isinstance(s.tool_result.data, dict)
            and s.tool_result.data.get("rejected")
        ]
        assert len(rejected) >= 1

    def test_pre_dispatch_interception(self):
        from looplet.loop import LoopConfig, composable_loop
        from looplet.tools import ToolSpec
        from looplet.types import ToolResult

        state = SimpleState()
        from tests.conftest import MockLLMBackend

        llm = MockLLMBackend(['{"tool": "search", "args": {}}', '{"tool": "done", "args": {}}'])

        dispatched = []

        def _search(**kwargs):
            dispatched.append("real_search")
            return {}

        reg = _make_registry_with_done(
            ToolSpec(
                name="search",
                description="s",
                parameters={},
                execute=_search,
            )
        )

        class InterceptHook:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "search":
                    return ToolResult(
                        tool="search", args_summary="intercepted", data={"cached": True}
                    )
                return None

        list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[InterceptHook()],
                config=LoopConfig(max_steps=5),
            )
        )
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
            ToolSpec(name="search", description="s", parameters={}, execute=_search),
            ToolSpec(name="think", description="t", free=True, parameters={}, execute=_think),
        )

        steps = list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                config=LoopConfig(max_steps=10),
            )
        )
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
            ToolSpec(name="search", description="s", parameters={}, execute=_search)
        )

        class StopAfterOneHook:
            def should_stop(self, state, step_num, new_entities):
                return step_num >= 1

        steps = list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[StopAfterOneHook()],
                config=LoopConfig(max_steps=20),
            )
        )
        # Should stop very early
        assert len(steps) <= 3


class TestExtractEntitiesSignatureDispatch:
    """Loop dispatches ``extract_entities`` based on its signature.

    Stateless ``(data) -> list[str]`` callables keep the legacy 1-arg
    contract. Stateful ``(data, state=None) -> list[str]`` callables
    receive the live state, so domain extractors can update
    ``state.*`` fields without becoming method-local closures.
    """

    def _run_loop(self, extractor, *, max_steps: int = 2):
        """Run a tiny loop with a scripted MockLLM and the given extractor."""
        import json

        from looplet import (
            DefaultState,
            LoopConfig,
            MockLLMBackend,
            composable_loop,
            register_done_tool,
        )
        from looplet.tools import BaseToolRegistry, ToolSpec

        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="echo",
                description="echo",
                parameters={"text": "string"},
                execute=lambda **kw: {"text": kw.get("text", ""), "tag": "ent-1"},
            )
        )
        register_done_tool(tools)

        responses = [
            json.dumps({"tool": "echo", "args": {"text": "hi"}, "reasoning": "x"}),
            json.dumps({"tool": "done", "args": {"summary": "ok"}, "reasoning": "x"}),
        ]
        llm = MockLLMBackend(responses=responses)
        state = DefaultState(max_steps=max_steps)
        config = LoopConfig(max_steps=max_steps, extract_entities=extractor)
        steps = list(
            composable_loop(
                llm=llm,
                task={"id": "t"},
                tools=tools,
                state=state,
                config=config,
            )
        )
        return steps, state

    def test_legacy_one_arg_extractor_keeps_working(self):
        """``extract_entities(data) -> list[str]`` — backward compat."""
        seen: list = []

        def stateless(data):
            seen.append(data)
            return ["A"]

        steps, _ = self._run_loop(stateless)
        assert seen, "extractor should have been called"
        assert all(isinstance(d, dict) for d in seen)

    def test_two_arg_extractor_receives_state(self):
        """``extract_entities(data, state=None)`` — gets live state."""
        observed = {"states": []}

        def stateful(data, state=None):
            observed["states"].append(state)
            # Mutate state to prove identity.
            if state is not None and not hasattr(state, "_marker"):
                state._marker = "set"
            return ["B"]

        steps, state = self._run_loop(stateful)
        assert observed["states"], "extractor should have been called"
        assert all(s is state for s in observed["states"])
        assert getattr(state, "_marker", None) == "set"

    def test_kwargs_extractor_receives_state(self):
        """``extract_entities(data, **kw)`` — sig-detect picks it up too."""
        observed = {"states": []}

        def varkw(data, **kw):
            observed["states"].append(kw.get("state"))
            return ["C"]

        steps, state = self._run_loop(varkw)
        assert observed["states"]
        assert all(s is state for s in observed["states"])


class TestPreLoopToolsKwarg:
    """Hooks that declare ``tools=`` get the live registry at pre_loop."""

    def _run(self, hook, *, max_steps: int = 2):
        import json

        from looplet import (
            DefaultState,
            LoopConfig,
            MockLLMBackend,
            composable_loop,
            register_done_tool,
        )
        from looplet.tools import BaseToolRegistry, ToolSpec

        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="echo",
                description="echo",
                parameters={"text": "string"},
                execute=lambda **kw: {"text": kw.get("text", "")},
            )
        )
        register_done_tool(tools)
        responses = [
            json.dumps({"tool": "echo", "args": {"text": "hi"}, "reasoning": "x"}),
            json.dumps({"tool": "done", "args": {"summary": "ok"}, "reasoning": "x"}),
        ]
        llm = MockLLMBackend(responses=responses)
        state = DefaultState(max_steps=max_steps)
        list(
            composable_loop(
                llm=llm,
                task={"id": "t"},
                tools=tools,
                hooks=[hook],
                state=state,
                config=LoopConfig(max_steps=max_steps),
            )
        )
        return tools

    def test_legacy_three_arg_pre_loop_keeps_working(self):
        observed = {"called": 0}

        class Legacy:
            def pre_loop(self, state, session_log, context):
                observed["called"] += 1

        self._run(Legacy())
        assert observed["called"] == 1

    def test_pre_loop_with_tools_kwarg_receives_registry(self):
        observed = {"received": None}

        class TakesTools:
            def pre_loop(self, state, session_log, context, tools=None):
                observed["received"] = tools

        tools = self._run(TakesTools())
        assert observed["received"] is tools

    def test_pre_loop_can_register_derived_tool(self):
        """A pre_loop hook can register tools the loop will then dispatch."""
        from looplet.tools import ToolSpec

        class Registrar:
            def pre_loop(self, state, session_log, context, tools=None):
                tools.register(
                    ToolSpec(
                        name="derived",
                        description="derived at load time",
                        parameters={},
                        execute=lambda **kw: {"hello": "world"},
                    )
                )

        tools = self._run(Registrar())
        assert "derived" in {s.name for s in tools._tools.values()}

    def test_kwargs_pre_loop_receives_tools(self):
        observed = {"got_tools": False}

        class VarKw:
            def pre_loop(self, state, session_log, context, **kw):
                observed["got_tools"] = "tools" in kw

        self._run(VarKw())
        assert observed["got_tools"] is True


# ══════════════════════════════════════════════════════════════════
# LoopContext / hook bind() — closures-over-state are first class
# ══════════════════════════════════════════════════════════════════


class TestLoopContextBind:
    """Hooks that define ``bind(ctx)`` receive a mutable handle to live
    loop state. Replaces the closures-over-state anti-pattern that
    used to break workspace round-trip via ``to_config()``."""

    def test_bind_called_once_with_loop_context(self):
        from looplet.loop import LoopConfig, LoopContext, composable_loop

        captured: dict[str, object] = {}

        class CaptureHook:
            def bind(self, ctx):
                captured["ctx"] = ctx
                captured["call_count"] = captured.get("call_count", 0) + 1

        state = SimpleState()
        llm = _make_done_llm([], done_summary="ok")
        reg = _make_registry_with_done()
        list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[CaptureHook()],
                config=LoopConfig(max_steps=5),
            )
        )
        assert captured["call_count"] == 1
        ctx = captured["ctx"]
        assert isinstance(ctx, LoopContext)
        assert ctx.state is state
        assert ctx.tools is reg
        # config is the resolved LoopConfig instance
        assert ctx.config is not None

    def test_step_num_mutates_as_loop_progresses(self):
        from looplet.loop import LoopConfig, composable_loop

        observed_step_nums: list[int] = []

        class WatchHook:
            def bind(self, ctx):
                self.ctx = ctx

            def pre_prompt(self, state, session_log, context, step_num):
                # Reading from the bound ctx returns the live step_num,
                # not a stale snapshot from bind()-time.
                observed_step_nums.append(self.ctx.step_num)
                return None

        state = SimpleState()
        llm = _make_done_llm([], done_summary="ok")
        reg = _make_registry_with_done()
        list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[WatchHook()],
                config=LoopConfig(max_steps=3),
            )
        )
        # At least one step ran; ctx.step_num was non-zero on entry.
        assert observed_step_nums
        assert observed_step_nums[0] >= 1

    def test_hook_without_bind_works_unchanged(self):
        """Backward compat: hooks lacking ``bind`` aren't broken."""
        from looplet.loop import LoopConfig, composable_loop

        called = []

        class LegacyHook:
            def pre_loop(self, state, session_log, context):
                called.append("pre_loop")

        state = SimpleState()
        llm = _make_done_llm([], done_summary="ok")
        reg = _make_registry_with_done()
        list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[LegacyHook()],
                config=LoopConfig(max_steps=3),
            )
        )
        assert called == ["pre_loop"]

    def test_loop_context_carries_tools_and_resources(self):
        """``ctx.tools`` is the live registry; ``ctx.resources`` is
        sourced from config when available."""
        from looplet.loop import LoopConfig, composable_loop

        seen: dict = {}

        class ToolsHook:
            def bind(self, ctx):
                seen["tools_is_registry"] = ctx.tools is not None
                seen["resources_is_dict"] = isinstance(ctx.resources, dict)

        state = SimpleState()
        llm = _make_done_llm([], done_summary="ok")
        reg = _make_registry_with_done()
        list(
            composable_loop(
                llm,
                state=state,
                tools=reg,
                hooks=[ToolsHook()],
                config=LoopConfig(max_steps=3),
            )
        )
        assert seen == {"tools_is_registry": True, "resources_is_dict": True}
