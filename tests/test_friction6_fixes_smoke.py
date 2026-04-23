"""Round-6 friction fixes: CostTracker + post_dispatch on done()."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.router import CostTracker
from looplet.telemetry import MetricsCollector, MetricsHook, Tracer, TracingHook
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


def _run(hook):
    responses = [
        '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
    ]
    return list(
        composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=_tools(),
            state=DefaultState(max_steps=5),
            hooks=[hook],
            config=LoopConfig(max_steps=5),
        )
    )


class _NullBackend:
    def generate(self, prompt, **kwargs):
        return ""

    def generate_with_tools(self, prompt, tools, **kwargs):
        return ""


class TestCostTrackerSignature:
    def test_positional_backend(self):
        tracker = CostTracker(_NullBackend(), cost_per_1k_input=0.003, cost_per_1k_output=0.015)
        assert tracker is not None

    def test_keyword_backend_still_works(self):
        tracker = CostTracker(backend=_NullBackend(), cost_per_1k_input=0.0, cost_per_1k_output=0.0)
        assert tracker is not None

    def test_missing_backend_raises(self):
        with pytest.raises(TypeError, match="backend"):
            CostTracker()  # type: ignore[call-arg]


class TestPostDispatchOnDone:
    def test_metrics_hook_counts_done_step(self):
        collector = MetricsCollector()
        steps = _run(MetricsHook(collector))
        assert len(steps) == 2
        assert collector.total_steps == 2
        assert collector.tool_call_histogram.get("done") == 1
        assert collector.tool_call_histogram.get("add") == 1

    def test_tracing_hook_captures_done_span(self):
        tracer = Tracer()
        _run(TracingHook(tracer))
        assert len(tracer.root_spans) == 1
        assert len(tracer.root_spans[0].children) == 2
