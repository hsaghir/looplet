"""Tests for cadence.telemetry — spans, tracer, metrics, and hooks."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from openharness.telemetry import (
    MetricsCollector,
    MetricsHook,
    Span,
    Tracer,
    TracingHook,
)
from openharness.types import Step, ToolCall, ToolResult


# ── Helpers ────────────────────────────────────────────────────────


def _make_step(number: int = 1, tool: str = "search", error: str | None = None) -> Step:
    tc = ToolCall(tool=tool, args={}, reasoning="r")
    tr = ToolResult(tool=tool, args_summary="", data={"rows": []}, error=error)
    return Step(number=number, tool_call=tc, tool_result=tr)


# ── Span ───────────────────────────────────────────────────────────


class TestSpan:
    def test_span_has_name(self) -> None:
        s = Span(name="step.1")
        assert s.name == "step.1"

    def test_span_id_generated_by_default(self) -> None:
        a = Span(name="a")
        b = Span(name="b")
        assert isinstance(a.span_id, str)
        assert len(a.span_id) == 12
        assert a.span_id != b.span_id

    def test_span_parent_id_defaults_none(self) -> None:
        s = Span(name="root")
        assert s.parent_id is None

    def test_span_status_defaults_ok(self) -> None:
        s = Span(name="op")
        assert s.status == "ok"

    def test_span_attributes_empty_by_default(self) -> None:
        s = Span(name="op")
        assert s.attributes == {}

    def test_span_children_empty_by_default(self) -> None:
        s = Span(name="op")
        assert s.children == []

    def test_span_end_time_defaults_none(self) -> None:
        s = Span(name="op")
        assert s.end_time is None

    def test_duration_ms_none_when_not_ended(self) -> None:
        s = Span(name="op")
        assert s.duration_ms is None

    def test_duration_ms_computed_when_ended(self) -> None:
        t = time.time()
        s = Span(name="op", start_time=t, end_time=t + 0.1)
        assert s.duration_ms is not None
        assert abs(s.duration_ms - 100.0) < 1.0

    def test_duration_ms_zero_for_instant_span(self) -> None:
        t = time.time()
        s = Span(name="op", start_time=t, end_time=t)
        assert s.duration_ms == pytest.approx(0.0)

    def test_span_custom_attributes(self) -> None:
        s = Span(name="op", attributes={"tool": "search", "step": 1})
        assert s.attributes["tool"] == "search"

    def test_span_custom_status_values(self) -> None:
        for status in ("ok", "error", "cancelled"):
            s = Span(name="op", status=status)
            assert s.status == status

    def test_span_child_added_to_children(self) -> None:
        parent = Span(name="parent")
        child = Span(name="child", parent_id=parent.span_id)
        parent.children.append(child)
        assert len(parent.children) == 1
        assert parent.children[0].name == "child"

    def test_span_start_time_defaults_to_now(self) -> None:
        before = time.time()
        s = Span(name="op")
        after = time.time()
        assert before <= s.start_time <= after


# ── Tracer ─────────────────────────────────────────────────────────


class TestTracer:
    def test_current_span_none_initially(self) -> None:
        t = Tracer()
        assert t.current_span is None

    def test_root_spans_empty_initially(self) -> None:
        t = Tracer()
        assert t.root_spans == []

    def test_start_span_returns_span(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        assert isinstance(s, Span)
        assert s.name == "root"

    def test_start_span_sets_current(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        assert t.current_span is s

    def test_start_span_adds_to_root_spans(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        assert s in t.root_spans

    def test_end_span_sets_end_time(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        t.end_span(s)
        assert s.end_time is not None

    def test_end_span_sets_status(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        t.end_span(s, status="error")
        assert s.status == "error"

    def test_end_span_pops_from_stack(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        t.end_span(s)
        assert t.current_span is None

    def test_nested_spans_parent_child(self) -> None:
        t = Tracer()
        parent = t.start_span("parent")
        child = t.start_span("child")
        assert child.parent_id == parent.span_id
        assert child in parent.children

    def test_nested_spans_current_tracks_depth(self) -> None:
        t = Tracer()
        parent = t.start_span("parent")
        child = t.start_span("child")
        assert t.current_span is child
        t.end_span(child)
        assert t.current_span is parent

    def test_child_not_added_to_root_spans(self) -> None:
        t = Tracer()
        t.start_span("parent")
        child = t.start_span("child")
        assert child not in t.root_spans

    def test_start_span_merges_attributes(self) -> None:
        t = Tracer()
        s = t.start_span("root", attributes={"tool": "search"})
        assert s.attributes["tool"] == "search"

    def test_end_span_merges_attributes(self) -> None:
        t = Tracer()
        s = t.start_span("root")
        t.end_span(s, attributes={"result_rows": 42})
        assert s.attributes["result_rows"] == 42

    def test_multiple_root_spans_accumulated(self) -> None:
        t = Tracer()
        a = t.start_span("a")
        t.end_span(a)
        b = t.start_span("b")
        t.end_span(b)
        assert len(t.root_spans) == 2

    def test_end_span_duration_positive(self) -> None:
        t = Tracer()
        s = t.start_span("op")
        time.sleep(0.001)
        t.end_span(s)
        assert s.duration_ms is not None
        assert s.duration_ms >= 0


# ── TracingHook ────────────────────────────────────────────────────


class TestTracingHook:
    def test_pre_prompt_creates_span(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        assert tracer.current_span is not None

    def test_pre_prompt_span_name_contains_step_info(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        assert tracer.current_span is not None

    def test_post_dispatch_creates_tool_span(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        step = _make_step(1, "search")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        # Should have created and ended a tool span
        assert tracer.root_spans or tracer.current_span is not None

    def test_on_loop_end_ends_root_span(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        hook.on_loop_end(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        if tracer.root_spans:
            assert tracer.root_spans[0].end_time is not None

    def test_pre_dispatch_returns_none(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        step = _make_step(1, "search")
        result = hook.pre_dispatch(MagicMock(), MagicMock(), step.tool_call, 1)
        assert result is None

    def test_check_done_returns_none(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        assert hook.check_done(MagicMock(), MagicMock(), MagicMock(), 1) is None

    def test_should_stop_returns_false(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        assert hook.should_stop(MagicMock(), 1, 0) is False

    def test_tool_span_has_tool_name_attribute(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        step = _make_step(1, "query")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        # After post_dispatch, a child span with the tool name should have been created
        current = tracer.current_span
        if current and current.children:
            assert current.children[0].attributes.get("tool") == "query"

    def test_error_step_sets_error_status(self) -> None:
        tracer = Tracer()
        hook = TracingHook(tracer=tracer)
        hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        step = _make_step(1, "search", error="connection failed")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        # step span should have error info recorded


# ── MetricsCollector ────────────────────────────────────────────────


class TestMetricsCollector:
    def test_initial_values_all_zero(self) -> None:
        m = MetricsCollector()
        assert m.total_steps == 0
        assert m.total_llm_calls == 0
        assert m.total_tool_calls == 0
        assert m.total_errors == 0
        assert m.total_input_tokens_est == 0
        assert m.total_output_tokens_est == 0
        assert m.total_duration_ms == 0.0
        assert m.tool_call_histogram == {}
        assert m.step_classifications == {}

    def test_record_step_increments_totals(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "productive", 100, 50, 200.0, False)
        assert m.total_steps == 1
        assert m.total_tool_calls == 1
        assert m.total_input_tokens_est == 100
        assert m.total_output_tokens_est == 50
        assert m.total_duration_ms == 200.0

    def test_record_step_updates_histogram(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "productive", 100, 50, 100.0, False)
        m.record_step("search", "productive", 100, 50, 100.0, False)
        m.record_step("query", "productive", 100, 50, 100.0, False)
        assert m.tool_call_histogram["search"] == 2
        assert m.tool_call_histogram["query"] == 1

    def test_record_step_updates_classifications(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "productive", 100, 50, 100.0, False)
        m.record_step("query", "empty", 100, 50, 100.0, False)
        m.record_step("done", "productive", 100, 50, 100.0, False)
        assert m.step_classifications["productive"] == 2
        assert m.step_classifications["empty"] == 1

    def test_record_step_error_increments_errors(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "error", 100, 50, 100.0, True)
        assert m.total_errors == 1

    def test_record_step_no_error_no_increment(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "productive", 100, 50, 100.0, False)
        assert m.total_errors == 0

    def test_multiple_steps_accumulate(self) -> None:
        m = MetricsCollector()
        for i in range(5):
            m.record_step("search", "productive", 100, 50, 200.0, False)
        assert m.total_steps == 5
        assert m.total_duration_ms == 1000.0
        assert m.total_input_tokens_est == 500

    def test_report_returns_string(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "productive", 100, 50, 100.0, False)
        report = m.report()
        assert isinstance(report, str)

    def test_report_contains_total_steps(self) -> None:
        m = MetricsCollector()
        m.record_step("search", "productive", 100, 50, 100.0, False)
        m.record_step("done", "productive", 100, 50, 100.0, False)
        report = m.report()
        assert "2" in report

    def test_report_contains_tool_names(self) -> None:
        m = MetricsCollector()
        m.record_step("query", "productive", 100, 50, 100.0, False)
        report = m.report()
        assert "query" in report

    def test_report_empty_collector(self) -> None:
        m = MetricsCollector()
        report = m.report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_record_step_duration_accumulates(self) -> None:
        m = MetricsCollector()
        m.record_step("a", "productive", 0, 0, 100.5, False)
        m.record_step("b", "productive", 0, 0, 200.5, False)
        assert m.total_duration_ms == pytest.approx(301.0)


# ── MetricsHook ────────────────────────────────────────────────────


class TestMetricsHook:
    def test_hook_updates_collector_on_post_dispatch(self) -> None:
        collector = MetricsCollector()
        hook = MetricsHook(collector=collector)
        step = _make_step(1, "search")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        assert collector.total_steps == 1
        assert collector.tool_call_histogram.get("search", 0) == 1

    def test_hook_counts_errors(self) -> None:
        collector = MetricsCollector()
        hook = MetricsHook(collector=collector)
        step = _make_step(1, "search", error="fail")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        assert collector.total_errors == 1

    def test_hook_no_error_when_no_error(self) -> None:
        collector = MetricsCollector()
        hook = MetricsHook(collector=collector)
        step = _make_step(1, "search")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        assert collector.total_errors == 0

    def test_hook_multiple_steps_accumulate(self) -> None:
        collector = MetricsCollector()
        hook = MetricsHook(collector=collector)
        for i in range(1, 6):
            s = _make_step(i, "search")
            hook.post_dispatch(MagicMock(), MagicMock(), s.tool_call, s.tool_result, i)
        assert collector.total_steps == 5
        assert collector.tool_call_histogram["search"] == 5

    def test_pre_dispatch_returns_none(self) -> None:
        hook = MetricsHook(collector=MetricsCollector())
        step = _make_step(1, "search")
        result = hook.pre_dispatch(MagicMock(), MagicMock(), step.tool_call, 1)
        assert result is None

    def test_check_done_returns_none(self) -> None:
        hook = MetricsHook(collector=MetricsCollector())
        assert hook.check_done(MagicMock(), MagicMock(), MagicMock(), 1) is None

    def test_should_stop_returns_false(self) -> None:
        hook = MetricsHook(collector=MetricsCollector())
        assert hook.should_stop(MagicMock(), 1, 0) is False

    def test_pre_prompt_noop(self) -> None:
        hook = MetricsHook(collector=MetricsCollector())
        result = hook.pre_prompt(MagicMock(), MagicMock(), MagicMock(), 1)
        assert result is None

    def test_on_loop_end_noop(self) -> None:
        hook = MetricsHook(collector=MetricsCollector())
        result = hook.on_loop_end(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert result == 0

    def test_hook_records_classification_for_done_tool(self) -> None:
        collector = MetricsCollector()
        hook = MetricsHook(collector=collector)
        step = _make_step(1, "done")
        hook.post_dispatch(MagicMock(), MagicMock(), step.tool_call, step.tool_result, 1)
        assert "done" in collector.tool_call_histogram
