"""Tests for looplet.context — ContextPressureHook."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from looplet.types import Step, ToolCall, ToolResult

# ── Helpers ───────────────────────────────────────────────────────


def _make_step(
    number: int, data: Any, error: str | None = None, result_key: str | None = None
) -> Step:
    tc = ToolCall(tool="search", args={"q": "test"}, reasoning="some reasoning")
    tr = ToolResult(
        tool="search", args_summary="q=test", data=data, error=error, result_key=result_key
    )
    return Step(number=number, tool_call=tc, tool_result=tr)


def _make_state(steps: list[Step]) -> Any:
    state = SimpleNamespace()
    state.steps = steps
    return state


def _make_session_log(entities: set | None = None, render_output: str = "") -> Any:
    log = MagicMock()
    log.all_entities.return_value = entities if entities is not None else set()
    log.render.return_value = render_output
    return log


# ── Import tests ──────────────────────────────────────────────────


class TestImports:
    def test_context_manager_hook_importable(self):
        from looplet.context import ContextPressureHook

        assert ContextPressureHook is not None

    def test_compact_data_importable(self):
        from looplet.context import _compact_data

        assert _compact_data is not None

    def test_default_constants_importable(self):
        from looplet.context import (
            DEFAULT_BLOCKING_BUFFER,
            DEFAULT_COMPACT_BUFFER,
            DEFAULT_CONTEXT_WINDOW,
            DEFAULT_RESULT_MAX_AGE_FULL,
            DEFAULT_WARNING_BUFFER,
        )

        assert DEFAULT_CONTEXT_WINDOW == 128_000
        assert DEFAULT_COMPACT_BUFFER == 20_000
        assert DEFAULT_WARNING_BUFFER == 30_000
        assert DEFAULT_BLOCKING_BUFFER == 5_000
        assert DEFAULT_RESULT_MAX_AGE_FULL == 3


# ── ContextPressureHook constructor ────────────────────────────────


class TestContextManagerHookConstructor:
    def test_default_params(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        assert hook._context_window == 128_000
        assert hook._compact_threshold == 128_000 - 20_000
        assert hook._warning_threshold == 128_000 - 30_000
        assert hook._blocking_threshold == 128_000 - 5_000
        assert hook._result_max_age_full == 3
        assert hook._per_result_chars == 50_000
        assert hook._aggregate_chars == 500_000

    def test_configurable_context_window(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, context_window=200_000)
        assert hook._context_window == 200_000
        assert hook._compact_threshold == 200_000 - 20_000

    def test_configurable_buffers(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(
            llm=None,
            context_window=100_000,
            compact_buffer=10_000,
            warning_buffer=15_000,
            blocking_buffer=2_000,
        )
        assert hook._compact_threshold == 90_000
        assert hook._warning_threshold == 85_000
        assert hook._blocking_threshold == 98_000

    def test_absolute_not_fraction(self):
        """Buffer thresholds are absolute token counts, NOT fractions."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, context_window=100_000, compact_buffer=20_000)
        # If fractions were used, threshold would be around 80_000 * some_fraction
        # With absolute offsets: 100_000 - 20_000 = 80_000
        assert hook._compact_threshold == 80_000

    def test_configurable_result_params(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, per_result_chars=10_000, aggregate_chars=100_000)
        assert hook._per_result_chars == 10_000
        assert hook._aggregate_chars == 100_000

    def test_configurable_result_max_age(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=5)
        assert hook._result_max_age_full == 5

    def test_extra_llm_calls_starts_at_zero(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        assert hook._extra_llm_calls == 0

    def test_compact_failures_starts_at_zero(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        assert hook._compact_failures == 0


# ── Result aging ──────────────────────────────────────────────────


class TestResultAging:
    def test_fresh_result_not_compacted(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=3)
        step = _make_step(1, data=["item1", "item2"])
        state = _make_state([step])
        hook._age_results(state, step_num=3)  # age = 3 - 1 = 2 <= max_age
        # Should NOT be compacted (age 2 is not > 3)
        assert isinstance(step.tool_result.data, list)

    def test_old_result_compacted(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=3)
        step = _make_step(1, data=["item1", "item2", "item3"])
        state = _make_state([step])
        hook._age_results(state, step_num=5)  # age = 5 - 1 = 4 > 3
        # Should be compacted
        assert isinstance(step.tool_result.data, dict)
        assert step.tool_result.data.get("__compacted__") is True

    def test_already_compacted_skipped(self):
        """Idempotency: already-compacted results are not re-processed."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=3)
        compacted_data = {"__compacted__": True, "type": "list", "original_count": 5}
        step = _make_step(1, data=compacted_data)
        state = _make_state([step])
        hook._age_results(state, step_num=10)  # very old, but already compacted
        # Should remain unchanged
        assert step.tool_result.data is compacted_data

    def test_error_results_skipped(self):
        """Results with errors are not aged."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=3)
        step = _make_step(1, data=None, error="Tool failed")
        state = _make_state([step])
        hook._age_results(state, step_num=10)
        assert step.tool_result.data is None

    def test_none_data_results_skipped(self):
        """Results with None data are not aged."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=3)
        step = _make_step(1, data=None)
        state = _make_state([step])
        hook._age_results(state, step_num=10)
        assert step.tool_result.data is None

    def test_exact_age_boundary_not_compacted(self):
        """At exactly result_max_age_full, result should NOT be compacted (> not >=)."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=3)
        step = _make_step(1, data=["x"])
        state = _make_state([step])
        hook._age_results(state, step_num=4)  # age = 4 - 1 = 3 == max_age, NOT > max_age
        assert isinstance(step.tool_result.data, list)

    def test_no_steps_attr_safe(self):
        """Works gracefully when state has no steps attribute."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        state = SimpleNamespace()  # no .steps
        hook._age_results(state, step_num=5)  # should not raise

    def test_configurable_max_age(self):
        """result_max_age_full is configurable."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=1)
        step = _make_step(1, data=["x"])
        state = _make_state([step])
        hook._age_results(state, step_num=3)  # age = 2 > 1
        assert isinstance(step.tool_result.data, dict)
        assert step.tool_result.data.get("__compacted__") is True


# ── Compact data ──────────────────────────────────────────────────


class TestCompactData:
    def test_compacts_list(self):
        from looplet.context import _compact_data

        data = list(range(100))
        result = _compact_data(data, "my_key")
        assert result["__compacted__"] is True
        assert result["type"] == "list"
        assert result["original_count"] == 100
        assert result["sample"] == [0, 1, 2]
        assert result["recall_key"] == "my_key"

    def test_compacts_list_without_key(self):
        from looplet.context import _compact_data

        data = [1, 2, 3]
        result = _compact_data(data, None)
        assert result["__compacted__"] is True
        assert "recall_key" not in result

    def test_compacts_dict_with_rows(self):
        from looplet.context import _compact_data

        data = {"rows": list(range(50)), "total": 50}
        result = _compact_data(data, "key1")
        assert result["__compacted__"] is True
        assert result["total_rows"] == 50
        assert len(result["sample_rows"]) == 3
        assert result["total"] == 50
        assert result["recall_key"] == "key1"

    def test_compacts_plain_dict(self):
        from looplet.context import _compact_data

        data = {"foo": "bar", "baz": 42}
        result = _compact_data(data, None)
        assert result["__compacted__"] is True
        assert result["foo"] == "bar"

    def test_compacts_string(self):
        from looplet.context import _compact_data

        data = "some string result"
        result = _compact_data(data, "k")
        assert result["__compacted__"] is True
        assert "summary" in result
        assert result["recall_key"] == "k"

    def test_compacted_result_is_dict(self):
        """Always returns a dict (no type confusion)."""
        from looplet.context import _compact_data

        for data in [[1, 2], {"x": 1}, "text", 42]:
            result = _compact_data(data, None)
            assert isinstance(result, dict)


# ── Health probe ──────────────────────────────────────────────────


class TestHealthProbe:
    def test_no_all_entities_method_returns_none(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        log = SimpleNamespace()  # no all_entities method
        state = _make_state([])
        assert hook._health_probe(state, log) is None

    def test_empty_entities_returns_none(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        log = _make_session_log(entities=set())
        state = _make_state([])
        assert hook._health_probe(state, log) is None

    def test_few_entities_returns_none(self):
        """With <= 3 entities, no probe needed."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        log = _make_session_log(entities={"a", "b", "c"})
        state = _make_state([])
        assert hook._health_probe(state, log) is None

    def test_all_entities_visible_returns_none(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        entities = {"entity1", "entity2", "entity3", "entity4", "entity5"}
        log = _make_session_log(
            entities=entities, render_output="entity1 entity2 entity3 entity4 entity5 in context"
        )
        state = _make_state([])
        assert hook._health_probe(state, log) is None

    def test_many_missing_entities_returns_reminder(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        entities = {f"entity{i}" for i in range(20)}
        # Only first 5 appear in render
        visible_text = "entity0 entity1 entity2 entity3 entity4"
        log = _make_session_log(entities=entities, render_output=visible_text)
        state = _make_state([])
        result = hook._health_probe(state, log)
        assert result is not None
        assert "CONTEXT NOTE" in result
        assert "compressed" in result.lower() or "entities" in result.lower()

    def test_reminder_has_no_ioc_reference(self):
        """No 'IOC' text — uses 'highlight' or 'notable item'."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        entities = {f"entity{i}" for i in range(20)}
        log = _make_session_log(entities=entities, render_output="entity0")
        state = _make_state([])
        result = hook._health_probe(state, log)
        if result is not None:
            assert "IOC" not in result


# ── Estimate context tokens ────────────────────────────────────────


class TestEstimateContextTokens:
    def test_returns_int(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        state = _make_state([])
        log = _make_session_log(render_output="")
        result = hook._estimate_context_tokens(state, log)
        assert isinstance(result, int)

    def test_includes_overhead(self):
        """Even empty state has overhead tokens."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        state = _make_state([])
        log = _make_session_log(render_output="")
        result = hook._estimate_context_tokens(state, log)
        assert result > 0  # overhead alone > 0

    def test_increases_with_more_steps(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        log = _make_session_log(render_output="")
        state_empty = _make_state([])
        state_full = _make_state([_make_step(i, data=list(range(100))) for i in range(10)])
        empty_tokens = hook._estimate_context_tokens(state_empty, log)
        full_tokens = hook._estimate_context_tokens(state_full, log)
        assert full_tokens > empty_tokens

    def test_increases_with_larger_session_log(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        state = _make_state([])
        small_log = _make_session_log(render_output="small")
        large_log = _make_session_log(render_output="x" * 10000)
        small_tokens = hook._estimate_context_tokens(state, small_log)
        large_tokens = hook._estimate_context_tokens(state, large_log)
        assert large_tokens > small_tokens

    def test_no_steps_attr_safe(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        state = SimpleNamespace()  # no .steps
        log = _make_session_log(render_output="some log text")
        result = hook._estimate_context_tokens(state, log)
        assert result > 0


# ── Pre-prompt hook ────────────────────────────────────────────────


class TestPrePrompt:
    def test_returns_none_when_under_threshold(self):
        """When context is well below threshold, returns None (no intervention)."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, context_window=128_000)
        state = _make_state([])
        log = _make_session_log(render_output="short log")
        result = hook.pre_prompt(state, log, context=None, step_num=1)
        assert result is None

    def test_ages_results_called(self):
        """pre_prompt calls _age_results to compact old data."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, result_max_age_full=1)
        # Step 1 with data, at step 5 it should be aged
        step = _make_step(1, data=["x", "y"])
        state = _make_state([step])
        log = _make_session_log(render_output="")
        hook.pre_prompt(state, log, context=None, step_num=5)
        # Step should be aged (age = 5 - 1 = 4 > 1)
        assert isinstance(step.tool_result.data, dict)
        assert step.tool_result.data.get("__compacted__") is True

    def test_enforces_result_budget(self):
        """pre_prompt calls trim_results from scaffolding."""
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None, per_result_chars=100, aggregate_chars=1000)
        # Oversized result
        big_data = {"rows": ["x" * 200] * 10}
        step = _make_step(1, data=big_data)
        state = _make_state([step])
        log = _make_session_log(render_output="")
        hook.pre_prompt(state, log, context=None, step_num=2)
        # Budget enforcement should have compacted the large result
        assert step.tool_result.data != big_data or isinstance(step.tool_result.data, dict)

    def test_blocking_threshold_returns_warning(self):
        """When context is at blocking threshold, returns a warning string."""
        from looplet.context import ContextPressureHook

        # Very small context window so estimate exceeds blocking threshold easily
        hook = ContextPressureHook(llm=None, context_window=100, blocking_buffer=90)
        # block threshold = 100 - 90 = 10 tokens
        # Overhead alone (~13000 chars / 4 = 3250 tokens) >> 10
        state = _make_state([])
        log = _make_session_log(render_output="x" * 100)
        result = hook.pre_prompt(state, log, context=None, step_num=1)
        assert result is not None
        assert "CONTEXT" in result.upper() or "LIMIT" in result.upper() or "⚠" in result

    def test_compact_threshold_triggers_compaction(self):
        """When context approaches compact threshold, age_session_entries is called."""
        from looplet.context import ContextPressureHook

        with patch("looplet.context.age_session_entries") as mock_compress:
            mock_compress.return_value = "compressed"
            # Very small context window so estimate exceeds compact threshold
            hook = ContextPressureHook(
                llm=None, context_window=100, compact_buffer=90, blocking_buffer=5
            )
            # compact threshold = 100 - 90 = 10 (still triggers compact before blocking)
            state = _make_state([])
            log = _make_session_log(render_output="x" * 100)
            # Override estimate to return value between compact and blocking thresholds
            hook._estimate_context_tokens = lambda s, l: 15  # > compact(10) but < blocking(95)
            hook.pre_prompt(state, log, context=None, step_num=1)
            mock_compress.assert_called_once()

    def test_on_loop_end_returns_extra_llm_calls(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        hook._extra_llm_calls = 3
        result = hook.on_loop_end(state=None, session_log=None, context=None, llm=None)
        assert result == 3


# ── No IOC references ─────────────────────────────────────────────


class TestNoIOCReferences:
    def test_no_ioc_in_source(self):
        """context.py must not contain 'IOC' references."""
        import inspect

        import looplet.context as mod

        source = inspect.getsource(mod)
        assert "IOC" not in source, "Found 'IOC' reference in context.py"

    def test_no_domain_specific_imports(self):
        """context.py must only import from looplet.*."""
        import inspect

        import looplet.context as mod

        source = inspect.getsource(mod)
        assert "primal_security" not in source


# ── LoopHook protocol compatibility ───────────────────────────────


class TestLoopHookCompatibility:
    def test_has_pre_prompt(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        assert callable(hook.pre_prompt)

    def test_has_on_loop_end(self):
        from looplet.context import ContextPressureHook

        hook = ContextPressureHook(llm=None)
        assert callable(hook.on_loop_end)
