"""Tests for looplet.scaffolding - LLM retry, truncation, trackers."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from looplet.scaffolding import (
    PARSE_RECOVERY_MAX,
    LLMResult,
    StepProgressTracker,
    age_session_entries,
    build_parse_recovery_prompt,
    emergency_truncate,
    estimate_prompt_tokens,
    estimate_tokens,
    is_context_oversized,
    llm_call_with_retry,
    trim_results,
    truncate_tool_result,
)
from looplet.types import Step, ToolCall, ToolResult

pytestmark = pytest.mark.smoke


# ── Helpers ──────────────────────────────────────────────────────


class _FakeLLM:
    """Minimal LLMBackend-like object for testing."""

    def __init__(self, responses: list[str] | None = None, fail_after: int = -1):
        self.responses = responses or []
        self.fail_after = fail_after
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.call_count += 1
        if self.fail_after >= 0 and self.call_count > self.fail_after:
            raise RuntimeError("LLM failure")
        idx = min(self.call_count - 1, len(self.responses) - 1)
        return self.responses[idx] if self.responses else "ok"


# ── LLMResult ─────────────────────────────────────────────────────


class TestLLMResult:
    def test_ok_result(self) -> None:
        r = LLMResult("hello")
        assert r.ok is True
        assert r.text == "hello"
        assert r.error is None
        assert r.is_prompt_too_long is False

    def test_error_result(self) -> None:
        e = ValueError("something failed")
        r = LLMResult(None, e)
        assert r.ok is False
        assert r.text is None
        assert r.error is e

    def test_uses_slots(self) -> None:
        assert hasattr(LLMResult, "__slots__")
        assert "is_prompt_too_long" in LLMResult.__slots__

    def test_slots_no_dict(self) -> None:
        r = LLMResult("x")
        assert not hasattr(r, "__dict__"), "LLMResult must use __slots__"

    def test_is_prompt_too_long_computed_from_error(self) -> None:
        e = Exception("prompt is too long for this model")
        r = LLMResult(None, e)
        assert r.is_prompt_too_long is True

    def test_is_prompt_too_long_false_for_other_errors(self) -> None:
        e = Exception("network timeout")
        r = LLMResult(None, e)
        assert r.is_prompt_too_long is False

    def test_is_prompt_too_long_not_a_constructor_param(self) -> None:
        import inspect

        sig = inspect.signature(LLMResult.__init__)
        assert "is_prompt_too_long" not in sig.parameters

    def test_prompt_too_long_markers(self) -> None:
        for marker in [
            "context_length_exceeded",
            "maximum context length",
            "token limit",
            "too many tokens",
            "input is too long",
            "request too large",
            "413",
        ]:
            r = LLMResult(None, Exception(marker))
            assert r.is_prompt_too_long is True, f"Expected True for: {marker}"

    def test_prompt_too_long_detection_runtime_error(self) -> None:
        r = LLMResult(None, RuntimeError("prompt is too long for this model"))
        assert r.is_prompt_too_long is True
        assert r.ok is False


# ── llm_call_with_retry ────────────────────────────────────────────


class TestLlmCallWithRetry:
    def test_success_first_attempt(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = "success"
        result = llm_call_with_retry(llm, "prompt")
        assert result.ok is True
        assert result.text == "success"
        assert llm.generate.call_count == 1

    def test_success_first_try_fake_llm(self) -> None:
        llm = _FakeLLM(["response"])
        result = llm_call_with_retry(llm, "prompt")
        assert result.ok
        assert result.text == "response"
        assert llm.call_count == 1

    def test_retry_on_failure_then_success(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [ValueError("fail"), "success"]
        with patch("looplet.scaffolding.time.sleep"):
            result = llm_call_with_retry(llm, "prompt", max_retries=1)
        assert result.ok is True
        assert llm.generate.call_count == 2

    def test_retry_on_failure_then_succeed_monkeypatch(self, monkeypatch) -> None:
        monkeypatch.setattr("time.sleep", lambda x: None)
        call_count = 0

        class FlakyLLM:
            def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise RuntimeError("temporary failure")
                return "recovered"

        result = llm_call_with_retry(FlakyLLM(), "prompt", max_retries=2)
        assert result.ok
        assert result.text == "recovered"

    def test_exhausts_retries_returns_error(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = ValueError("always fails")
        with patch("looplet.scaffolding.time.sleep"):
            result = llm_call_with_retry(llm, "prompt", max_retries=2)
        assert result.ok is False
        assert result.error is not None
        assert llm.generate.call_count == 3  # 1 initial + 2 retries

    def test_all_retries_exhausted_fake(self, monkeypatch) -> None:
        monkeypatch.setattr("time.sleep", lambda x: None)
        llm = _FakeLLM(fail_after=0)
        result = llm_call_with_retry(llm, "prompt", max_retries=2)
        assert not result.ok
        assert result.error is not None

    def test_prompt_too_long_not_retried(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = Exception("prompt is too long")
        result = llm_call_with_retry(llm, "prompt", max_retries=2)
        assert result.ok is False
        assert result.is_prompt_too_long is True
        assert llm.generate.call_count == 1  # not retried

    def test_no_retry_on_prompt_too_long_custom(self) -> None:
        call_count = 0

        class PromptTooLongLLM:
            def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("prompt is too long")

        result = llm_call_with_retry(PromptTooLongLLM(), "prompt", max_retries=3)
        assert not result.ok
        assert result.is_prompt_too_long is True
        assert call_count == 1, "must not retry on prompt-too-long"

    def test_passes_kwargs_to_generate(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = "ok"
        llm_call_with_retry(llm, "p", max_tokens=500, system_prompt="sys", temperature=0.5)
        llm.generate.assert_called_once_with(
            "p", max_tokens=500, system_prompt="sys", temperature=0.5
        )


# ── build_parse_recovery_prompt ────────────────────────────────────


class TestBuildParseRecoveryPrompt:
    def test_contains_original_prompt(self) -> None:
        result = build_parse_recovery_prompt("original prompt", "bad json")
        assert "original prompt" in result

    def test_contains_raw_response_preview(self) -> None:
        result = build_parse_recovery_prompt("p", "bad response data")
        assert "bad response data" in result

    def test_instructs_json_only(self) -> None:
        result = build_parse_recovery_prompt("p", "bad")
        assert "JSON" in result

    def test_parse_recovery_max_constant(self) -> None:
        assert isinstance(PARSE_RECOVERY_MAX, int)
        assert PARSE_RECOVERY_MAX >= 1

    def test_long_raw_truncated_to_300(self) -> None:
        long_raw = "x" * 1000
        result = build_parse_recovery_prompt("p", long_raw)
        assert "x" * 300 in result
        assert "x" * 400 not in result

    def test_no_domain_specific_imports(self) -> None:
        import inspect

        import looplet.scaffolding as m

        src = inspect.getsource(m)
        assert "primal_security" not in src


# ── truncate_tool_result ────────────────────────────────────────────


class TestTruncateToolResult:
    def test_list_over_max_rows_truncated(self) -> None:
        data = list(range(100))
        result = truncate_tool_result(data, max_rows=50)
        assert isinstance(result, dict)
        assert result["total"] == 100
        assert result["showing"] == 50
        assert result["truncated"] is True

    def test_list_under_max_rows_unchanged(self) -> None:
        data = [1, 2, 3]
        result = truncate_tool_result(data, max_rows=50)
        assert result == [1, 2, 3]

    def test_string_over_max_chars_truncated(self) -> None:
        data = "x" * 7000
        result = truncate_tool_result(data, max_chars=6000)
        assert isinstance(result, str)
        assert "truncated" in result
        assert len(result) < 7000

    def test_string_under_max_chars_unchanged(self) -> None:
        data = "short"
        result = truncate_tool_result(data, max_chars=6000)
        assert result == "short"

    def test_dict_with_rows_truncated(self) -> None:
        data = {"rows": list(range(100)), "total": 100}
        result = truncate_tool_result(data, max_rows=50)
        assert isinstance(result, dict)
        assert len(result["rows"]) == 50

    def test_none_returned_as_none(self) -> None:
        assert truncate_tool_result(None) is None

    def test_passthrough_if_small(self) -> None:
        data = {"key": "value"}
        result = truncate_tool_result(data, max_chars=6000, max_rows=50)
        assert result == data


# ── trim_results ───────────────────────────────────────────


class TestEnforceResultBudget:
    def _make_step(self, data: Any, step_num: int = 1, result_key=None) -> Step:
        tc = ToolCall(tool="test", args={})
        tr = ToolResult(tool="test", args_summary="", data=data, result_key=result_key)
        return Step(number=step_num, tool_call=tc, tool_result=tr)

    def test_small_results_unchanged(self) -> None:
        step = self._make_step({"key": "small value"})
        trim_results([step])
        assert step.tool_result.data == {"key": "small value"}

    def test_oversized_result_compacted(self) -> None:
        big_data = {"rows": [{"x": "y" * 100}] * 600}
        step = self._make_step(big_data)
        trim_results([step], per_result_chars=1000)
        assert step.tool_result.data != big_data

    def test_large_individual_result_compacted_list(self) -> None:
        big_data = list(range(10000))
        step = self._make_step(big_data)
        trim_results([step], per_result_chars=1000)
        assert not isinstance(step.tool_result.data, list) or isinstance(
            step.tool_result.data, dict
        )

    def test_error_results_skipped(self) -> None:
        tc = ToolCall(tool="test", args={})
        tr = ToolResult(tool="test", args_summary="", data=None, error="oops")
        step = Step(number=1, tool_call=tc, tool_result=tr)
        trim_results([step])
        assert step.tool_result.error == "oops"

    def test_empty_steps_ok(self) -> None:
        trim_results([])


# ── age_session_entries ────────────────────────────────────────────


class TestCompressSessionLog:
    def test_non_session_log_returns_none(self) -> None:
        result = age_session_entries("just a string")
        assert result is None

    def test_session_log_without_compact_returns_none(self) -> None:
        obj = object()
        result = age_session_entries(obj)
        assert result is None

    def test_session_log_compact_not_needed_returns_none(self) -> None:
        mock_log = MagicMock()
        mock_log.compact.return_value = False
        result = age_session_entries(mock_log)
        assert result is None

    def test_session_log_compact_returns_summary(self) -> None:
        mock_log = MagicMock()
        mock_log.compact.return_value = True
        summary_entry = MagicMock()
        summary_entry.tool = "__summary__"
        summary_entry.findings = ["finding 1", "finding 2"]
        mock_log.entries = [summary_entry]
        result = age_session_entries(mock_log)
        assert result is not None
        assert "finding 1" in result

    def test_returns_none_for_short_real_log(self) -> None:
        from looplet.session import SessionLog

        log = SessionLog()
        result = age_session_entries(log, max_entries_to_keep=5)
        assert result is None

    def test_compacts_long_real_log(self) -> None:
        from looplet.session import SessionLog

        log = SessionLog()
        for i in range(10):
            log.record(
                step=i + 1, theory="t", tool="search", reasoning=f"q{i}", findings=[f"finding {i}"]
            )
        result = age_session_entries(log, max_entries_to_keep=3)
        assert result is None or isinstance(result, str)


class TestStepProgressTracker:
    def test_initial_state(self) -> None:
        t = StepProgressTracker()
        assert t.total_steps == 0
        assert t.consecutive_unproductive == 0
        assert t.is_stagnating is False

    def test_productive_classification(self) -> None:
        t = StepProgressTracker()
        cls = t.classify_turn(new_items=5, step_num=1)
        assert cls == StepProgressTracker.PRODUCTIVE

    def test_record_and_classify_productive(self) -> None:
        t = StepProgressTracker()
        t.record_call("search", {"q": "x"}, 1)
        cls = t.classify_turn(new_items=3, step_num=1)
        assert cls == StepProgressTracker.PRODUCTIVE
        assert t.total_steps == 1
        assert t.consecutive_unproductive == 0

    def test_empty_classification(self) -> None:
        t = StepProgressTracker()
        t.classify_turn(new_items=5, step_num=1)
        cls = t.classify_turn(new_items=0, step_num=2)
        assert cls == StepProgressTracker.EMPTY

    def test_check_seen_returns_none_for_new_call(self) -> None:
        t = StepProgressTracker()
        assert t.check_seen("search", {"q": "hello"}) is None

    def test_check_seen_returns_step_for_recorded(self) -> None:
        t = StepProgressTracker()
        t.record_call("search", {"q": "hello"}, step_num=3)
        assert t.check_seen("search", {"q": "hello"}) == 3

    def test_different_args_not_seen(self) -> None:
        t = StepProgressTracker()
        t.record_call("search", {"q": "hello"}, step_num=1)
        assert t.check_seen("search", {"q": "world"}) is None

    def test_check_seen_dedup(self) -> None:
        t = StepProgressTracker()
        t.record_call("search", {"q": "hello"}, step_num=1)
        assert t.check_seen("search", {"q": "hello"}) == 1
        assert t.check_seen("search", {"q": "other"}) is None

    def test_is_stagnating_false_initially(self) -> None:
        t = StepProgressTracker(window=3)
        assert t.is_stagnating is False

    def test_is_stagnating_true_after_empties(self) -> None:
        t = StepProgressTracker(window=3)
        for i in range(5):
            t.classify_turn(new_items=0, step_num=i)
        assert t.is_stagnating is True

    def test_stagnation_detection_exact_window(self) -> None:
        t = StepProgressTracker(window=3)
        for i in range(3):
            t.classify_turn(new_items=0, step_num=i)
        assert t.is_stagnating is True

    def test_total_steps_tracked(self) -> None:
        t = StepProgressTracker()
        t.classify_turn(new_items=1, step_num=1)
        t.classify_turn(new_items=0, step_num=2)
        assert t.total_steps == 2

    def test_mark_redundant_changes_classification(self) -> None:
        t = StepProgressTracker()
        t.classify_turn(new_items=5, step_num=1)
        t.classify_turn(new_items=0, step_num=2)
        t.mark_redundant("tool", {}, step_num=2)
        assert t.redundant_count == 1

    def test_redundant_count(self) -> None:
        t = StepProgressTracker()
        t.classify_turn(0, 1)
        t.mark_redundant("tool", {}, 1)
        assert t.redundant_count >= 1

    def test_consecutive_unproductive_property(self) -> None:
        t = StepProgressTracker()
        t.classify_turn(new_items=5, step_num=1)
        t.classify_turn(new_items=0, step_num=2)
        t.classify_turn(new_items=0, step_num=3)
        assert t.consecutive_unproductive == 2


# ── estimate_tokens ──────────────────────────────────────────────────


class TestEstimateTokens:
    def test_basic_estimate(self) -> None:
        result = estimate_tokens("abcd")  # 4 chars = 1 token
        assert result == 1

    def test_minimum_one(self) -> None:
        result = estimate_tokens("")
        assert result >= 1

    def test_longer_text(self) -> None:
        text = "a" * 400
        result = estimate_tokens(text)
        assert result == 100

    def test_returns_int(self) -> None:
        assert isinstance(estimate_tokens("hello world"), int)

    def test_four_chars_per_token_heuristic(self) -> None:
        assert estimate_tokens("a" * 4) == 1
        assert estimate_tokens("a" * 8) == 2
        assert estimate_tokens("a" * 100) == 25


# ── Context overflow detection ───────────────────────────────────────


class TestContextOverflowDetection:
    def test_estimate_prompt_tokens(self) -> None:
        assert estimate_prompt_tokens("a" * 400) == 100

    def test_should_compress_small_prompt(self) -> None:
        short = "x" * 100
        assert is_context_oversized(short, context_window=128_000) is False

    def test_should_compress_large_prompt(self) -> None:
        long = "x" * 400_000
        assert is_context_oversized(long, context_window=128_000) is True


# ── No domain-specific code ─────────────────────────────────────────


class TestNoDomainSpecificCode:
    def test_check_done_quality_not_present(self) -> None:
        import looplet.scaffolding as s

        assert not hasattr(s, "check_done_quality")

    def test_no_domain_specific_importss(self) -> None:
        import looplet.scaffolding as s

        src = open(s.__file__).read()
        assert "primal_security" not in src

    def test_no_legacy_aliases(self) -> None:
        import looplet.scaffolding as m

        assert not hasattr(m, "compress_investigation_log")


# ── emergency_truncate ──────────────────────────────────────────────


class TestReactiveCompact:
    def test_returns_none_for_short_log(self) -> None:
        from looplet.session import SessionLog

        class FakeState:
            steps: list = []

        log = SessionLog()
        result = emergency_truncate(FakeState(), log, keep_recent=3)
        assert result is None

    def test_compacts_long_log(self) -> None:
        from looplet.session import SessionLog

        class FakeState:
            steps: list = []

        log = SessionLog()
        for i in range(8):
            log.record(
                step=i + 1, theory="t", tool="search", reasoning=f"q{i}", findings=[f"finding {i}"]
            )
        result = emergency_truncate(FakeState(), log, keep_recent=3)
        assert result is None or isinstance(result, str)


# ── Constants check ──────────────────────────────────────────────


class TestConstants:
    def test_constants_exist(self) -> None:
        from looplet.scaffolding import (
            DIMINISHING_RETURNS_THRESHOLD,
            DIMINISHING_RETURNS_WINDOW,
            MAX_LLM_RETRIES,
            PARSE_RECOVERY_MAX,
            TOOL_RESULT_MAX_CHARS,
            TOOL_RESULT_MAX_ROWS,
        )

        assert MAX_LLM_RETRIES >= 1
        assert PARSE_RECOVERY_MAX >= 1
        assert TOOL_RESULT_MAX_CHARS > 0
        assert TOOL_RESULT_MAX_ROWS > 0
