"""Tests for :mod:`looplet.resilient`."""

from __future__ import annotations

import time

import pytest

from looplet.resilient import ResilientBackend, RetryExhausted


class FlakyBackend:
    """Raises a configurable sequence of exceptions, then succeeds."""

    def __init__(self, errors: list[BaseException], result: str = "ok") -> None:
        self._errors = list(errors)
        self._result = result
        self.calls = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        if self._errors:
            exc = self._errors.pop(0)
            raise exc
        return self._result


class ToolsCapableFlakyBackend(FlakyBackend):
    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return {"tool": "done", "args": {}}


class TestRetryBehavior:
    def test_success_first_try(self) -> None:
        inner = FlakyBackend(errors=[], result="hi")
        llm = ResilientBackend(inner, retries=3, sleep=lambda _s: None)
        assert llm.generate("p") == "hi"
        assert inner.calls == 1

    def test_retries_then_succeeds(self) -> None:
        inner = FlakyBackend(errors=[RuntimeError("x"), RuntimeError("y")], result="ok")
        sleeps: list[float] = []
        llm = ResilientBackend(inner, retries=3, sleep=sleeps.append, jitter=0)
        assert llm.generate("p") == "ok"
        assert inner.calls == 3
        assert len(sleeps) == 2
        # Exponential backoff: 0.5, 1.0
        assert sleeps[0] == pytest.approx(0.5)
        assert sleeps[1] == pytest.approx(1.0)

    def test_exhausts_retries(self) -> None:
        errs = [RuntimeError(f"e{i}") for i in range(5)]
        inner = FlakyBackend(errors=errs)
        llm = ResilientBackend(inner, retries=3, sleep=lambda _s: None)
        with pytest.raises(RetryExhausted) as excinfo:
            llm.generate("p")
        assert inner.calls == 3
        assert len(excinfo.value.attempts) == 3
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_retry_on_predicate_filters(self) -> None:
        inner = FlakyBackend(errors=[ValueError("don't retry me")])
        llm = ResilientBackend(
            inner,
            retries=5,
            sleep=lambda _s: None,
            retry_on=lambda exc: not isinstance(exc, ValueError),
        )
        with pytest.raises(ValueError):
            llm.generate("p")
        assert inner.calls == 1

    def test_keyboard_interrupt_not_retried(self) -> None:
        inner = FlakyBackend(errors=[KeyboardInterrupt()])
        llm = ResilientBackend(inner, retries=5, sleep=lambda _s: None)
        with pytest.raises(KeyboardInterrupt):
            llm.generate("p")
        assert inner.calls == 1


class TestTimeoutBehavior:
    def test_timeout_raises(self) -> None:
        class SlowBackend:
            def generate(self, prompt, **_kw):
                time.sleep(1.0)
                return "too late"

        llm = ResilientBackend(
            SlowBackend(),
            retries=1,
            timeout_s=0.05,
            sleep=lambda _s: None,
        )
        with pytest.raises(RetryExhausted) as excinfo:
            llm.generate("p")
        assert isinstance(excinfo.value.attempts[0], TimeoutError)

    def test_timeout_retries(self) -> None:
        state = {"slow_calls": 0}

        class HalfSlow:
            def generate(self, prompt, **_kw):
                state["slow_calls"] += 1
                if state["slow_calls"] == 1:
                    time.sleep(0.5)
                return "ok"

        llm = ResilientBackend(
            HalfSlow(),
            retries=3,
            timeout_s=0.05,
            sleep=lambda _s: None,
        )
        assert llm.generate("p") == "ok"


class TestGenerateWithTools:
    def test_wraps_generate_with_tools_when_available(self) -> None:
        inner = ToolsCapableFlakyBackend(errors=[RuntimeError("x")])
        llm = ResilientBackend(inner, retries=3, sleep=lambda _s: None)
        assert hasattr(llm, "generate_with_tools")
        assert llm.generate_with_tools("p", tools=[]) == {"tool": "done", "args": {}}
        assert inner.calls == 2

    def test_no_generate_with_tools_when_not_available(self) -> None:
        inner = FlakyBackend(errors=[])
        llm = ResilientBackend(inner, retries=1)
        assert not hasattr(llm, "generate_with_tools")


class TestBackoff:
    def test_jitter_applied(self) -> None:
        sleeps: list[float] = []
        inner = FlakyBackend(errors=[RuntimeError("a"), RuntimeError("b")])
        llm = ResilientBackend(
            inner,
            retries=3,
            sleep=sleeps.append,
            base_delay_s=1.0,
            max_delay_s=10.0,
            jitter=0.5,
        )
        llm.generate("p")
        # Both delays should fall within [0.5, 1.5] for attempt 1 (raw=1.0)
        # and [1.0, 3.0] for attempt 2 (raw=2.0).
        assert 0.5 <= sleeps[0] <= 1.5
        assert 1.0 <= sleeps[1] <= 3.0

    def test_max_delay_caps_backoff(self) -> None:
        sleeps: list[float] = []
        errs = [RuntimeError(f"e{i}") for i in range(10)]
        inner = FlakyBackend(errors=errs)
        llm = ResilientBackend(
            inner,
            retries=10,
            sleep=sleeps.append,
            base_delay_s=1.0,
            max_delay_s=2.0,
            jitter=0,
        )
        with pytest.raises(RetryExhausted):
            llm.generate("p")
        # Raw delays: 1, 2, 4, 8, ... — cap at 2.0 means everything >= 2 is 2
        for d in sleeps[2:]:
            assert d == pytest.approx(2.0)


class TestValidation:
    def test_retries_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ResilientBackend(FlakyBackend(errors=[]), retries=0)
