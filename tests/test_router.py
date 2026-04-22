"""Tests for looplet.router — multi-model routing and cost tracking."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from looplet.router import (
    CostTracker,
    FallbackRouter,
    ModelProfile,
    ModelRouter,
    RoutingLLMBackend,
    SimpleRouter,
)
from looplet.types import LLMBackend

# ── Helpers ─────────────────────────────────────────────────────


class MockLLM:
    """Minimal LLMBackend for testing."""

    def __init__(self, response: str = "mock response") -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self.response


class FailingLLM:
    """LLMBackend that always raises."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        raise RuntimeError("primary failed")


# ── ModelProfile ─────────────────────────────────────────────────


def test_model_profile_creation():
    llm = MockLLM()
    profile = ModelProfile(name="gpt-4", backend=llm)
    assert profile.name == "gpt-4"
    assert profile.backend is llm
    assert profile.cost_per_1k_input == 0.0
    assert profile.cost_per_1k_output == 0.0
    assert profile.context_window == 100000
    assert profile.strengths == []


def test_model_profile_with_costs():
    llm = MockLLM()
    profile = ModelProfile(
        name="gpt-4",
        backend=llm,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        context_window=128000,
        strengths=["reasoning", "code"],
    )
    assert profile.cost_per_1k_input == 0.03
    assert profile.cost_per_1k_output == 0.06
    assert profile.context_window == 128000
    assert "reasoning" in profile.strengths


def test_model_profile_strengths_independent():
    """Each profile should have its own strengths list."""
    a = ModelProfile(name="a", backend=MockLLM())
    b = ModelProfile(name="b", backend=MockLLM())
    a.strengths.append("fast")
    assert b.strengths == []


# ── ModelRouter Protocol ─────────────────────────────────────────


def test_model_router_is_protocol():
    assert hasattr(ModelRouter, "select")


def test_simple_router_satisfies_model_router():
    router = SimpleRouter({}, default_profile=ModelProfile("d", MockLLM()))
    assert isinstance(router, ModelRouter)


# ── SimpleRouter ──────────────────────────────────────────────────


def test_simple_router_selects_by_purpose():
    fast = MockLLM("fast")
    smart = MockLLM("smart")
    profiles = {
        "reasoning": ModelProfile("smart", smart),
        "recovery": ModelProfile("fast", fast),
    }
    default = ModelProfile("default", MockLLM("default"))
    router = SimpleRouter(profiles, default_profile=default)

    assert router.select("reasoning") is smart
    assert router.select("recovery") is fast


def test_simple_router_falls_back_to_default():
    default_llm = MockLLM("default")
    router = SimpleRouter(
        {"reasoning": ModelProfile("x", MockLLM())},
        default_profile=ModelProfile("default", default_llm),
    )
    result = router.select("unknown_purpose")
    assert result is default_llm


def test_simple_router_select_returns_llm_backend():
    llm = MockLLM()
    router = SimpleRouter(
        {"reasoning": ModelProfile("m", llm)},
        default_profile=ModelProfile("d", MockLLM()),
    )
    backend = router.select("reasoning")
    # Should satisfy LLMBackend protocol
    result = backend.generate("hello")
    assert result == "mock response"


def test_simple_router_empty_profiles_uses_default():
    default_llm = MockLLM("fallback")
    router = SimpleRouter({}, default_profile=ModelProfile("d", default_llm))
    assert router.select("reasoning") is default_llm
    assert router.select("compaction") is default_llm


# ── FallbackRouter ───────────────────────────────────────────────


def test_fallback_router_uses_primary_when_ok():
    primary = MockLLM("primary response")
    fallback = MockLLM("fallback response")
    router = FallbackRouter(primary=primary, fallback=fallback)

    backend = router.select("reasoning")
    result = backend.generate("test prompt")
    assert result == "primary response"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


def test_fallback_router_switches_to_fallback_on_error():
    primary = FailingLLM()
    fallback = MockLLM("fallback response")
    router = FallbackRouter(primary=primary, fallback=fallback)

    backend = router.select("reasoning")
    result = backend.generate("test prompt")
    assert result == "fallback response"


def test_fallback_router_always_returns_same_backend():
    primary = MockLLM()
    fallback = MockLLM()
    router = FallbackRouter(primary=primary, fallback=fallback)

    b1 = router.select("reasoning")
    b2 = router.select("compaction")
    # Both select calls return a backend that wraps primary+fallback
    assert b1.generate("x") == "mock response"
    assert b2.generate("x") == "mock response"


def test_fallback_router_passes_kwargs_through():
    primary = MockLLM()
    fallback = MockLLM()
    router = FallbackRouter(primary=primary, fallback=fallback)
    backend = router.select("reasoning")
    backend.generate("test", max_tokens=500, temperature=0.5)
    assert primary.calls[0]["max_tokens"] == 500


def test_fallback_router_satisfies_model_router():
    router = FallbackRouter(primary=MockLLM(), fallback=MockLLM())
    assert isinstance(router, ModelRouter)


# ── CostTracker ──────────────────────────────────────────────────


def test_cost_tracker_initial_state():
    llm = MockLLM()
    tracker = CostTracker(backend=llm, cost_per_1k_input=0.01, cost_per_1k_output=0.02)
    assert tracker.total_cost == 0.0
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0
    assert tracker.call_count == 0


def test_cost_tracker_counts_calls():
    llm = MockLLM("hello world")
    tracker = CostTracker(backend=llm, cost_per_1k_input=0.0, cost_per_1k_output=0.0)
    tracker.generate("prompt")
    tracker.generate("another prompt")
    assert tracker.call_count == 2


def test_cost_tracker_accumulates_tokens():
    llm = MockLLM("one two three")  # 3 words → some output tokens
    tracker = CostTracker(backend=llm, cost_per_1k_input=0.0, cost_per_1k_output=0.0)
    tracker.generate("input word count test here")  # 5 words input
    assert tracker.total_input_tokens > 0
    assert tracker.total_output_tokens > 0


def test_cost_tracker_accumulates_cost():
    llm = MockLLM("response text here")
    tracker = CostTracker(
        backend=llm,
        cost_per_1k_input=1.0,  # $1 per 1k input tokens
        cost_per_1k_output=2.0,  # $2 per 1k output tokens
    )
    tracker.generate("hello world test input")
    assert tracker.total_cost > 0.0


def test_cost_tracker_cost_grows_with_calls():
    llm = MockLLM("some output")
    tracker = CostTracker(backend=llm, cost_per_1k_input=1.0, cost_per_1k_output=1.0)
    tracker.generate("first call prompt")
    cost_after_1 = tracker.total_cost
    tracker.generate("second call prompt")
    assert tracker.total_cost > cost_after_1


def test_cost_tracker_report_returns_string():
    llm = MockLLM()
    tracker = CostTracker(backend=llm, cost_per_1k_input=0.03, cost_per_1k_output=0.06)
    tracker.generate("test prompt")
    report = tracker.report()
    assert isinstance(report, str)
    assert len(report) > 0


def test_cost_tracker_report_contains_summary_info():
    llm = MockLLM("hello")
    tracker = CostTracker(backend=llm, cost_per_1k_input=0.03, cost_per_1k_output=0.06)
    tracker.generate("hello world")
    tracker.generate("another call")
    report = tracker.report()
    # Should mention calls and cost
    assert "2" in report or "call" in report.lower()


def test_cost_tracker_delegating_generate():
    """CostTracker.generate must return the backend's response unchanged."""
    llm = MockLLM("expected output")
    tracker = CostTracker(backend=llm, cost_per_1k_input=0.0, cost_per_1k_output=0.0)
    result = tracker.generate("test", max_tokens=100, temperature=0.1)
    assert result == "expected output"
    assert llm.calls[0]["max_tokens"] == 100


def test_cost_tracker_satisfies_llm_backend():
    tracker = CostTracker(backend=MockLLM(), cost_per_1k_input=0.0, cost_per_1k_output=0.0)
    assert isinstance(tracker, LLMBackend)


# ── RoutingLLMBackend ────────────────────────────────────────────


def test_routing_llm_backend_default_purpose():
    llm = MockLLM("routed response")
    router = SimpleRouter(
        {"reasoning": ModelProfile("m", llm)},
        default_profile=ModelProfile("d", MockLLM()),
    )
    backend = RoutingLLMBackend(router=router)
    assert backend.purpose == "reasoning"


def test_routing_llm_backend_generate_delegates():
    llm = MockLLM("from reasoning backend")
    router = SimpleRouter(
        {"reasoning": ModelProfile("m", llm)},
        default_profile=ModelProfile("d", MockLLM()),
    )
    backend = RoutingLLMBackend(router=router)
    result = backend.generate("hello")
    assert result == "from reasoning backend"


def test_routing_llm_backend_set_purpose():
    reasoning_llm = MockLLM("reasoning")
    recovery_llm = MockLLM("recovery")
    router = SimpleRouter(
        {
            "reasoning": ModelProfile("r", reasoning_llm),
            "recovery": ModelProfile("rec", recovery_llm),
        },
        default_profile=ModelProfile("d", MockLLM()),
    )
    backend = RoutingLLMBackend(router=router)
    assert backend.generate("test") == "reasoning"

    backend.set_purpose("recovery")
    assert backend.purpose == "recovery"
    assert backend.generate("test") == "recovery"


def test_routing_llm_backend_satisfies_llm_backend():
    router = SimpleRouter({}, default_profile=ModelProfile("d", MockLLM()))
    backend = RoutingLLMBackend(router=router)
    assert isinstance(backend, LLMBackend)


def test_routing_llm_backend_passes_kwargs():
    llm = MockLLM()
    router = SimpleRouter(
        {"reasoning": ModelProfile("m", llm)},
        default_profile=ModelProfile("d", MockLLM()),
    )
    backend = RoutingLLMBackend(router=router)
    backend.generate("prompt", max_tokens=300, temperature=0.8)
    assert llm.calls[0]["max_tokens"] == 300
