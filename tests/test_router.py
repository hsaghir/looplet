"""Tests for looplet.router - multi-model routing and cost tracking."""

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


@pytest.mark.parametrize("kind", ["cost", "routing", "fallback"])
@pytest.mark.parametrize("native", [False, True])
def test_wrappers_preserve_supported_options_and_active_usage(kind, native):
    class Backend(MockLLM):
        last_usage = {"total_tokens": 7}

        def generate(self, prompt, **kwargs):
            self.options = kwargs
            return "ok"

        def generate_with_tools(self, prompt, *, tools, **kwargs):
            self.options = kwargs
            return [{"type": "tool_use", "name": "done", "input": {}}]

    backend = Backend()
    if kind == "cost":
        wrapped = CostTracker(backend)
    elif kind == "routing":
        wrapped = RoutingLLMBackend(
            SimpleRouter({}, default_profile=ModelProfile("default", backend))
        )
    else:
        wrapped = FallbackRouter(primary=FailingLLM(), fallback=backend).select("reasoning")
        if native:
            wrapped = FallbackRouter(primary=backend, fallback=backend).select("reasoning")
    method = wrapped.generate_with_tools if native else wrapped.generate
    method("prompt", reasoning_effort="high", **({"tools": []} if native else {}))

    assert backend.options["reasoning_effort"] == "high"
    assert wrapped.last_usage == {"total_tokens": 7}


@pytest.mark.parametrize("kind", ["routing", "fallback"])
@pytest.mark.parametrize("call_first", [False, True])
def test_mutable_provider_fields_reach_the_capability_owner(kind, call_first):
    from looplet.provenance import RecordingLLMBackend
    from looplet.resilient import ResilientBackend

    class Backend:
        def __init__(self, *, fail=False):
            self.fail = fail
            self._client = object()
            self._state_path = "original"
            self.last_usage = {"total_tokens": 1}

        def generate(self, prompt, **kwargs):
            if self.fail:
                raise RuntimeError("primary failed")
            return "ok"

    first = Backend(fail=kind == "fallback")
    second = Backend()
    if kind == "routing":
        routed = RoutingLLMBackend(
            SimpleRouter(
                {"next": ModelProfile("next", second)},
                default_profile=ModelProfile("first", first),
            )
        )
    else:
        routed = FallbackRouter(primary=first, fallback=second).select("reasoning")
    wrapped = ResilientBackend(CostTracker(RecordingLLMBackend(routed)), retries=1)
    if call_first:
        wrapped.generate("prompt")
        if kind == "routing":
            routed.set_purpose("next")
    owner = second if kind == "fallback" and call_first else first
    untouched = first if owner is second else second
    untouched_client = untouched._client
    replacement = object()
    assert wrapped._client is owner._client

    wrapped._client = replacement
    wrapped._state_path = "restored"
    wrapped.last_usage = {"total_tokens": 9}

    assert owner._client is replacement and wrapped._client is replacement
    assert owner._state_path == wrapped._state_path == "restored"
    assert owner.last_usage == wrapped.last_usage == {"total_tokens": 9}
    assert untouched._client is untouched_client
    assert untouched._state_path == "original"
    assert untouched.last_usage == {"total_tokens": 1}
    assert not {"_client", "_state_path", "last_usage"}.intersection(routed.__dict__)


def test_retry_ownership_follows_next_route_but_usage_follows_last_call():
    from looplet._backend_contract import attempt_limit
    from looplet.resilient import ResilientBackend

    first = MockLLM("first")
    first.last_usage = {"total_tokens": 7}
    second = MockLLM("second")
    second.last_usage = {"total_tokens": 9}
    managed = ResilientBackend(first, retries=1)
    router = SimpleRouter(
        {"managed": ModelProfile("first", managed), "plain": ModelProfile("second", second)},
        default_profile=ModelProfile("first", managed),
    )
    wrapped = RoutingLLMBackend(router, default_purpose="managed")
    assert attempt_limit(wrapped, 2) == 1
    wrapped.generate("prompt")
    wrapped.set_purpose("plain")

    assert attempt_limit(wrapped, 2) == 3
    assert wrapped.last_usage == {"total_tokens": 7}
    wrapped.generate("prompt")
    assert wrapped.last_usage == {"total_tokens": 9}


@pytest.mark.parametrize("first_budgeted", [False, True])
def test_request_budgets_follow_next_route(first_budgeted):
    from looplet.subagent import SharedModelBudget

    budget = SharedModelBudget(2)
    first, second = MockLLM("first"), MockLLM("second")
    routed = RoutingLLMBackend(
        SimpleRouter(
            {"next": ModelProfile("second", second if first_budgeted else budget.wrap(second))},
            default_profile=ModelProfile("first", budget.wrap(first) if first_budgeted else first),
        )
    )
    wrapped = budget.wrap(routed)
    assert wrapped.generate("first") == "first"
    routed.set_purpose("next")
    assert wrapped.generate("second") == "second"
    assert budget.remaining == 0


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


@pytest.mark.parametrize("primary_async", [False, True])
@pytest.mark.parametrize("fallback_async", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_fallback_router_recovers_sync_async_failure(primary_async, fallback_async, native):
    import inspect
    import threading

    observed = []
    owner_thread = threading.get_ident()
    expected = [{"type": "text", "text": "fallback"}] if native else "fallback"

    class Backend:
        def __init__(self, name):
            self.name = name
            self.last_usage = {"total_tokens": 7 if name == "primary" else 9}

        def generate(self, prompt, **kwargs):
            observed.append(self.name)
            self.options = kwargs
            if self.name == "primary":
                raise RuntimeError("primary failed")
            if primary_async and not fallback_async:
                assert threading.get_ident() != owner_thread
            return expected

        generate_with_tools = generate

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        generate_with_tools = generate

    primary = (AsyncBackend if primary_async else Backend)("primary")
    fallback = (AsyncBackend if fallback_async else Backend)("fallback")
    wrapped = FallbackRouter(primary=primary, fallback=fallback).select("reasoning")
    method = wrapped.generate_with_tools if native else wrapped.generate
    pending = method("request", reasoning_effort="high", **({"tools": []} if native else {}))
    response = await pending if inspect.isawaitable(pending) else pending

    assert response == expected
    assert observed == ["primary", "fallback"]
    assert fallback.options["reasoning_effort"] == "high"
    assert wrapped.last_usage == fallback.last_usage


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_fallback_router_async_admission_and_cancellation(native, cancelled):
    import asyncio
    import inspect

    from looplet.subagent import SharedModelBudget

    observed = []
    failure = asyncio.CancelledError("cancelled") if cancelled else RuntimeError("primary failed")

    class Primary:
        async def generate(self, prompt, **kwargs):
            observed.append("primary")
            raise failure

        generate_with_tools = generate

    class Fallback:
        async def generate(self, prompt, **kwargs):
            observed.append("fallback")
            return "fallback"

        generate_with_tools = generate

    routed = FallbackRouter(primary=Primary(), fallback=Fallback()).select("reasoning")
    budget = SharedModelBudget(1)
    wrapped = routed if cancelled else budget.wrap(routed)
    method = wrapped.generate_with_tools if native else wrapped.generate
    with pytest.raises(type(failure)) as caught:
        pending = method("request", **({"tools": []} if native else {}))
        if inspect.isawaitable(pending):
            await pending

    assert caught.value is failure
    assert observed == ["primary"]
    if not cancelled:
        assert budget.remaining == 0


@pytest.mark.parametrize("primary_async", [False, True])
@pytest.mark.parametrize("fallback_async", [False, True])
@pytest.mark.parametrize("route_fallback", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("admitted", [False, True])
async def test_fallback_stream_switches_only_before_delivered_prefix(
    primary_async, fallback_async, route_fallback, partial, admitted
):
    from contextlib import nullcontext

    from looplet.backend_contract import single_model_request

    observed = []
    closed = []
    delivered = []
    failure = ConnectionError("primary stream failed")

    class Backend:
        def __init__(self, name):
            self.name = name
            self.last_usage = {"total_tokens": 7 if name == "primary" else 9}

        def generate(self, prompt, **kwargs):
            return "unused"

        def stream(self, prompt, **kwargs):
            observed.append(self.name)
            try:
                if self.name == "primary":
                    if partial:
                        yield "partial"
                    raise failure
                yield "fallback"
            finally:
                closed.append(self.name)

    class AsyncBackend(Backend):
        async def stream(self, prompt, **kwargs):
            iterator = super().stream(prompt, **kwargs)
            try:
                for chunk in iterator:
                    yield chunk
            finally:
                iterator.close()

    primary = (AsyncBackend if primary_async else Backend)("primary")
    fallback = (AsyncBackend if fallback_async else Backend)("fallback")
    if route_fallback:
        fallback = RoutingLLMBackend(
            SimpleRouter({}, default_profile=ModelProfile("fallback", fallback))
        )
    wrapped = FallbackRouter(primary=primary, fallback=fallback).select("reasoning")

    async def consume():
        iterator = wrapped.stream("request")
        if hasattr(iterator, "__aiter__"):
            async for chunk in iterator:
                delivered.append(chunk)
        else:
            delivered.extend(iterator)

    with single_model_request() if admitted else nullcontext():
        if partial or admitted:
            with pytest.raises(ConnectionError) as caught:
                await consume()
            assert caught.value is failure
        else:
            await consume()

    expected_chunks = ["partial"] if partial else ([] if admitted else ["fallback"])
    assert delivered == expected_chunks
    assert observed == (["primary"] if partial or admitted else ["primary", "fallback"])
    assert closed == observed
    if not partial and not admitted:
        assert wrapped.last_usage == {"total_tokens": 9}


@pytest.mark.parametrize("primary_failure", [False, True])
@pytest.mark.parametrize("fallback_async", [False, True])
async def test_async_fallback_stream_factories_preserve_protocol_and_worker(
    primary_failure, fallback_async
):
    from threading import get_ident

    loop_thread = get_ident()
    factories = []
    sync_threads = []
    closed = []

    class Stream:
        def __init__(self, name):
            self.name = name
            self.iterator = iter([name])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterator)

        def close(self):
            closed.append(self.name)

    class Primary:
        def generate(self, prompt, **kwargs):
            return "unused"

        def stream(self, prompt, **kwargs):
            factories.append("primary")
            sync_threads.append(get_ident())
            if primary_failure:
                raise RuntimeError("primary factory failed")
            return Stream("primary")

    class Fallback(Primary):
        def stream(self, prompt, **kwargs):
            factories.append("fallback")
            sync_threads.append(get_ident())
            return Stream("fallback")

    class AsyncFallback(Fallback):
        async def stream(self, prompt, **kwargs):
            factories.append("fallback")
            try:
                yield "fallback"
            finally:
                closed.append("fallback")

    fallback = AsyncFallback() if fallback_async else Fallback()
    wrapped = FallbackRouter(primary=Primary(), fallback=fallback).select("reasoning")
    assert [chunk async for chunk in wrapped.stream("request")] == [
        "fallback" if primary_failure else "primary"
    ]
    assert factories == (["primary", "fallback"] if primary_failure else ["primary"])
    assert closed == (["fallback"] if primary_failure else ["primary"])
    assert sync_threads and all(thread != loop_thread for thread in sync_threads)


@pytest.mark.parametrize("fallback_factory_failure", [False, True])
def test_sync_fallback_stream_has_one_cleanup_owner(fallback_factory_failure):
    factories = []
    closed = []
    primary_error = RuntimeError("primary stream failed")
    fallback_error = RuntimeError("fallback factory failed")

    class Stream:
        def __init__(self, name):
            self.name = name
            self.iterator = iter([name])

        def __iter__(self):
            return self

        def __next__(self):
            if self.name == "primary":
                raise primary_error
            return next(self.iterator)

        def close(self):
            closed.append(self.name)

    class Backend:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, **kwargs):
            return "unused"

        def stream(self, prompt, **kwargs):
            factories.append(self.name)
            if self.name == "fallback" and fallback_factory_failure:
                raise fallback_error
            return Stream(self.name)

    wrapped = FallbackRouter(primary=Backend("primary"), fallback=Backend("fallback")).select(
        "reasoning"
    )
    iterator = wrapped.stream("request")
    if fallback_factory_failure:
        with pytest.raises(RuntimeError) as caught:
            next(iterator)
        assert caught.value is fallback_error
    else:
        assert next(iterator) == "fallback"
        iterator.close()
    assert factories == ["primary", "fallback"]
    assert closed == (["primary"] if fallback_factory_failure else ["primary", "fallback"])


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fallback_stream_close_before_first_read(asynchronous):
    factories = []

    class Backend:
        def generate(self, prompt, **kwargs):
            return "unused"

        def stream(self, prompt, **kwargs):
            factories.append(prompt)
            yield "unused"

    backend = Backend()
    wrapped = FallbackRouter(primary=backend, fallback=backend).select("reasoning")
    iterator = wrapped.stream("request")
    if asynchronous:
        await iterator.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
    else:
        iterator.close()
        with pytest.raises(StopIteration):
            next(iterator)
    assert factories == []


async def test_fallback_stream_factory_cancellation_closes_created_source():
    import asyncio
    from threading import Event

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    released = Event()
    factories = []
    closed = []

    class Stream:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("Cancelled factory must not advance its stream")

        def close(self):
            closed.append(True)

    class Backend:
        def generate(self, prompt, **kwargs):
            return "unused"

        def stream(self, prompt, **kwargs):
            factories.append(prompt)
            stream = Stream()
            loop.call_soon_threadsafe(started.set)
            assert released.wait(5)
            return stream

    backend = Backend()
    wrapped = FallbackRouter(primary=backend, fallback=backend).select("reasoning")
    iterator = wrapped.stream("request")
    pending = asyncio.create_task(anext(iterator))
    try:
        await started.wait()
        pending.cancel("cancel factory")
    finally:
        released.set()
    with pytest.raises(asyncio.CancelledError, match="cancel factory"):
        await pending
    assert factories == ["request"] and closed == [True]


@pytest.mark.parametrize("termination", ["complete", "close"])
async def test_async_fallback_reads_preserve_sync_generator_context(termination):
    from contextvars import ContextVar

    from looplet.provenance import RecordingLLMBackend

    marker = ContextVar("stream_marker", default="host")
    closed = []

    class Backend:
        def generate(self, prompt, **kwargs):
            return "unused"

        def stream(self, prompt, **kwargs):
            token = marker.set("stream")
            try:
                yield "partial"
                assert marker.get() == "stream"
                yield "rest"
            finally:
                marker.reset(token)
                closed.append(marker.get())

    recorder = RecordingLLMBackend(Backend())
    wrapped = FallbackRouter(primary=recorder, fallback=Backend()).select("reasoning")
    iterator = wrapped.stream("request")
    assert await anext(iterator) == "partial"
    assert marker.get() == "host"
    if termination == "close":
        await iterator.aclose()
    else:
        assert [chunk async for chunk in iterator] == ["rest"]

    assert marker.get() == "host" and closed == ["host"]
    assert len(recorder.calls) == 1
    assert recorder.calls[0].response == ("partial" if termination == "close" else "partialrest")


@pytest.mark.parametrize("native", [False, True])
def test_fallback_router_honors_single_request_admission(native):
    from looplet.backend_contract import backend_request_policy
    from looplet.subagent import SharedModelBudget

    observed = []

    class Backend(MockLLM):
        def generate(self, prompt, **kwargs):
            observed.append(self.response)
            if self.response == "primary":
                raise RuntimeError("primary failed")
            return self.response

        def generate_with_tools(self, prompt, *, tools, **kwargs):
            return self.generate(prompt, **kwargs)

    routed = FallbackRouter(primary=Backend("primary"), fallback=Backend("fallback")).select(
        "reasoning"
    )
    budget = SharedModelBudget(1)
    wrapped = budget.wrap(routed)
    method = wrapped.generate_with_tools if native else wrapped.generate
    with pytest.raises(RuntimeError, match="primary failed"):
        method("prompt", **({"tools": []} if native else {}))

    assert observed == ["primary"]
    assert budget.remaining == 0
    assert backend_request_policy(routed).manages_retries
    assert routed.generate("prompt") == "fallback"
    assert observed == ["primary", "primary", "fallback"]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("wrapper", ["fallback", "resilient"])
def test_wrappers_reject_unsupported_retry_owner_before_admission(legacy, wrapper):
    from looplet.backend_contract import BackendRequestPolicy, TerminalLLMError
    from looplet.resilient import ResilientBackend
    from looplet.subagent import SharedModelBudget

    primary = MockLLM("primary")
    if legacy:
        primary._looplet_manages_retries = True
    else:
        primary.request_policy = BackendRequestPolicy(manages_retries=True)
    routed = (
        FallbackRouter(primary=primary, fallback=MockLLM()).select("reasoning")
        if wrapper == "fallback"
        else ResilientBackend(primary, retries=3)
    )
    budget = SharedModelBudget(1)

    with pytest.raises(TerminalLLMError, match="single-request admission"):
        budget.wrap(routed).generate("prompt")

    assert primary.calls == []
    assert budget.remaining == 1


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("wrapper", ["fallback", "resilient"])
def test_manual_admission_rejects_unsupported_nested_retry_owner(native, wrapper):
    from looplet.backend_contract import BackendRequestPolicy, TerminalLLMError
    from looplet.resilient import ResilientBackend
    from looplet.subagent import SharedModelBudget

    class Backend(MockLLM):
        request_policy = BackendRequestPolicy(manages_retries=True)

        def generate_with_tools(self, prompt, *, tools, **kwargs):
            return self.generate(prompt)

    primary = Backend()
    routed = (
        FallbackRouter(primary=primary, fallback=Backend()).select("reasoning")
        if wrapper == "fallback"
        else ResilientBackend(primary, retries=3)
    )
    budget = SharedModelBudget(1)
    method = routed.generate_with_tools if native else routed.generate
    with pytest.raises(TerminalLLMError, match="single-request admission"):
        with budget.admit():
            method("prompt", **({"tools": []} if native else {}))

    assert primary.calls == []
    assert budget.remaining == 0


@pytest.mark.parametrize("primary_budgeted", [False, True])
def test_fallback_budget_metadata_follows_next_primary(primary_budgeted):
    from looplet.subagent import SharedModelBudget

    class Backend(MockLLM):
        def generate(self, prompt, **kwargs):
            if self.response == "fail":
                raise RuntimeError("primary failed")
            return self.response

    budget = SharedModelBudget(2)
    primary, fallback = Backend("fail"), MockLLM("fallback")
    routed = FallbackRouter(
        primary=budget.wrap(primary) if primary_budgeted else primary,
        fallback=fallback if primary_budgeted else budget.wrap(fallback),
    ).select("reasoning")
    assert routed.generate("first") == "fallback"
    assert budget.remaining == 1
    primary.response = "primary"

    assert budget.wrap(routed).generate("second") == "primary"
    assert budget.remaining == 0


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


def test_fallback_router_hides_native_tools_when_fallback_lacks_support():
    class NativePrimary(MockLLM):
        def generate_with_tools(self, prompt, *, tools, **kwargs):
            return [{"type": "tool_use", "name": "done", "input": {}}]

    backend = FallbackRouter(
        primary=NativePrimary(),
        fallback=MockLLM("fallback response"),
    ).select("reasoning")

    assert not hasattr(backend, "generate_with_tools")


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
