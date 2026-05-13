"""Tests for looplet.cost — token + cost tracking."""

from __future__ import annotations

from looplet.cost import (
    MODEL_PRICES,
    CostHook,
    CostTracker,
    ModelPrice,
    extract_usage,
)
from looplet.events import EventPayload, LifecycleEvent


class _AnthropicUsage:
    def __init__(self) -> None:
        self.input_tokens = 1000
        self.output_tokens = 500
        self.cache_read_input_tokens = 200
        self.cache_creation_input_tokens = 50


class _AnthropicResponse:
    def __init__(self) -> None:
        self.usage = _AnthropicUsage()


def test_extract_usage_anthropic_attrs() -> None:
    u = extract_usage(_AnthropicResponse())
    assert u == {"input": 1000, "output": 500, "cache_read": 200, "cache_write": 50}


def test_extract_usage_openai_dict() -> None:
    resp = {
        "usage": {
            "prompt_tokens": 800,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 600},
        }
    }
    u = extract_usage(resp)
    assert u == {"input": 800, "output": 200, "cache_read": 600, "cache_write": 0}


def test_extract_usage_none_safe() -> None:
    assert extract_usage(None) == {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def test_extract_usage_unknown_shape_returns_zeros() -> None:
    assert extract_usage("just a string")["input"] == 0


def test_tracker_aggregates() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    t.record({"input": 1000, "output": 200, "cache_read": 0, "cache_write": 0})
    t.record({"input": 500, "output": 100, "cache_read": 800, "cache_write": 0})
    assert t.calls == 2
    assert t.input_tokens == 1500
    assert t.output_tokens == 300
    assert t.cache_read_tokens == 800
    assert t.total_tokens == 1500 + 300 + 800
    assert t.cache_hit_ratio > 0.0
    assert t.total_cost > 0.0


def test_tracker_unknown_model_zero_cost() -> None:
    t = CostTracker(model="mystery-model", prices=MODEL_PRICES)
    t.record({"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0})
    assert t.total_cost == 0.0
    assert t.input_tokens == 1000  # tokens still aggregated


def test_tracker_pricing_math() -> None:
    # Custom price: $1/M in, $2/M out → 1M input + 1M output = $3
    t = CostTracker(
        model="x",
        prices={"x": ModelPrice(input=1.0, output=2.0)},
    )
    t.record({"input": 1_000_000, "output": 1_000_000, "cache_read": 0, "cache_write": 0})
    assert t.total_cost == 3.0


def test_cost_hook_listens_to_post_llm_response() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)

    # Wrong event → no-op
    hook.on_event(
        EventPayload(event=LifecycleEvent.PRE_LLM_CALL, raw_response=_AnthropicResponse())
    )
    assert t.calls == 0

    # Right event → records
    hook.on_event(
        EventPayload(event=LifecycleEvent.POST_LLM_RESPONSE, raw_response=_AnthropicResponse())
    )
    assert t.calls == 1
    assert t.input_tokens == 1000


def test_summary_string() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    t.record({"input": 100, "output": 50, "cache_read": 0, "cache_write": 0})
    s = t.summary()
    assert "calls=1" in s
    assert "in=100" in s
    assert "$" in s


def test_to_dict_serializable() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    t.record({"input": 100, "output": 50, "cache_read": 10, "cache_write": 0})
    d = t.to_dict()
    assert d["calls"] == 1 and d["input_tokens"] == 100 and "total_cost_usd" in d


def test_cost_hook_falls_back_to_backend_last_usage() -> None:
    # When the loop hands the hook a plain string (the default), the
    # CostHook should fall back to backend.last_usage.
    class _FakeBackend:
        last_usage = {"input": 50, "output": 25, "cache_read": 0, "cache_write": 0}

    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t, backend=_FakeBackend())
    hook.on_event(EventPayload(event=LifecycleEvent.POST_LLM_RESPONSE, raw_response="just text"))
    assert t.calls == 1 and t.input_tokens == 50 and t.output_tokens == 25


def test_cost_hook_prefers_raw_response_when_present() -> None:
    # If both raw_response carries usage AND backend has last_usage,
    # raw_response wins (it represents the most recent call).
    class _FakeBackend:
        last_usage = {"input": 999, "output": 999, "cache_read": 0, "cache_write": 0}

    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t, backend=_FakeBackend())
    hook.on_event(
        EventPayload(event=LifecycleEvent.POST_LLM_RESPONSE, raw_response=_AnthropicResponse())
    )
    assert t.input_tokens == 1000  # from raw_response, not 999


def test_char_fallback_when_usage_missing() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)
    hook.on_event(
        EventPayload(
            event=LifecycleEvent.POST_LLM_RESPONSE,
            prompt="x" * 4000,
            raw_response="y" * 1200,  # plain string -> no usage shape
        )
    )
    # No real token data
    assert not t.has_token_data
    # But chars were recorded
    assert t.prompt_chars == 4000 and t.response_chars == 1200
    # Estimates: 4000/4=1000 in, 1200/4=300 out
    assert t.estimated_input_tokens == 1000
    assert t.estimated_output_tokens == 300
    # Estimated cost > 0 because the model is in the price table
    assert t.estimated_cost > 0
    # Summary shows the (est) marker
    s = t.summary()
    assert "(est" in s and "prompt_chars=4,000" in s


def test_summary_real_usage_format() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    t.record({"input": 10, "output": 5, "cache_read": 0, "cache_write": 0})
    s = t.summary()
    assert "(est" not in s
    assert "in=10" in s and "cost=$" in s


def test_to_dict_includes_estimates() -> None:
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    t.record_chars(prompt_chars=400, response_chars=200)
    t.calls = 1
    d = t.to_dict()
    assert d["prompt_chars"] == 400 and d["response_chars"] == 200
    assert d["estimated_input_tokens"] == 100
    assert d["estimated_output_tokens"] == 50
    assert d["has_token_data"] is False


# ---------------------------------------------------------------------------
# Native-tool path: raw_response is a list of content blocks, not a string.
# Without coercion, char-based fallback under-reports response_chars to 0
# whenever native tools are used (the default since spec-v2).
# ---------------------------------------------------------------------------
def test_char_fallback_with_native_tool_block_list_dict_shape() -> None:
    """Anthropic-style serialized blocks: list of dicts with type/text/name/input."""
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)
    blocks = [
        {"type": "text", "text": "I'll search for that."},
        {
            "type": "tool_use",
            "name": "search",
            "input": {"query": "python decorators", "limit": 10},
        },
    ]
    hook.on_event(
        EventPayload(
            event=LifecycleEvent.POST_LLM_RESPONSE,
            prompt="x" * 1000,
            raw_response=blocks,
        )
    )
    # Response chars must NOT be zero — native-tool blocks have measurable
    # length (text + tool name + JSON-serialized args).
    assert t.response_chars > 0
    expected = (
        len("I'll search for that.")
        + len("search")
        + len('{"query":"python decorators","limit":10}')
    )
    assert t.response_chars == expected


def test_char_fallback_with_native_tool_block_list_object_shape() -> None:
    """Provider SDK objects: blocks expose .text or .name + .input attrs."""

    class _TextBlock:
        type = "text"
        text = "Looking up that file."

    class _ToolUseBlock:
        type = "tool_use"
        name = "read"
        input = {"file_path": "src/main.py"}

    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)
    hook.on_event(
        EventPayload(
            event=LifecycleEvent.POST_LLM_RESPONSE,
            prompt="p" * 500,
            raw_response=[_TextBlock(), _ToolUseBlock()],
        )
    )
    assert t.response_chars > 0
    expected = len("Looking up that file.") + len("read") + len('{"file_path":"src/main.py"}')
    assert t.response_chars == expected


def test_char_fallback_block_list_text_only() -> None:
    """All-text block list (no tool calls) still extracts cleanly."""
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)
    hook.on_event(
        EventPayload(
            event=LifecycleEvent.POST_LLM_RESPONSE,
            prompt="",
            raw_response=[{"type": "text", "text": "hello world"}],
        )
    )
    assert t.response_chars == len("hello world")


def test_char_fallback_block_list_handles_unserializable_input() -> None:
    """tool_use.input that can't be JSON-encoded falls back to str()."""

    class _Weird:
        def __repr__(self) -> str:
            return "<weird>"

    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)
    hook.on_event(
        EventPayload(
            event=LifecycleEvent.POST_LLM_RESPONSE,
            prompt="",
            raw_response=[{"type": "tool_use", "name": "x", "input": _Weird()}],
        )
    )
    # No exception; some chars recorded for name + str(input).
    assert t.response_chars >= len("x")


def test_char_fallback_unknown_shape_returns_zero() -> None:
    """Plain-string path still works; arbitrary objects yield 0 chars (no crash)."""
    t = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hook = CostHook(t)
    hook.on_event(
        EventPayload(
            event=LifecycleEvent.POST_LLM_RESPONSE,
            prompt="prompt",
            raw_response=object(),  # not str, not list
        )
    )
    assert t.prompt_chars == len("prompt")
    assert t.response_chars == 0
