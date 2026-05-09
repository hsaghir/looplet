"""Cost & token-usage tracking for looplet runs.

Pi exposes per-session token + cost in its footer; looplet hasn't until
now. This module adds an opt-in :class:`CostTracker` plus a
:class:`CostHook` that listens for ``POST_LLM_RESPONSE`` lifecycle
events and aggregates usage from whatever the backend returns.

The tracker is **provider-shape agnostic**: it inspects the raw
response object for the common attribute / dict shapes used by the
OpenAI and Anthropic Python SDKs. If the backend doesn't surface
usage, the tracker silently records zero — it never raises from a
hook.

Usage::

    from looplet import composable_loop, LoopConfig
    from looplet.cost import CostHook, CostTracker, MODEL_PRICES

    tracker = CostTracker(model="claude-sonnet-4-5", prices=MODEL_PRICES)
    hooks = [CostHook(tracker)]

    for step in composable_loop(llm=..., hooks=hooks, ...):
        print(step.pretty(), f"  cum=${tracker.total_cost:.4f}")

    print(tracker.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from looplet.events import EventPayload, LifecycleEvent

__all__ = [
    "CostTracker",
    "CostHook",
    "ModelPrice",
    "MODEL_PRICES",
    "extract_usage",
]


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model.

    Numbers are USD per 1M tokens. Set ``cache_read`` to None when the
    provider doesn't expose cached-input pricing.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


# Conservative defaults for common 2025-vintage models. Update as needed
# in user code; looplet does not auto-fetch pricing.
#
# Both dot and dash spellings are registered because providers
# disagree: Anthropic's API uses dashes (``claude-sonnet-4-5``) while
# the Copilot proxy and several routers use dots
# (``claude-sonnet-4.5``). Same money either way.
_anthropic_prices = {
    "claude-sonnet-4-5": ModelPrice(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75),
    "claude-opus-4-5": ModelPrice(input=15.0, output=75.0, cache_read=1.50, cache_write=18.75),
    "claude-haiku-4-5": ModelPrice(input=1.0, output=5.0, cache_read=0.10, cache_write=1.25),
}
MODEL_PRICES: dict[str, ModelPrice] = {
    **_anthropic_prices,
    # Dot-spelled aliases for proxies / routers that use them.
    **{k.replace("-4-5", "-4.5"): v for k, v in _anthropic_prices.items()},
    # Future Anthropic pricing tier seen on some proxies.
    "claude-opus-4.7": ModelPrice(input=15.0, output=75.0, cache_read=1.50, cache_write=18.75),
    "claude-opus-4-7": ModelPrice(input=15.0, output=75.0, cache_read=1.50, cache_write=18.75),
    # OpenAI
    "gpt-4o": ModelPrice(input=2.50, output=10.0, cache_read=1.25),
    "gpt-4o-mini": ModelPrice(input=0.15, output=0.60, cache_read=0.075),
    "gpt-5": ModelPrice(input=5.0, output=20.0, cache_read=2.50),
    "gpt-4.1": ModelPrice(input=3.0, output=12.0, cache_read=0.75),
}


def extract_usage(raw_response: Any) -> dict[str, int]:
    """Pull a normalised ``{input, output, cache_read, cache_write}`` dict.

    Inspects common shapes:

    * Anthropic: ``response.usage.input_tokens`` /
      ``output_tokens`` / ``cache_read_input_tokens`` /
      ``cache_creation_input_tokens``.
    * OpenAI: ``response.usage.prompt_tokens`` / ``completion_tokens`` /
      ``prompt_tokens_details.cached_tokens``.
    * Plain dict: same keys at top level or under ``usage``.

    Returns zeros for fields not found. Never raises.
    """
    out = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    if raw_response is None:
        return out

    usage = getattr(raw_response, "usage", None)
    if usage is None and isinstance(raw_response, dict):
        usage = raw_response.get("usage", raw_response)

    if usage is None:
        return out

    def _g(obj: Any, key: str) -> int:
        if isinstance(obj, dict):
            v = obj.get(key, 0)
        else:
            v = getattr(obj, key, 0)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    # Anthropic shape
    out["input"] = _g(usage, "input_tokens") or _g(usage, "prompt_tokens")
    out["output"] = _g(usage, "output_tokens") or _g(usage, "completion_tokens")
    out["cache_read"] = _g(usage, "cache_read_input_tokens")
    out["cache_write"] = _g(usage, "cache_creation_input_tokens")

    # OpenAI cached-token nested shape
    if not out["cache_read"]:
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        if details is not None:
            out["cache_read"] = _g(details, "cached_tokens")

    return out


@dataclass
class CostTracker:
    """Aggregates token usage and cost across one agent run.

    Args:
        model: Model name used as the key into ``prices``. If the model
            is missing from ``prices``, cost is reported as 0 but token
            counts are still aggregated.
        prices: Mapping of model name → :class:`ModelPrice`. Pass
            :data:`MODEL_PRICES` for built-in defaults, or your own
            dict for a custom router setup.
    """

    model: str = ""
    prices: dict[str, ModelPrice] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0
    per_call: list[dict[str, int]] = field(default_factory=list)

    # Char-based fallback stats — populated even when the provider
    # omits usage. ``CostHook`` feeds these from the prompt+response
    # lengths it sees. Useful as a coarse cost proxy when running
    # behind proxies (Copilot, OpenRouter free tier, ...) that strip
    # the ``usage`` block.
    prompt_chars: int = 0
    response_chars: int = 0
    chars_per_token: float = 4.0  # conservative industry heuristic

    def record(self, usage: dict[str, int]) -> None:
        """Add one LLM-call usage dict to the running totals."""
        self.calls += 1
        self.input_tokens += usage.get("input", 0)
        self.output_tokens += usage.get("output", 0)
        self.cache_read_tokens += usage.get("cache_read", 0)
        self.cache_write_tokens += usage.get("cache_write", 0)
        self.per_call.append(dict(usage))

    def record_chars(self, *, prompt_chars: int, response_chars: int) -> None:
        """Add one LLM-call's prompt/response sizes (provider-agnostic).

        Safe to call even when ``record(usage)`` already ran for the
        same call — the two counters live independently. Typical
        callsite: a thin wrapper around the backend that knows both
        the input and output strings.
        """
        self.prompt_chars += max(0, int(prompt_chars))
        self.response_chars += max(0, int(response_chars))

    @property
    def has_token_data(self) -> bool:
        """True if the provider supplied any token counts."""
        return any(
            (self.input_tokens, self.output_tokens, self.cache_read_tokens, self.cache_write_tokens)
        )

    @property
    def estimated_input_tokens(self) -> int:
        """Char-based estimate; only meaningful when ``has_token_data`` is False."""
        return int(self.prompt_chars / self.chars_per_token)

    @property
    def estimated_output_tokens(self) -> int:
        return int(self.response_chars / self.chars_per_token)

    @property
    def estimated_cost(self) -> float:
        """USD estimate from prompt/response chars, when token data is missing.

        Uses the model's ``input``/``output`` rates (no cache discount —
        char-based estimates can't tell cached and fresh tokens apart).
        Returns 0.0 when the model is not in the prices table.
        """
        price = self.prices.get(self.model)
        if price is None:
            return 0.0
        cost = (self.estimated_input_tokens / 1e6) * price.input
        cost += (self.estimated_output_tokens / 1e6) * price.output
        return round(cost, 6)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def total_cost(self) -> float:
        """USD spend so far, computed from prices[model]. Zero if unknown."""
        price = self.prices.get(self.model)
        if price is None:
            return 0.0
        # Cached reads are billed at cache_read; the remaining input is at full input rate.
        # For providers that do NOT separately bill cache reads (cache_read=None),
        # treat them as regular input.
        regular_input = self.input_tokens
        cache_read = self.cache_read_tokens
        cache_write = self.cache_write_tokens

        cost = (regular_input / 1e6) * price.input
        cost += (self.output_tokens / 1e6) * price.output
        if price.cache_read is not None:
            cost += (cache_read / 1e6) * price.cache_read
        else:
            cost += (cache_read / 1e6) * price.input
        if price.cache_write is not None:
            cost += (cache_write / 1e6) * price.cache_write
        return round(cost, 6)

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens served from cache; 0.0 when no input."""
        denom = self.input_tokens + self.cache_read_tokens
        if denom == 0:
            return 0.0
        return round(self.cache_read_tokens / denom, 4)

    def summary(self) -> str:
        """One-line human summary.

        When the provider returned token usage, shows real numbers and
        ``cost=$X``. When usage is missing (e.g. Copilot proxy),
        falls back to char-based stats and labels the cost ``(est)``
        so it's never confused with a billable figure.
        """
        if self.has_token_data:
            return (
                f"calls={self.calls} "
                f"in={self.input_tokens} out={self.output_tokens} "
                f"cache_read={self.cache_read_tokens} cache_write={self.cache_write_tokens} "
                f"hit_ratio={self.cache_hit_ratio:.1%} cost=${self.total_cost:.4f}"
            )
        # Fallback: chars + estimates, marked as such.
        return (
            f"calls={self.calls} "
            f"prompt_chars={self.prompt_chars:,} response_chars={self.response_chars:,} "
            f"~in={self.estimated_input_tokens:,} ~out={self.estimated_output_tokens:,} "
            f"cost~=${self.estimated_cost:.4f} (est, no usage from provider)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
            "total_cost_usd": self.total_cost,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost_usd": self.estimated_cost,
            "has_token_data": self.has_token_data,
        }


@dataclass
class CostHook:
    """Hook that feeds POST_LLM_RESPONSE events into a :class:`CostTracker`.

    Install via the ``hooks=`` list on :func:`composable_loop`. The
    tracker is shared and can be inspected after the run completes,
    or polled mid-run for live cost display.

    Args:
        tracker: The :class:`CostTracker` to feed.
        backend: Optional backend reference. When the loop's
            ``raw_response`` is a plain string (the default — the loop
            hands hooks the parsed text, not the API response object),
            :class:`CostHook` falls back to ``backend.last_usage``.
            Both shipped backends (:class:`looplet.backends.OpenAIBackend`,
            :class:`looplet.backends.AnthropicBackend`) populate this
            attribute after every call.
    """

    tracker: CostTracker
    backend: Any = None

    def on_event(self, payload: EventPayload) -> None:
        if payload.event != LifecycleEvent.POST_LLM_RESPONSE:
            return
        usage = extract_usage(payload.raw_response)
        if not any(usage.values()) and self.backend is not None:
            backend_usage = getattr(self.backend, "last_usage", None)
            if backend_usage:
                usage = dict(backend_usage)
        self.tracker.record(usage)
        # Always record char counts too — cheap, and the only stat we
        # have when the provider strips ``usage`` from responses.
        prompt_text = payload.prompt or ""
        response_text = payload.raw_response if isinstance(payload.raw_response, str) else ""
        self.tracker.record_chars(
            prompt_chars=len(prompt_text),
            response_chars=len(response_text),
        )
