"""Shared, inspectable defaults for Python and cartridge agent runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from looplet.budget import ContextBudget, ThresholdCompactHook
from looplet.compact import DefaultCompactService
from looplet.limits import ToolCallBudgetHook
from looplet.stagnation import StagnationHook

if TYPE_CHECKING:
    from looplet.loop import LoopConfig

__all__ = ["ResolvedHarness", "resolve_harness"]


@dataclass(frozen=True)
class ResolvedHarness:
    """Effective per-run configuration and hooks, without starting a run."""

    config: LoopConfig
    hooks: list[Any]

    def describe(self) -> dict[str, Any]:
        """Return a JSON-compatible explanation of the effective harness."""
        service = self.config.compact_service
        return {
            "profile": self.config.harness_profile,
            "disabled_defaults": list(self.config.disabled_defaults),
            "context_window": self.config.context_window,
            "max_steps": self.config.max_steps,
            "max_tokens": self.config.max_tokens,
            "max_turn_continuations": self.config.max_turn_continuations,
            "typed_history": self.config.uses_typed_history,
            "max_model_calls": self.config.max_model_calls,
            "max_tool_calls": next(
                (hook.max_calls for hook in self.hooks if isinstance(hook, ToolCallBudgetHook)),
                None,
            ),
            "compact_service": type(service).__name__ if service is not None else None,
            "hooks": [type(hook).__name__ for hook in self.hooks],
        }


def resolve_harness(
    config: LoopConfig | None = None,
    hooks: list[Any] | None = None,
) -> ResolvedHarness:
    """Compose standard defaults underneath explicit host configuration.

    ``execution-v1`` installs deterministic compaction without steering or
    derived tool quotas. ``standard-v1`` retains a non-blocking
    stagnation nudge, and a per-invocation tool-dispatch budget. ``minimal``
    keeps only explicitly supplied policies.
    Config and hook containers are not mutated; live host objects retain
    their identity. Resolution does not call a model or access the filesystem.
    """
    from looplet.loop import LoopConfig

    effective = replace(config) if config is not None else LoopConfig()
    resolved_hooks = list(hooks or ())
    if effective.harness_profile not in {"execution-v1", "standard-v1", "minimal"}:
        raise ValueError("harness_profile must be 'execution-v1', 'standard-v1', or 'minimal'")
    disabled = set(effective.disabled_defaults)
    unknown = disabled - {"compaction", "stagnation", "tool_budget"}
    if unknown:
        raise ValueError(f"unknown disabled_defaults: {sorted(unknown)}")
    if effective.harness_profile == "minimal":
        return ResolvedHarness(effective, resolved_hooks)

    if "compaction" not in disabled:
        if effective.compact_service is None:
            effective.compact_service = DefaultCompactService(use_llm_summary=False)
        if effective.context_window >= 4 and not any(
            callable(getattr(hook, "should_compact", None)) for hook in resolved_hooks
        ):
            window = effective.context_window
            buffer = min(13_000, max(0, (window - 1) // 10))
            budget = ContextBudget(
                context_window=window,
                warning_at=int(window * 0.6),
                error_at=int(window * 0.8),
                compact_buffer=buffer,
            )
            resolved_hooks.append(
                ThresholdCompactHook(
                    budget,
                    fire_tier="warning" if effective.harness_profile == "execution-v1" else "error",
                )
            )

    if effective.harness_profile == "execution-v1":
        return ResolvedHarness(effective, resolved_hooks)

    if "stagnation" not in disabled and not any(
        isinstance(hook, StagnationHook) for hook in resolved_hooks
    ):
        resolved_hooks.append(
            StagnationHook(
                fingerprint="result",
                threshold=6,
                ignore_tools={effective.done_tool, *effective.done_tools, "think"},
            )
        )
    if "tool_budget" not in disabled and not any(
        isinstance(hook, ToolCallBudgetHook) for hook in resolved_hooks
    ):
        resolved_hooks.append(ToolCallBudgetHook(max_calls=max(16, effective.max_steps * 4)))
    return ResolvedHarness(effective, resolved_hooks)
