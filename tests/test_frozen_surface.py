"""Frozen public-surface guard (RPC foundation §1.6).

Pins the v1.0 API-contract symbols that the RPC-foundation work MUST NOT
have changed. If any assertion here fails, the change is no longer
additive and the version bump is no longer a minor one.
"""

from __future__ import annotations

import dataclasses
import inspect

from looplet import LLMBackend, LoopHook, Step, composable_loop

# The seven frozen hook method names (ROADMAP "The hook protocol").
FROZEN_HOOK_METHODS = (
    "pre_loop",
    "pre_prompt",
    "pre_dispatch",
    "post_dispatch",
    "check_done",
    "should_stop",
    "on_loop_end",
)


def test_composable_loop_signature_frozen() -> None:
    # The frozen contract is the set of accepted parameter NAMES (all six are
    # valid call kwargs per the README); their positional/keyword kind is not
    # pinned here, only their continued availability.
    params = inspect.signature(composable_loop).parameters
    for name in ("llm", "tools", "task", "state", "config", "hooks"):
        assert name in params, f"composable_loop lost frozen parameter {name!r}"


def test_loophook_method_names_frozen() -> None:
    members = set(dir(LoopHook))
    for name in FROZEN_HOOK_METHODS:
        assert name in members, f"LoopHook lost frozen method {name!r}"


def test_step_first_fields_frozen() -> None:
    assert dataclasses.is_dataclass(Step), "Step must remain a dataclass"
    names = [f.name for f in dataclasses.fields(Step)]
    assert names[:3] == ["number", "tool_call", "tool_result"], (
        f"Step's leading frozen fields changed: {names[:3]}"
    )


def test_llmbackend_generate_frozen() -> None:
    # LLMBackend.generate is the frozen text-generation entry point; it takes a
    # `prompt`. (The tool-calling path is a separate method.)
    params = inspect.signature(LLMBackend.generate).parameters
    assert "prompt" in params, "LLMBackend.generate lost its `prompt` parameter"
