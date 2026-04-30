"""Tests for the additive ``tool_call`` kwarg on ``check_done``.

The loop dispatches ``check_done`` with or without the ``tool_call``
kwarg depending on the hook's signature, so existing legacy hooks
(``check_done(self, state, session_log, context, step_num)``) keep
working unchanged while new quality gates can inspect the candidate
``done()`` payload.
"""

from __future__ import annotations

from typing import Any

from looplet import (
    BaseToolRegistry,
    Block,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import ToolCall


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="lookup",
            description="lookup",
            parameters={"key": "str"},
            execute=lambda *, key: {"key": key, "value": {"x": 1}.get(key, "MISSING")},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="done",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


def _run(responses: list[str], hooks: list[Any]) -> list[Any]:
    return list(
        composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=_tools(),
            state=DefaultState(max_steps=8),
            hooks=hooks,
            config=LoopConfig(max_steps=8),
        )
    )


def test_legacy_check_done_signature_still_works() -> None:
    """A hook with the original 4-arg signature must keep working."""
    seen: list[int] = []

    class Legacy:
        def check_done(self, state, session_log, context, step_num):
            seen.append(step_num)
            return None

    steps = _run(
        ['{"tool":"done","args":{"answer":"x"},"reasoning":""}'],
        [Legacy()],
    )

    assert len(steps) == 1
    assert steps[0].tool_call.tool == "done"
    assert seen, "check_done was not called"


def test_check_done_receives_candidate_tool_call() -> None:
    """A hook that accepts ``tool_call`` sees the candidate done() call."""
    captured: list[ToolCall] = []

    class Inspector:
        def check_done(self, state, session_log, context, step_num, tool_call):
            captured.append(tool_call)
            return None

    _run(
        ['{"tool":"done","args":{"answer":"hello"},"reasoning":""}'],
        [Inspector()],
    )

    assert len(captured) == 1
    assert captured[0].tool == "done"
    assert captured[0].args == {"answer": "hello"}


def test_check_done_can_block_unsupported_answer() -> None:
    """A done-gate using ``tool_call`` can reject a fabricated claim
    and the loop should re-prompt the agent (which then submits a
    grounded answer)."""

    class GroundedGate:
        def __init__(self) -> None:
            self.observed_facts: set[str] = set()

        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            if isinstance(tool_result.data, dict):
                for k, v in tool_result.data.items():
                    self.observed_facts.add(f"{k}={v}")
            return None

        def check_done(self, state, session_log, context, step_num, tool_call):
            answer = str(tool_call.args.get("answer", ""))
            for fragment in answer.split(" and "):
                f = fragment.strip()
                if "=" in f and f not in self.observed_facts:
                    return Block(
                        f"unsupported claim: {f!r} not in observed facts {sorted(self.observed_facts)}"
                    )
            return None

    gate = GroundedGate()
    steps = _run(
        [
            '{"tool":"lookup","args":{"key":"x"},"reasoning":""}',
            # Agent fabricates y=99 — should be blocked.
            '{"tool":"done","args":{"answer":"key=x and value=1 and y=99"},"reasoning":""}',
            # On retry, agent submits a grounded answer.
            '{"tool":"done","args":{"answer":"key=x and value=1"},"reasoning":""}',
        ],
        [gate],
    )

    final_done_steps = [s for s in steps if s.tool_call.tool == "done"]
    assert len(final_done_steps) >= 2
    rejected = final_done_steps[0]
    assert rejected.tool_result.data.get("rejected") is True
    accepted = final_done_steps[-1]
    assert accepted.tool_result.data.get("rejected") is not True
    assert accepted.tool_call.args["answer"] == "key=x and value=1"


def test_check_done_with_var_kwargs_signature() -> None:
    """A hook using ``**kwargs`` should also receive ``tool_call``."""
    captured: list[Any] = []

    class Vararg:
        def check_done(self, state, session_log, context, step_num, **kwargs):
            captured.append(kwargs.get("tool_call"))
            return None

    _run(
        ['{"tool":"done","args":{"answer":"ok"},"reasoning":""}'],
        [Vararg()],
    )

    assert len(captured) == 1
    assert captured[0] is not None
    assert captured[0].tool == "done"


def test_signature_cache_keys_on_func_not_bound_method_id() -> None:
    """Regression: the ``_accepts_tool_call_kwarg`` cache must key on the
    underlying function, not the bound-method object's ``id()``. Bound
    methods are ephemeral in CPython and their ids get reused for
    unrelated methods, which used to poison the cache and call hooks
    that don't accept ``tool_call`` with the kwarg anyway.
    """
    from looplet.loop import _CHECK_DONE_ACCEPTS_TOOL_CALL, _accepts_tool_call_kwarg

    class WithKwarg:
        def check_done(self, state, session_log, context, step_num, tool_call=None):
            return None

    class WithoutKwarg:
        def check_done(self, state, session_log, context, step_num):
            return None

    a = WithKwarg()
    b = WithoutKwarg()

    # Each class's check_done caches independently and produces the
    # correct answer regardless of how many bound-method objects come
    # and go in between.
    for _ in range(50):
        assert _accepts_tool_call_kwarg(a.check_done) is True
        assert _accepts_tool_call_kwarg(b.check_done) is False

    # Two distinct cache entries (one per class), not one per
    # short-lived bound-method object.
    func_keys = {id(WithKwarg.check_done), id(WithoutKwarg.check_done)}
    assert func_keys.issubset(_CHECK_DONE_ACCEPTS_TOOL_CALL.keys())
