"""Smoke tests for prompt caching primitives."""
from __future__ import annotations

import pytest

from openharness import (
    BaseToolRegistry,
    CachePolicy,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from openharness.cache import CacheBreakDetector, CacheBreakpoint, CacheControl, compute_breakpoints
from openharness.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools():
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="done", description="finish",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


class _CacheAwareBackend:
    """Records every ``cache_breakpoints`` kwarg it receives."""
    def __init__(self, responses):
        self._r = list(responses)
        self.received: list[list[CacheBreakpoint] | None] = []

    def generate(
        self, prompt, *, max_tokens=2000, system_prompt="",
        temperature=0.2, cache_breakpoints=None,
    ):
        self.received.append(cache_breakpoints)
        return self._r.pop(0)


class TestCachePolicy:
    def test_compute_breakpoints_order_and_hashes_stable(self):
        pol = CachePolicy(
            system_prompt=CacheControl(),
            tool_schemas=CacheControl(ttl="1h"),
            memory=CacheControl(),
        )
        a = compute_breakpoints(pol, system_prompt="S", tool_schemas_text="T", memory_text="M")
        b = compute_breakpoints(pol, system_prompt="S", tool_schemas_text="T", memory_text="M")
        assert [bp.label for bp in a] == ["system_prompt", "tool_schemas", "memory"]
        assert [bp.hash for bp in a] == [bp.hash for bp in b]
        assert a[1].control.ttl == "1h"

    def test_different_content_different_hash(self):
        pol = CachePolicy(system_prompt=CacheControl())
        a = compute_breakpoints(pol, system_prompt="A", tool_schemas_text="", memory_text="")
        b = compute_breakpoints(pol, system_prompt="B", tool_schemas_text="", memory_text="")
        assert a[0].hash != b[0].hash

    def test_empty_policy_no_breakpoints(self):
        pol = CachePolicy()
        assert compute_breakpoints(pol, system_prompt="x", tool_schemas_text="x", memory_text="x") == []


class TestCacheBreakDetector:
    def test_no_breaks_on_stable_content(self):
        pol = CachePolicy(system_prompt=CacheControl())
        d = CacheBreakDetector(pol)
        d.record(0, system_prompt="S", tool_schemas_text="", memory_text="")
        d.record(1, system_prompt="S", tool_schemas_text="", memory_text="")
        d.record(2, system_prompt="S", tool_schemas_text="", memory_text="")
        assert d.breaks == []

    def test_records_break_on_change(self):
        pol = CachePolicy(system_prompt=CacheControl())
        d = CacheBreakDetector(pol)
        d.record(0, system_prompt="S", tool_schemas_text="", memory_text="")
        d.record(1, system_prompt="S2", tool_schemas_text="", memory_text="")
        assert len(d.breaks) == 1
        step, section, _, _ = d.breaks[0]
        assert step == 1 and section == "system_prompt"


class TestLoopIntegration:
    def test_policy_threads_breakpoints_into_backend(self):
        b = _CacheAwareBackend(['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}'])
        pol = CachePolicy(system_prompt=CacheControl(), tool_schemas=CacheControl())
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[], config=LoopConfig(max_steps=2, system_prompt="SYS", cache_policy=pol),
        ))
        assert len(b.received) == 1
        bps = b.received[0]
        assert bps is not None
        labels = [bp.label for bp in bps]
        assert "system_prompt" in labels
        assert "tool_schemas" in labels

    def test_no_policy_no_breakpoints(self):
        b = _CacheAwareBackend(['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}'])
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[], config=LoopConfig(max_steps=2),
        ))
        assert b.received == [None]

    def test_backend_without_kwarg_still_works(self):
        class _Plain:
            def generate(self, prompt, **kw): return '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}'
        pol = CachePolicy(system_prompt=CacheControl())
        # Should not raise — cache_breakpoints is filtered out when backend
        # doesn't declare it.
        list(composable_loop(
            llm=_Plain(), tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[], config=LoopConfig(max_steps=2, cache_policy=pol),
        ))

    def test_detector_hook_integration(self):
        """CacheBreakDetector is detected by the loop via isinstance, not via hook methods."""
        import warnings

        b = _CacheAwareBackend([
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ])
        pol = CachePolicy(system_prompt=CacheControl())
        det = CacheBreakDetector(pol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            list(composable_loop(
                llm=b, tools=_tools(), state=DefaultState(max_steps=2),
                hooks=[det], config=LoopConfig(max_steps=2, system_prompt="S", cache_policy=pol),
            ))
        # Detector recorded one turn, no breaks (only one turn).
        assert det.breaks == []
