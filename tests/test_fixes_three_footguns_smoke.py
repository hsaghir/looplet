"""Regression tests for the three deep-research-agent footguns.

Covers:
  1. eval_discover must NOT pick up re-exported decorators like
     ``eval_mark`` itself.
  2. ProvenanceSink(redact=...) must redact BEFORE forwarding to the
     wrapped backend (not just before disk write).
  3. EvalContext must expose ``stop_reason`` / ``completed`` so evals
     can dispatch on early hook-stop vs normal done().
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    ProvenanceSink,
    ToolSpec,
    composable_loop,
)
from looplet.evals import EvalContext, eval_discover
from looplet.testing import MockLLMBackend

pytestmark = pytest.mark.smoke


# ── 1. eval_discover ignores imported decorators ────────────────────


def test_eval_discover_skips_reexported_decorator(tmp_path: Path) -> None:
    """``from looplet import eval_mark`` must not be collected as an eval."""
    f = tmp_path / "eval_me.py"
    f.write_text(
        "from looplet import eval_mark\n\n@eval_mark('x')\ndef eval_one(ctx):\n    return True\n"
    )
    evaluators = eval_discover(f)
    names = [e.__name__ for e in evaluators]
    assert names == ["eval_one"], (
        f"expected only eval_one, got {names} (decorator leaked into discovery)"
    )


def test_eval_discover_skips_imported_from_other_module(tmp_path: Path) -> None:
    """Any function whose __module__ isn't the eval file is filtered out."""
    lib = tmp_path / "my_lib.py"
    lib.write_text("def eval_helper(ctx):\n    return 1.0\n")
    f = tmp_path / "eval_discover_test.py"
    f.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from my_lib import eval_helper  # noqa: F401\n"
        "\n"
        "def eval_local(ctx):\n"
        "    return True\n"
    )
    evaluators = eval_discover(f)
    assert [e.__name__ for e in evaluators] == ["eval_local"]


# ── 2. ProvenanceSink redacts upstream by default ───────────────────


class _EchoBackend:
    """Minimal backend that records exactly what prompt it was given."""

    def __init__(self) -> None:
        self.last_prompt: str = ""
        self.last_system: str = ""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        return '{"tool": "done", "args": {"answer": "ok"}, "reasoning": "done"}'


def _redact(s: str) -> str:
    return s.replace("secret@example.com", "[EMAIL]").replace("SECRET", "[REDACTED]")


def test_redact_scrubs_upstream_by_default(tmp_path: Path) -> None:
    """``redact=`` should strip PII BEFORE the prompt reaches the backend."""
    inner = _EchoBackend()
    sink = ProvenanceSink(dir=tmp_path, redact=_redact)
    llm = sink.wrap_llm(inner)

    llm.generate("Please email secret@example.com with token SECRET.")

    assert "secret@example.com" not in inner.last_prompt
    assert "SECRET" not in inner.last_prompt
    assert "[EMAIL]" in inner.last_prompt
    assert "[REDACTED]" in inner.last_prompt

    # And on disk.
    sink.flush()
    prompt_file = next(tmp_path.glob("call_*_prompt.txt"))
    disk = prompt_file.read_text()
    assert "secret@example.com" not in disk
    assert "SECRET" not in disk


def test_redact_upstream_can_be_disabled(tmp_path: Path) -> None:
    """Opt-out: record-only redaction for the legacy case."""
    inner = _EchoBackend()
    sink = ProvenanceSink(dir=tmp_path, redact=_redact, redact_upstream=False)
    llm = sink.wrap_llm(inner)

    llm.generate("token is SECRET")

    # Provider saw raw prompt.
    assert "SECRET" in inner.last_prompt
    # Disk still redacted.
    sink.flush()
    disk = next(tmp_path.glob("call_*_prompt.txt")).read_text()
    assert "SECRET" not in disk


# ── 3. EvalContext.stop_reason exposed for early-stop evals ─────────


def test_eval_context_stop_reason_done() -> None:
    """Normal completion sets stop_reason == 'done'."""
    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="done",
            description="finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )

    llm = MockLLMBackend(
        responses=[
            '{"tool": "done", "args": {"answer": "ok"}, "reasoning": "done"}',
        ]
    )
    state = DefaultState(max_steps=5)
    for _ in composable_loop(
        llm=llm,
        tools=tools,
        state=state,
        config=LoopConfig(max_steps=5),
        task={"goal": "x"},
    ):
        pass
    assert getattr(state, "_stop_reason", None) == "done"


def test_eval_context_stop_reason_hook_stop() -> None:
    """Hook-triggered termination sets stop_reason to the hook's label."""

    class StopAfterOne:
        def should_stop(self, state, step_num, new_entities):
            return True

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="noop", description="noop", parameters={}, execute=lambda: {"ok": True})
    )
    tools.register(
        ToolSpec(
            name="done",
            description="finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )

    llm = MockLLMBackend(
        responses=[
            '{"tool": "noop", "args": {}, "reasoning": "step 1"}',
            '{"tool": "done", "args": {"answer": "never"}, "reasoning": "never reached"}',
        ]
    )
    state = DefaultState(max_steps=5)
    for _ in composable_loop(
        llm=llm,
        tools=tools,
        state=state,
        config=LoopConfig(max_steps=5),
        task={"goal": "x"},
        hooks=[StopAfterOne()],
    ):
        pass
    assert getattr(state, "_stop_reason", None) != "done"
    # And EvalContext exposes it.
    ctx = EvalContext(
        steps=list(state.steps),
        stop_reason=getattr(state, "_stop_reason", None),
    )
    assert ctx.completed is False
    assert ctx.stop_reason in {"hook_stop", None} or isinstance(ctx.stop_reason, str)


def test_eval_context_completed_property() -> None:
    ctx = EvalContext(steps=[], stop_reason="done")
    assert ctx.completed is True
    ctx2 = EvalContext(steps=[], stop_reason="hook_stop")
    assert ctx2.completed is False
    ctx3 = EvalContext(steps=[])
    assert ctx3.completed is False
    assert ctx3.stop_reason is None
