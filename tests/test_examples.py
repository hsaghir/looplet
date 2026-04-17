"""Tests for cadence example agents — verify they run without errors."""
from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stdout


def _run_main(module_path: str) -> str:
    """Import a module and call its run() function, capturing stdout."""
    mod = importlib.import_module(module_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.run()
    return buf.getvalue()


# ── calculator_agent ─────────────────────────────────────────────


def test_calculator_agent_importable():
    import openharness.examples.calculator_agent  # noqa: F401


def test_calculator_agent_run_completes():
    output = _run_main("openharness.examples.calculator_agent")
    assert isinstance(output, str)


def test_calculator_agent_produces_answer():
    output = _run_main("openharness.examples.calculator_agent")
    # Should mention the final answer (25 = 5 * 5, computed via scripted steps)
    assert "answer" in output.lower() or "25" in output or "done" in output.lower()


def test_calculator_agent_shows_steps():
    output = _run_main("openharness.examples.calculator_agent")
    # Should show some step output
    assert len(output) > 0


def test_calculator_agent_has_main_block():
    import openharness.examples.calculator_agent as m
    assert hasattr(m, "run")


# ── research_agent ───────────────────────────────────────────────


def test_research_agent_importable():
    import openharness.examples.research_agent  # noqa: F401


def test_research_agent_run_completes():
    output = _run_main("openharness.examples.research_agent")
    assert isinstance(output, str)


def test_research_agent_shows_output():
    output = _run_main("openharness.examples.research_agent")
    assert len(output) > 0


def test_research_agent_has_main_block():
    import openharness.examples.research_agent as m
    assert hasattr(m, "run")


def test_research_agent_uses_session_log():
    """Verify the example imports and uses SessionLog."""
    import inspect

    import openharness.examples.research_agent as m
    source = inspect.getsource(m)
    assert "SessionLog" in source


def test_research_agent_theory_tracking():
    """Verify the example has theory tracking."""
    import inspect

    import openharness.examples.research_agent as m
    source = inspect.getsource(m)
    assert "theory" in source.lower()


# ── code_review_agent ────────────────────────────────────────────


def test_code_review_agent_importable():
    import openharness.examples.code_review_agent  # noqa: F401


def test_code_review_agent_run_completes():
    output = _run_main("openharness.examples.code_review_agent")
    assert isinstance(output, str)


def test_code_review_agent_shows_output():
    output = _run_main("openharness.examples.code_review_agent")
    assert len(output) > 0


def test_code_review_agent_has_main_block():
    import openharness.examples.code_review_agent as m
    assert hasattr(m, "run")


def test_code_review_agent_uses_streaming():
    """Verify the example uses StreamingHook."""
    import inspect

    import openharness.examples.code_review_agent as m
    source = inspect.getsource(m)
    assert "StreamingHook" in source or "streaming" in source.lower()


def test_code_review_agent_has_quality_gate_hook():
    """Verify the example defines a QualityGateHook."""
    import inspect

    import openharness.examples.code_review_agent as m
    source = inspect.getsource(m)
    assert "QualityGate" in source


def test_code_review_agent_uses_callback_emitter():
    """Verify the example uses CallbackEmitter for event printing."""
    import inspect

    import openharness.examples.code_review_agent as m
    source = inspect.getsource(m)
    assert "CallbackEmitter" in source
