"""Behavior checks for launch-facing ASCII output branches."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from looplet import BaseToolRegistry, DefaultState, LoopConfig, composable_loop
from looplet.cartridge.portability import _read_yaml_file
from looplet.cli import spec_commands
from looplet.evals import eval_cli
from looplet.recovery import FailureScenario, RecoveryAction, RecoveryRecipe, RecoveryRegistry
from looplet.tools import ToolSpec


def _done_tools() -> BaseToolRegistry:
    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="done",
            description="Finish.",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return tools


def test_conformance_failure_output_uses_ascii_separator(tmp_path, monkeypatch, capsys):
    from looplet import cartridge

    for name in ("missing", "broken", "mismatch"):
        fixture = tmp_path / name
        fixture.mkdir()
        (fixture / "expected.json").write_text(json.dumps({"expected": True}))
        if name != "missing":
            (fixture / "cartridge").mkdir()

    def load(path, *, strict):
        assert strict is True
        if "broken" in path:
            raise ValueError("broken manifest")
        return object()

    monkeypatch.setattr(cartridge, "cartridge_to_preset", load)
    monkeypatch.setattr(spec_commands, "_summarise_preset", lambda _preset: {"actual": True})

    result = spec_commands.cmd_conform(SimpleNamespace(fixtures=tmp_path, verbose=False))
    output = capsys.readouterr().out

    assert result == 1
    assert "conformance - 3 fixture(s)" in output
    assert "FAIL missing - missing cartridge/" in output
    assert "FAIL broken - load error: ValueError: broken manifest" in output
    assert "FAIL mismatch - summary mismatch" in output


def test_portability_ignores_malformed_yaml(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("[unterminated")

    assert _read_yaml_file(config) == {}


def test_verbose_eval_cli_uses_ascii_missing_score_marker(tmp_path, capsys):
    run = tmp_path / "runs" / "one"
    run.mkdir(parents=True)
    (run / "trajectory.json").write_text(
        json.dumps({"steps": [], "task": {}, "termination_reason": "done"})
    )
    evaluator = tmp_path / "eval_label.py"
    evaluator.write_text("def eval_label(ctx):\n    return 'pass'\n")

    assert eval_cli([str(run.parent), "--evals", str(evaluator), "--verbose"]) == 0
    assert "one:  - " in capsys.readouterr().out


def test_briefing_budget_adds_ascii_truncation_marker():
    class Backend:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt, **_kwargs):
            self.prompt = prompt
            return '{"tool":"done","args":{"answer":"ok"}}'

    class Briefing:
        def pre_prompt(self, *_args):
            return "x" * 100

    backend = Backend()
    list(
        composable_loop(
            llm=backend,
            tools=_done_tools(),
            hooks=[Briefing()],
            config=LoopConfig(max_steps=1, max_briefing_tokens=1),
            state=DefaultState(max_steps=1),
            task={},
        )
    )

    assert "briefing truncated - token budget exceeded" in backend.prompt


def test_preflight_warning_uses_ascii_separator(caplog):
    class Backend:
        def generate(self, *_args, **_kwargs):
            raise AssertionError("preflight should skip the backend")

    with caplog.at_level(logging.WARNING):
        list(
            composable_loop(
                llm=Backend(),
                tools=_done_tools(),
                config=LoopConfig(max_steps=1, context_window=1, reactive_recovery=True),
                state=DefaultState(max_steps=1),
                task={},
            )
        )

    assert "exceeds safe limit - running recovery" in caplog.text


def test_aborted_parse_recovery_uses_ascii_error():
    registry = RecoveryRegistry()
    registry.register(
        RecoveryRecipe(
            FailureScenario.PARSE_ERROR,
            lambda _context: RecoveryAction("abort", message="stop now"),
        )
    )

    class Backend:
        def generate(self, *_args, **_kwargs):
            return "not json"

    steps = list(
        composable_loop(
            llm=Backend(),
            tools=_done_tools(),
            config=LoopConfig(max_steps=1, recovery_registry=registry),
            state=DefaultState(max_steps=1),
            task={},
        )
    )

    assert steps[0].tool_result.error == "Parse error - recovery aborted: stop now"
