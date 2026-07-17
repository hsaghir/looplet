"""Cross-runtime trajectory conformance.

Both runtimes - looplet's ``composable_loop`` and the from-scratch
``tinyloop`` - receive the same scripted sequence of tool calls and
must produce identical step traces (modulo timing and call_id
formatting). This is the strongest portability claim in SPEC.md:
not just "the cartridge loads", but "the cartridge *runs the same
trajectory* on a second runtime built from scratch".

The fixture (``08_trajectory_two_tools``) exercises three steps:
``echo`` → ``double`` → ``done``. Both runtimes are driven by the
same script and asserted against the same ``expected_trajectory.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TINYLOOP_PATH = REPO_ROOT / "examples" / "alt_runtime" / "tinyloop.py"
FIXTURE = REPO_ROOT / "tests" / "conformance" / "fixtures" / "08_trajectory_two_tools"

SCRIPT = [
    {"tool": "echo", "args": {"message": "hi"}},
    {"tool": "double", "args": {"n": 4}},
    {"tool": "done", "args": {"summary": "ok"}},
]


@pytest.fixture(scope="module")
def tinyloop_module():
    spec = importlib.util.spec_from_file_location("tinyloop_alt_traj", TINYLOOP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["tinyloop_alt_traj"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("tinyloop_alt_traj", None)


def _normalise_tinyloop(steps: list[dict]) -> list[dict]:
    """Project tinyloop steps to the spec-pinned trajectory subset."""
    out: list[dict] = []
    for i, s in enumerate(steps, start=1):
        out.append(
            {
                "step": i,
                "tool": s["tool"],
                "args": s["args"],
                "result": s["result"],
                "error": s["error"],
            }
        )
    return out


def _normalise_looplet(steps: list) -> list[dict]:
    """Project looplet Step objects to the same subset."""
    out: list[dict] = []
    for s in steps:
        out.append(
            {
                "step": s.number,
                "tool": s.tool_call.tool,
                "args": s.tool_call.args,
                "result": s.tool_result.data,
                "error": s.tool_result.error,
            }
        )
    return out


def test_tinyloop_trajectory_matches_expected(tinyloop_module) -> None:
    cart = tinyloop_module.load_cartridge(FIXTURE / "cartridge")
    raw = tinyloop_module.run_scripted(cart, SCRIPT)
    actual = _normalise_tinyloop(raw)
    expected = json.loads((FIXTURE / "expected_trajectory.json").read_text())
    assert actual == expected


def test_looplet_trajectory_matches_expected() -> None:
    from looplet import LoopConfig, MockLLMBackend, composable_loop
    from looplet.cartridge import cartridge_to_preset
    from looplet.types import DefaultState

    preset = cartridge_to_preset(str(FIXTURE / "cartridge"), strict=True)
    # MockLLMBackend script - JSON-formatted tool calls, one per turn.
    responses = [
        json.dumps({"tool": c["tool"], "args": c["args"], "reasoning": "scripted"}) for c in SCRIPT
    ]
    llm = MockLLMBackend(responses=responses)
    config = LoopConfig(
        max_steps=preset.config.max_steps,
        system_prompt=preset.config.system_prompt,
        done_tool=preset.config.done_tool,
        use_native_tools=False,
    )
    state = DefaultState(max_steps=config.max_steps)
    raw = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=state,
            config=config,
            task={"goal": "scripted run"},
        )
    )
    actual = _normalise_looplet(raw)
    expected = json.loads((FIXTURE / "expected_trajectory.json").read_text())
    assert actual == expected


def test_both_runtimes_agree(tinyloop_module) -> None:
    """The portability claim: both runtimes produce the SAME trajectory."""
    from looplet import LoopConfig, MockLLMBackend, composable_loop
    from looplet.cartridge import cartridge_to_preset
    from looplet.types import DefaultState

    cart = tinyloop_module.load_cartridge(FIXTURE / "cartridge")
    tiny = _normalise_tinyloop(tinyloop_module.run_scripted(cart, SCRIPT))

    preset = cartridge_to_preset(str(FIXTURE / "cartridge"), strict=True)
    responses = [
        json.dumps({"tool": c["tool"], "args": c["args"], "reasoning": "scripted"}) for c in SCRIPT
    ]
    config = LoopConfig(
        max_steps=preset.config.max_steps,
        system_prompt=preset.config.system_prompt,
        done_tool=preset.config.done_tool,
        use_native_tools=False,
    )
    big = _normalise_looplet(
        list(
            composable_loop(
                llm=MockLLMBackend(responses=responses),
                tools=preset.tools,
                state=DefaultState(max_steps=config.max_steps),
                config=config,
                task={"goal": "scripted run"},
            )
        )
    )
    assert tiny == big
