"""Run the hello.workspace via three different runtimes.

All three use the same MockLLMBackend script so the demo is offline
and deterministic. The point is that the same workspace artifact
plugs into multiple runtimes without modification.

Run::

    uv run python run_three_ways.py
"""

from __future__ import annotations

import json
from pathlib import Path

from looplet import (
    DefaultState,
    composable_loop,
    workspace_to_preset,
)
from looplet.subagent import run_sub_loop
from looplet.testing import MockLLMBackend

REPO = Path(__file__).resolve().parents[3]
HELLO_WS = REPO / "examples" / "hello.workspace"


def make_backend() -> MockLLMBackend:
    """Scripted: greet then done."""
    return MockLLMBackend(
        responses=[
            json.dumps({"tool": "greet", "args": {"name": "Alice"}, "reasoning": "say hi"}),
            json.dumps({"tool": "done", "args": {"answer": "greeted"}, "reasoning": "wrap up"}),
        ]
    )


def via_local_loop() -> int:
    print("=== runtime 1: local composable_loop ===")
    backend = make_backend()
    preset = workspace_to_preset(str(HELLO_WS), runtime={"workspace": str(REPO)})
    state = DefaultState(max_steps=preset.config.max_steps)
    n = 0
    for step in composable_loop(
        llm=backend,
        tools=preset.tools,
        state=state,
        config=preset.config,
        task={"goal": "say hi to Alice"},
    ):
        print(" ", step.pretty())
        n += 1
    return n


def via_subagent() -> int:
    print("=== runtime 2: sub-agent invocation ===")
    backend = make_backend()
    preset = workspace_to_preset(str(HELLO_WS), runtime={"workspace": str(REPO)})
    result = run_sub_loop(
        llm=backend,
        tools=preset.tools,
        config=preset.config,
        task={"goal": "say hi to Alice"},
        max_steps=preset.config.max_steps,
        system_prompt=preset.config.system_prompt,
    )
    steps = result.get("steps", []) or []
    n = len(steps) if isinstance(steps, list) else int(steps)
    print(f"  sub-agent ran {n} step(s)")
    return n


def via_replay() -> int:
    """Re-run the same workspace with the same scripted responses,
    confirming determinism. In a real provenance setup, this would
    load saved trajectory data instead of re-scripting."""
    print("=== runtime 3: replay (deterministic re-run) ===")
    backend = make_backend()
    preset = workspace_to_preset(str(HELLO_WS), runtime={"workspace": str(REPO)})
    state = DefaultState(max_steps=preset.config.max_steps)
    seen_tools = []
    for step in composable_loop(
        llm=backend,
        tools=preset.tools,
        state=state,
        config=preset.config,
        task={"goal": "say hi to Alice"},
    ):
        seen_tools.append(step.tool_call.tool)
    print("  trajectory tools:", seen_tools)
    return len(seen_tools)


def main() -> None:
    n_local = via_local_loop()
    print()
    n_sub = via_subagent()
    print()
    n_replay = via_replay()
    print()
    print(f"steps: local={n_local}, sub-agent={n_sub}, replay={n_replay}")


if __name__ == "__main__":
    main()
