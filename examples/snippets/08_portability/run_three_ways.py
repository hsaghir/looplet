"""Run the hello.cartridge via three different runtimes.

All three use the same MockLLMBackend script so the demo is offline
and reproducible. The point is that the same cartridge artifact
plugs into multiple runtimes without modification.

Run::

    uv run python run_three_ways.py
"""

from __future__ import annotations

import json
from pathlib import Path

from looplet import cartridge_to_preset
from looplet.subagent import run_sub_loop
from looplet.testing import MockLLMBackend

REPO = Path(__file__).resolve().parents[3]
HELLO_WS = REPO / "examples" / "hello.cartridge"


def make_backend() -> MockLLMBackend:
    """Scripted: greet then done."""
    return MockLLMBackend(
        responses=[
            json.dumps({"tool": "greet", "args": {"name": "Alice"}, "reasoning": "say hi"}),
            json.dumps({"tool": "done", "args": {"summary": "greeted"}, "reasoning": "wrap up"}),
        ]
    )


def via_local_loop() -> int:
    print("=== runtime 1: local preset.run ===")
    backend = make_backend()
    with cartridge_to_preset(str(HELLO_WS), runtime={"project_root": str(REPO)}) as preset:
        count = 0
        for step in preset.run(backend, task={"goal": "say hi to Alice"}):
            print(" ", step.pretty())
            count += 1
        return count


def via_subagent() -> int:
    print("=== runtime 2: sub-agent invocation ===")
    backend = make_backend()
    with cartridge_to_preset(str(HELLO_WS), runtime={"project_root": str(REPO)}) as preset:
        result = run_sub_loop(
            llm=backend,
            sub_tools=preset.tools,
            config=preset.config,
            hooks=preset.hooks,
            task={"goal": "say hi to Alice"},
            max_steps=preset.config.max_steps,
        )
    steps = result.get("steps", []) or []
    n = len(steps) if isinstance(steps, list) else int(steps)
    print(f"  sub-agent ran {n} step(s)")
    return n


def via_scripted_rerun() -> int:
    """Run the same cartridge again with a fresh scripted backend.

    This is not captured-response replay: it does not load provenance
    artifacts through ``replay_loop()``.
    """
    print("=== runtime 3: fresh scripted run ===")
    backend = make_backend()
    with cartridge_to_preset(str(HELLO_WS), runtime={"project_root": str(REPO)}) as preset:
        seen_tools = [
            step.tool_call.tool for step in preset.run(backend, task={"goal": "say hi to Alice"})
        ]
    print("  trajectory tools:", seen_tools)
    return len(seen_tools)


def main() -> None:
    n_local = via_local_loop()
    print()
    n_sub = via_subagent()
    print()
    n_rerun = via_scripted_rerun()
    print()
    print(f"steps: local={n_local}, sub-agent={n_sub}, scripted={n_rerun}")


if __name__ == "__main__":
    main()
