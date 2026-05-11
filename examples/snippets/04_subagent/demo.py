"""Wrap a workspace as a tool, then call it from a parent loop.

The parent runs a tiny scripted loop with one custom tool,
``ask_helper``, whose body loads ``examples/hello.cartridge``,
runs ``run_sub_loop`` against it with a fresh task, and returns the
sub-agent's summary. The parent never directly imports the sub-agent;
it only knows it called a tool.

Requires OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from looplet import (
    DefaultState,
    LoopConfig,
    cartridge_to_preset,
    composable_loop,
    tool,
    tools_from,
)
from looplet.backends import OpenAIBackend
from looplet.subagent import run_sub_loop

REPO = Path(__file__).resolve().parents[3]
HELLO_WS = REPO / "examples" / "hello.cartridge"


def _backend() -> OpenAIBackend:
    return OpenAIBackend(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    parent_task = argv[0] if argv else "Use ask_helper to greet Alice, then call done."

    backend = _backend()
    sub_preset = cartridge_to_preset(str(HELLO_WS), runtime={"workspace": str(REPO)})

    @tool(description="Run the hello.cartridge sub-agent on a question.")
    def ask_helper(*, question: str) -> dict:
        result = run_sub_loop(
            llm=backend,
            tools=sub_preset.tools,
            config=sub_preset.config,
            task={"goal": question},
            max_steps=4,
            system_prompt=sub_preset.config.system_prompt,
        )
        return {"summary": str(result.get("summary", ""))[:240]}

    parent_tools = tools_from(
        [ask_helper],
        include_done=True,
        done_parameters={"summary": "What the parent learned from the sub-agent."},
    )
    parent_config = LoopConfig(
        max_steps=4,
        system_prompt=(
            "You are a coordinator. Use ask_helper to delegate, then call done() "
            "with what you learned. Never do work yourself."
        ),
    )
    state = DefaultState(max_steps=parent_config.max_steps)

    for step in composable_loop(
        llm=backend,
        tools=parent_tools,
        state=state,
        config=parent_config,
        task={"goal": parent_task},
    ):
        print(step.pretty())

    return 0


if __name__ == "__main__":
    sys.exit(main())
