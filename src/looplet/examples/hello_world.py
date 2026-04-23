"""Hello world — the simplest possible looplet agent.

This is the starting point. One tool. Real LLM. With eval.

    python -m looplet.examples.hello_world
    python -m looplet.examples.hello_world --model claude-sonnet-4
"""

from __future__ import annotations

import os

from looplet import (
    BaseToolRegistry,
    DefaultState,
    EvalContext,
    EvalHook,
    LoopConfig,
    composable_loop,
)
from looplet.tools import ToolSpec

# ── Eval: did the agent greet everyone? ──────────────────────────


def eval_greeted_everyone(ctx: EvalContext) -> float:
    """Check that the agent greeted both Alice and Bob."""
    greeted = set()
    for s in ctx.steps:
        args = getattr(s.tool_call, "args", {})
        if getattr(s.tool_call, "tool", "") == "greet":
            greeted.add(args.get("name", "").lower())
    expected = {"alice", "bob"}
    return len(greeted & expected) / len(expected)


def eval_completed(ctx: EvalContext) -> bool:
    """Did the agent call done()?"""
    return "done" in ctx.tool_sequence


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    from looplet.backends import OpenAIBackend

    url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:19823/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    api_key = os.environ.get("OPENAI_API_KEY", "x")
    llm = OpenAIBackend(base_url=url, api_key=api_key, model=model)

    from looplet.tools import register_done_tool  # noqa: PLC0415

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="greet",
            description="Return a greeting for a person.",
            parameters={"name": "str"},
            execute=lambda *, name: {"greeting": f"Hello, {name}!"},
        )
    )
    register_done_tool(tools, parameters={"answer": "str"})

    for step in composable_loop(
        llm=llm,
        tools=tools,
        state=DefaultState(max_steps=5),
        config=LoopConfig(max_steps=5),
        task={"goal": "Greet Alice and Bob, then finish."},
        hooks=[
            EvalHook(
                evaluators=[eval_greeted_everyone, eval_completed],
                verbose=True,
            )
        ],
    ):
        print(step.pretty())


if __name__ == "__main__":
    main()
