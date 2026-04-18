"""Hello world — the simplest possible openharness agent.

This is the starting point. One tool. Real LLM. With eval.

    python -m openharness.examples.hello_world
    python -m openharness.examples.hello_world --model claude-sonnet-4
"""
from __future__ import annotations

import os

from openharness import (
    BaseToolRegistry,
    DefaultState,
    EvalContext,
    EvalHook,
    LoopConfig,
    composable_loop,
)
from openharness.tools import ToolSpec

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
    from openharness.backends import OpenAIBackend

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:19823/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    llm = OpenAIBackend(OpenAI(base_url=url, api_key="x"), model=model)

    tools = BaseToolRegistry()
    tools.register(ToolSpec(
        name="greet",
        description="Return a greeting for a person.",
        parameters={"name": "str"},
        execute=lambda *, name: {"greeting": f"Hello, {name}!"},
    ))
    tools.register(ToolSpec(
        name="done",
        description="Signal completion.",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))

    for step in composable_loop(
        llm=llm,
        tools=tools,
        state=DefaultState(max_steps=5),
        config=LoopConfig(max_steps=5),
        task={"goal": "Greet Alice and Bob, then finish."},
        hooks=[EvalHook(
            evaluators=[eval_greeted_everyone, eval_completed],
            verbose=True,
        )],
    ):
        print(step.pretty())


if __name__ == "__main__":
    main()
