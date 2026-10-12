"""Hello world - the simplest possible looplet agent.

This is the starting point. One tool. Real LLM or scripted local run. With eval.

    python -m looplet.examples.hello_world
    python -m looplet.examples.hello_world --model claude-sonnet-4
    python -m looplet.examples.hello_world --scripted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

from looplet import (
    AgentPreset,
    DefaultState,
    EvalContext,
    EvalHook,
    LoopConfig,
    MockLLMBackend,
    run,
    stream,
    tool,
    tools_from,
)


@tool(description="Return a greeting for a person.")
def greet(*, name: str) -> dict:
    return {"greeting": f"Hello, {name}!"}


def build_tools():
    return tools_from(
        [greet],
        include_done=True,
        done_parameters={"answer": "Final answer after greeting everyone"},
    )


def _tool_call(tool_name: str, args: dict, reasoning: str) -> str:
    return json.dumps({"tool": tool_name, "args": args, "reasoning": reasoning})


def scripted_responses() -> list[str]:
    return [
        _tool_call("greet", {"name": "Alice"}, "greet Alice first"),
        _tool_call("greet", {"name": "Bob"}, "greet Bob next"),
        _tool_call(
            "done",
            {"answer": "Greeted Alice and Bob."},
            "finish after greeting everyone",
        ),
    ]


def build_agent(*, max_steps: int = 5, scripted: bool = False) -> AgentPreset:
    """Build fresh state, tools, and evaluation hooks for one execution."""
    return AgentPreset(
        tools=build_tools(),
        state=DefaultState(max_steps=max_steps),
        config=LoopConfig(max_steps=max_steps, use_native_tools=not scripted),
        hooks=[EvalHook(evaluators=[eval_greeted_everyone, eval_completed], verbose=True)],
    )


# ── Eval: did the agent greet everyone? ──────────────────────────


def eval_greeted_everyone(ctx: EvalContext) -> float:
    """Check that the agent greeted both Alice and Bob."""
    greeted = set()
    for step in ctx.steps:
        args = getattr(step.tool_call, "args", {})
        if getattr(step.tool_call, "tool", "") == "greet":
            greeted.add(args.get("name", "").lower())
    expected = {"alice", "bob"}
    return len(greeted & expected) / len(expected)


def eval_completed(ctx: EvalContext) -> bool:
    """Was the agent's completion accepted?"""
    return ctx.completed


# ── Main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from looplet.backends import OpenAIBackend, make_backend

    parser = argparse.ArgumentParser(description="Run the simplest looplet agent.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--stream", action="store_true", help="Print each step as it happens")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run a deterministic local demo with MockLLMBackend instead of a real model.",
    )
    args = parser.parse_args(argv)

    if args.scripted:
        llm = MockLLMBackend(responses=scripted_responses())
        model_label = "scripted MockLLMBackend"
    else:
        try:
            if args.base_url:
                llm = OpenAIBackend(
                    base_url=args.base_url,
                    api_key=os.environ.get("OPENAI_API_KEY") or "x",
                    model=args.model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
                )
            else:
                llm = make_backend(model=args.model)
        except (ImportError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        model_label = getattr(llm, "_model", type(llm).__name__)

    print("looplet hello world")
    print(f"Model: {model_label}")
    print("Tool protocol: json-text" if args.scripted else "Tool protocol: native")
    print()

    factory = partial(build_agent, max_steps=args.max_steps, scripted=args.scripted)
    task = "Greet Alice and Bob, then finish."
    if args.stream:
        with stream(factory, task, llm=llm) as execution:
            for step in execution:
                print(step.pretty())
        result = execution.result
    else:
        result = run(factory, task, llm=llm)
        for step in result.steps:
            print(step.pretty())
    print(f"Result: {result.status.value} ({result.termination_reason})")
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
