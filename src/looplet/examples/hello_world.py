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

from looplet import (
    DefaultState,
    EvalContext,
    EvalHook,
    LoopConfig,
    MockLLMBackend,
    composable_loop,
    probe_native_tool_support,
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
    """Did the agent call done()?"""
    return "done" in ctx.tool_sequence


# ── Main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from looplet.backends import OpenAIBackend

    parser = argparse.ArgumentParser(description="Run the simplest looplet agent.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="Model name")
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
        url = args.base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        api_key = os.environ.get("OPENAI_API_KEY", "x")
        llm = OpenAIBackend(base_url=url, api_key=api_key, model=model)
        model_label = model

    protocol_probe = probe_native_tool_support(llm)
    print("looplet hello world")
    print(f"Model: {model_label}")
    print(f"Tool protocol: {'native' if protocol_probe.supported else 'json-text'}")
    print(f"Probe: {protocol_probe.reason}")
    print()

    for step in composable_loop(
        llm=llm,
        tools=build_tools(),
        state=DefaultState(max_steps=args.max_steps),
        config=LoopConfig(max_steps=args.max_steps, use_native_tools=protocol_probe.supported),
        task={"goal": "Greet Alice and Bob, then finish."},
        hooks=[
            EvalHook(
                evaluators=[eval_greeted_everyone, eval_completed],
                verbose=True,
            )
        ],
    ):
        print(step.pretty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
