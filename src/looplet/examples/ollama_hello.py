"""Ollama hello world — run looplet against a local model, no API key.

Prereqs:
    curl -fsSL https://ollama.com/install.sh | sh      # install ollama
    ollama pull llama3.1:8b                            # pull a model
    ollama serve                                       # starts on :11434
    pip install "looplet[openai]"

Run:
    python -m looplet.examples.ollama_hello
    OLLAMA_MODEL=qwen2.5:7b python -m looplet.examples.ollama_hello
    python -m looplet.examples.ollama_hello --scripted
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from looplet import (
    DefaultState,
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


def main(argv: list[str] | None = None) -> int:
    from looplet.backends import OpenAIBackend

    parser = argparse.ArgumentParser(description="Run looplet against Ollama or a script.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--base-url", default=None, help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run a deterministic local demo with MockLLMBackend instead of Ollama.",
    )
    args = parser.parse_args(argv)

    if args.scripted:
        llm = MockLLMBackend(responses=scripted_responses())
        model_label = "scripted MockLLMBackend"
    else:
        base_url = args.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = args.model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        llm = OpenAIBackend(base_url=base_url, api_key="ollama", model=model)
        model_label = model

    protocol_probe = probe_native_tool_support(llm)
    print("looplet ollama hello")
    print(f"Model: {model_label}")
    print(f"Tool protocol: {'native' if protocol_probe.supported else 'json-text'}")
    print(f"Probe: {protocol_probe.reason}")
    print()

    for step in composable_loop(
        llm=llm,
        tools=build_tools(),
        state=DefaultState(max_steps=args.max_steps),
        config=LoopConfig(
            max_steps=args.max_steps,
            use_native_tools=protocol_probe.supported,
        ),
        task={"goal": "Greet Alice and Bob, then call done with a summary."},
    ):
        print(step.pretty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
