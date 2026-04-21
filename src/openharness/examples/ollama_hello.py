"""Ollama hello world — run openharness against a local model, no API key.

Prereqs:
    curl -fsSL https://ollama.com/install.sh | sh      # install ollama
    ollama pull llama3.1:8b                            # pull a model
    ollama serve                                       # starts on :11434
    pip install "openharness[openai]"

Run:
    python -m openharness.examples.ollama_hello
    OLLAMA_MODEL=qwen2.5:7b python -m openharness.examples.ollama_hello
"""
from __future__ import annotations

import os

from openharness import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from openharness.tools import ToolSpec


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit('pip install "openharness[openai]"')

    from openharness.backends import OpenAIBackend

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    llm = OpenAIBackend(OpenAI(base_url=base_url, api_key="ollama"), model=model)

    tools = BaseToolRegistry()
    tools.register(ToolSpec(
        name="greet",
        description="Return a greeting for a person.",
        parameters={"name": "str"},
        execute=lambda *, name: {"greeting": f"Hello, {name}!"},
    ))
    tools.register(ToolSpec(
        name="done",
        description="Finish with a final answer.",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))

    for step in composable_loop(
        llm=llm,
        tools=tools,
        state=DefaultState(max_steps=5),
        config=LoopConfig(max_steps=5),
        task={"goal": "Greet Alice and Bob, then call done with a summary."},
    ):
        print(step.pretty())


if __name__ == "__main__":
    main()
