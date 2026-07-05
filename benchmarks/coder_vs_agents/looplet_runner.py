#!/usr/bin/env python
"""Standalone driver: run the looplet *coder* cartridge on one task and emit
JSON metrics on a single marked line.

This mirrors exactly what ``looplet run-workspace`` does internally
(see ``looplet.cli.factory_commands.cmd_run_workspace``): build an
OpenAI backend from env, load the cartridge into a preset, and drive
``composable_loop``.  On top of that it instruments the backend to
count LLM calls and approximate token usage (the copilot proxy strips
the ``usage`` block, so we fall back to a chars/4 estimate).

Usage:
    looplet_runner.py <cartridge_dir> <project_root> <max_steps> <task>

Env:
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> int:
    if len(sys.argv) != 5:
        print("LOOPLET_RESULT " + json.dumps({"error": "bad args"}))
        return 2
    cartridge, project_root, max_steps_s, task = sys.argv[1:5]
    max_steps = int(max_steps_s)

    from looplet import cartridge_to_preset, composable_loop
    from looplet.backends import OpenAIBackend
    from looplet.types import DefaultState

    llm = OpenAIBackend(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "x"),
        model=os.environ["OPENAI_MODEL"],
    )

    # ── instrument the backend for a coarse token estimate ──
    counters = {"llm_calls": 0, "in_chars": 0, "out_chars": 0}

    def _wrap(method_name: str) -> None:
        orig = getattr(llm, method_name, None)
        if orig is None:
            return

        def wrapped(*args, **kwargs):  # noqa: ANN002, ANN003
            counters["llm_calls"] += 1
            try:
                # messages are usually the first positional arg
                counters["in_chars"] += len(str(args)) + len(str(kwargs))
            except Exception:  # noqa: BLE001
                pass
            result = orig(*args, **kwargs)
            try:
                counters["out_chars"] += len(str(result))
            except Exception:  # noqa: BLE001
                pass
            return result

        setattr(llm, method_name, wrapped)

    _wrap("generate")
    _wrap("generate_with_tools")

    runtime = {"project_root": os.path.abspath(project_root)}
    preset = cartridge_to_preset(cartridge, runtime=runtime)
    preset.config.max_steps = max_steps
    state = DefaultState(max_steps=max_steps)

    t0 = time.time()
    n_steps = 0
    summary = ""
    tools_used: dict[str, int] = {}
    error = None
    try:
        for step in composable_loop(
            llm=llm,
            config=preset.config,
            tools=preset.tools,
            state=state,
            hooks=preset.hooks,
            task={"goal": task},
        ):
            n_steps += 1
            tc = step.tool_call
            if tc is not None:
                tools_used[tc.tool] = tools_used.get(tc.tool, 0) + 1
                tr = step.tool_result
                if tc.tool == "done" and tr is not None and isinstance(tr.data, dict):
                    summary = str(tr.data.get("summary", ""))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0

    result = {
        "steps": n_steps,
        "llm_calls": counters["llm_calls"],
        "est_in_tokens": counters["in_chars"] // 4,
        "est_out_tokens": counters["out_chars"] // 4,
        "est_total_tokens": (counters["in_chars"] + counters["out_chars"]) // 4,
        "elapsed_s": round(elapsed, 2),
        "summary": summary[:400],
        "tools_used": tools_used,
        "error": error,
    }
    print("LOOPLET_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
