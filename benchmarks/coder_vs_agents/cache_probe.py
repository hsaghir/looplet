#!/usr/bin/env python
"""Measure the looplet coder cartridge's prompt-cache friendliness.

Prompt caching lets a provider reuse the identical leading tokens (the
"prefix") of successive LLM calls. This probe drives the coder loop and
measures how much of each call's prompt is an unchanged prefix of the
previous call's -- the *cacheable fraction*, i.e. the ceiling a
prefix-caching backend could reuse.

Actual cache hits aren't observable when looplet runs behind a proxy that
strips the response ``usage`` block, so prompt-prefix stability is the
measurable, backend-independent proxy for cache-friendliness. It is driven
entirely by prompt *layout*: any per-turn-varying content (a step counter, a
timestamp, a re-rendered block) placed ahead of the stable system + task +
history caps the cacheable prefix.

Run (any OpenAI-compatible endpoint):
    OPENAI_BASE_URL=... OPENAI_API_KEY=x OPENAI_MODEL=... python cache_probe.py
Pass one or more probe-task ids to run a subset.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CARTRIDGE = str(REPO / "examples" / "coder.cartridge")

PROBE_TASKS = [
    {
        "id": "multi_step_coding",
        "max_steps": 15,
        "prompt": (
            "Create a file stack.py with a Stack class supporting push, pop, "
            "peek, is_empty, and __len__ (pop or peek on an empty stack raises "
            "IndexError). Then create test_stack.py with pytest tests covering "
            "every method including the empty-stack errors, run pytest, and fix "
            "anything that fails. Then stop."
        ),
        "seed": {},
    },
    {
        "id": "one_shot_explain",
        "max_steps": 8,
        "prompt": (
            "Explain the difference between a process and a thread, with three "
            "concrete tradeoffs. Write the answer to answer.md, then stop."
        ),
        "seed": {},
    },
]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def probe(task: dict) -> dict:
    """Run one probe task and return its cacheable-prefix fraction."""
    from looplet import cartridge_to_preset, composable_loop
    from looplet.backends import OpenAIBackend
    from looplet.types import DefaultState

    ws = Path(tempfile.mkdtemp(prefix=f"cacheprobe-{task['id']}-"))
    for rel, content in task.get("seed", {}).items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    llm = OpenAIBackend(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "x"),
        model=os.environ["OPENAI_MODEL"],
    )

    # Capture the (system_prompt, prompt) sent on each LLM call.
    calls: list[str] = []

    def _instrument(name: str) -> None:
        orig = getattr(llm, name, None)
        if orig is None:
            return

        def wrapped(prompt, **kw):  # noqa: ANN001, ANN003
            calls.append((kw.get("system_prompt", "") or "") + "\x00" + str(prompt))
            return orig(prompt, **kw)

        setattr(llm, name, wrapped)

    _instrument("generate")
    _instrument("generate_with_tools")

    preset = cartridge_to_preset(CARTRIDGE, runtime={"project_root": str(ws)})
    preset.config.max_steps = task["max_steps"]
    state = DefaultState(max_steps=task["max_steps"])
    try:
        for _step in composable_loop(
            llm=llm,
            config=preset.config,
            tools=preset.tools,
            state=state,
            hooks=preset.hooks,
            task={"goal": task["prompt"]},
        ):
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"# {task['id']} loop error: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    pref = tot = 0
    for i in range(1, len(calls)):
        pref += _lcp(calls[i - 1], calls[i])
        tot += len(calls[i])
    return {
        "task": task["id"],
        "n_llm_calls": len(calls),
        "cacheable_fraction": round(pref / tot, 3) if tot else None,
    }


def main(argv: list[str]) -> int:
    tasks = [t for t in PROBE_TASKS if not argv or t["id"] in argv]
    rows = [probe(t) for t in tasks]
    print(f"\n{'task':22s} {'llm_calls':>9s} {'cacheable_prefix':>17s}")
    print("-" * 50)
    for r in rows:
        cf = r["cacheable_fraction"]
        shown = f"{cf * 100:.1f}%" if cf is not None else "n/a"
        print(f"{r['task']:22s} {r['n_llm_calls']:>9d} {shown:>17s}")
    print(
        "\ncacheable_prefix = fraction of each call's prompt reusable as an "
        "unchanged\nprefix from the previous call = the prompt-cache hit ceiling."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
