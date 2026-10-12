"""Run greeting children concurrently with no model service."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path

from looplet import cartridge_to_preset, run_sub_loops_async
from looplet.testing import AsyncMockLLMBackend

HELLO = Path(__file__).resolve().parents[2] / "hello.cartridge"


async def main(names: list[str] | None = None) -> list[dict]:
    names = names if names is not None else ["Alice", "Bob", "Charlie"]
    with ExitStack() as owned:
        requests = []
        for name in names:
            preset = owned.enter_context(cartridge_to_preset(HELLO))
            preset.config.max_steps = 3
            requests.append(
                {
                    "llm": AsyncMockLLMBackend(
                        [
                            json.dumps({"tool": "greet", "args": {"name": name}}),
                            json.dumps({"tool": "done", "args": {"summary": f"Greeted {name}"}}),
                        ],
                        cycle=False,
                    ),
                    "task": {"goal": f"Greet {name}"},
                    "sub_tools": preset.tools,
                    "hooks": preset.hooks,
                    "state": preset.state,
                    "config": preset.config,
                }
            )
        results = await run_sub_loops_async(requests, max_concurrency=2)
        for result in results:
            print(result["output"]["summary"])
        return results


if __name__ == "__main__":
    asyncio.run(main())
