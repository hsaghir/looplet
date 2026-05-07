# 08 — Cross-runtime portability

The same cartridge runs three ways:

1. **Local Python loop** — `composable_loop(...)` directly.
2. **Sub-agent** — invoked from another loop via `run_sub_loop(...)`.
3. **Replay** — a recorded trajectory is paired with the cartridge and
   re-executed against a `MockLLMBackend` to reproduce the run
   without the LLM.

This snippet runs the shipped [hello.workspace](../../hello.workspace)
all three ways with the same backend behaviour, demonstrating that
the artifact is invariant; the runtime is a choice.

```bash
uv run python examples/snippets/08_portability/run_three_ways.py
```

No real LLM is required: the snippet uses `MockLLMBackend` with a
scripted response so the demo is deterministic and offline.
