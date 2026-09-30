# 08 - Cross-runtime portability

The same cartridge runs three ways:

1. **Local Python loop** - `preset.run(...)` with the cartridge's owned wiring.
2. **Sub-agent** - invoked from another loop via `run_sub_loop(...)`.
3. **Fresh scripted rerun** - a new preset and `MockLLMBackend` run the same
   script without calling a provider. This is not captured-response replay;
   use `replay_loop()` with saved provenance for that workflow.

This snippet runs the shipped [hello.cartridge](../../hello.cartridge)
all three ways with the same backend behaviour, demonstrating that
the artifact is invariant; the runtime is a choice.

```bash
uv run python examples/snippets/08_portability/run_three_ways.py
```

No real LLM is required: the snippet uses `MockLLMBackend` with a
scripted response so the demo is deterministic and offline.
