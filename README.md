# openharness

A composable, tool-calling LLM agent harness — the inner loop extracted from
[cadence](https://github.com/hsaghir/cadence).

Provides:

- `composable_loop` / `async_composable_loop` — tool-calling LLM agent loop with hooks
- `LoopConfig`, `LoopHook` — configuration and lifecycle hooks
- `BaseToolRegistry`, `ToolSpec`, `register_think_tool` — tool plumbing
- `Conversation`, `SessionLog`, `InvestigationLog` — conversation & logging
- `Checkpoint`, `CheckpointStore` — resume/recovery
- `ContextManagerHook`, scaffolding utilities — context management
- Streaming events, telemetry (`Tracer`, `MetricsCollector`)
- Backends: OpenAI, Anthropic (sync + async, streaming variants)
- `run_sub_loop` — sub-agent spawning

## Install

```bash
uv add openharness
# or: pip install openharness
```

## Usage

```python
from openharness import composable_loop, LoopConfig, BaseToolRegistry

tools = BaseToolRegistry()
cfg = LoopConfig(max_steps=10)
composable_loop(backend=my_llm, tools=tools, config=cfg, ...)
```

See `src/openharness/examples/` for full examples.

## License

Same as cadence (see parent project).
