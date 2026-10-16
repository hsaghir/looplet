# 04 - Child Agent As Tool

A child agent is a tool whose implementation runs another loop.
The parent receives the child's result and continues its own loop.
Each child has separate state and conversation history.
Its tools can still use shared files and services.

`demo.py` calls the greeting cartridge from a synchronous parent.
Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` before running it:

```bash
uv run python examples/snippets/04_subagent/demo.py "Greet Alice"
```

`async_demo.py` runs three greeting children with a concurrency limit of two.
It uses fixed model responses and real greeting tools.
It needs no model service:

```bash
uv run python examples/snippets/04_subagent/async_demo.py
```

The async example requires Looplet `0.4.0` or newer.
Both examples explicitly load and close their child presets.
The loops remain ordinary Looplet loops.
Use `run_sub_loop_async` for one async child.
Use `run_sub_loops_async` for bounded parallel children.

See [host integration](../../../docs/host-integration.md#bounded-delegation) for permissions, shared call limits, and recovery.
