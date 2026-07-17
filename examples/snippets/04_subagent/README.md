# 04 - Sub-agent as tool

A cartridge can be invoked as a single tool call from another
cartridge. From the calling agent's perspective, this is just another
tool; the tool body happens to load a cartridge and run its loop.

This snippet shows a `wrap_workspace_as_tool` helper plus a 30-line
demo that runs the [hello.cartridge](../../hello.cartridge) as a
sub-agent of a parent script. No new abstraction - just a Python
function whose body calls `run_sub_loop`.

```bash
# Requires: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL set.
uv run python examples/snippets/04_subagent/demo.py "echo hello"
```

The interesting property: the parent does not know whether it called
a Python function, an MCP server, or another whole agent. All three
are tool calls. Recursion bottoms out when the calling agent decides.
