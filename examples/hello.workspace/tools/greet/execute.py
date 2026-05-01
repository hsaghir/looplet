"""Greet tool — top-level function, no closures.

Receives the shared ``greeting_log`` resource through ``ctx.resources``;
``tool.yaml`` declares ``requires: [greeting_log]`` and the dispatcher
resolves the ref against the workspace's resource registry.

Mutates the log so other components (e.g. PolitenessGate hook) can
audit greetings later.
"""

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, name: str) -> dict:
    text = f"Hello, {name}!"
    log = ctx.resources.get("greeting_log")
    if log is not None:
        log.record(name, text)
    return {"greeting": text}
