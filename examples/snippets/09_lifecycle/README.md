# 09 — Lifecycle artifacts: trajectories tagged with cartridge identity

A trajectory or eval result that says "we ran the agent on March 4,
commit `abc123`" is folklore. A trajectory tagged with a **cartridge
identity** (path, content hash, version) is a coherent reference.

This snippet computes a content hash for a cartridge, runs the agent
once, and writes the trajectory to disk with the cartridge identity
embedded. Later analyses can compare trajectories *across cartridge
versions* on the same task.

```bash
uv run python examples/snippets/09_lifecycle/tag_trajectory.py \
    examples/hello.cartridge
```

Output (truncated):

```
{
  "cartridge": {
    "path": ".../examples/hello.cartridge",
    "name": "hello",
    "content_sha256": "f3b21e...",
    "files": 7
  },
  "trajectory": [
    {"tool": "greet", "ok": true, "duration_ms": 1.2},
    {"tool": "done",  "ok": true, "duration_ms": 0.3}
  ]
}
```
