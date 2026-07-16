"""Tag a trajectory with cartridge identity (path + content hash).

Computes a stable SHA-256 over all non-cache cartridge files,
runs the agent once with a scripted MockLLMBackend, and writes a
JSON record that pairs the trajectory with the cartridge identity.

Run::

    python tag_trajectory.py <cartridge_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from looplet import DefaultState, cartridge_to_preset, composable_loop
from looplet.testing import MockLLMBackend


def cartridge_identity(cartridge: Path) -> dict:
    h = hashlib.sha256()
    files = []
    for path in sorted(cartridge.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(cartridge))
        data = path.read_bytes()
        h.update(rel.encode())
        h.update(b"\x00")
        h.update(hashlib.sha256(data).digest())
        files.append(rel)
    name = "?"
    manifest = cartridge / "cartridge.json"
    if manifest.is_file():
        try:
            name = json.loads(manifest.read_text()).get("name", "?")
        except json.JSONDecodeError:
            pass
    return {
        "path": str(cartridge),
        "name": name,
        "content_sha256": h.hexdigest(),
        "files": len(files),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: tag_trajectory.py <cartridge_dir>")
    cartridge = Path(sys.argv[1]).resolve()
    identity = cartridge_identity(cartridge)

    backend = MockLLMBackend(
        responses=[
            json.dumps({"tool": "greet", "args": {"name": "Alice"}, "reasoning": "say hi"}),
            json.dumps({"tool": "done", "args": {"summary": "greeted"}, "reasoning": "wrap up"}),
        ]
    )
    preset = cartridge_to_preset(
        str(cartridge),
        runtime={"project_root": str(cartridge.parent)},
    )
    state = DefaultState(max_steps=preset.config.max_steps)

    trajectory = []
    for step in composable_loop(
        llm=backend,
        tools=preset.tools,
        state=state,
        config=preset.config,
        task={"goal": "greet Alice"},
    ):
        trajectory.append(
            {
                "tool": step.tool_call.tool,
                "ok": step.tool_result.error is None,
                "duration_ms": round(step.tool_result.duration_ms, 2),
            }
        )

    record = {"cartridge": identity, "trajectory": trajectory}
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
