"""Back-compat + example-cartridge regression (RPC foundation §1.6).

Proves the RPC-foundation additions are non-breaking: a legacy client
that uses only load_workspace/set_backend/run/quit and reads only the
step/done/ready/error events still works — even though the server now
also emits `event` and `checkpoint` frames — and every shipped
local-first example cartridge still loads through the v1 loader.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from looplet.cartridge import cartridge_to_preset
from looplet.cartridge.scaffold import scaffold_cartridge
from looplet.rpc import RPCServer
from looplet.testing import MockLLMBackend

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# The README's five canonical, local-first example cartridges.
LOCAL_FIRST_EXAMPLES = [
    "hello.cartridge",
    "coder.cartridge",
    "dep_doctor.cartridge",
    "git_detective.cartridge",
    "threat_intel.cartridge",
]

# Pre-foundation clients only ever knew these event kinds.
LEGACY_EVENTS = {"ready", "step", "done", "error"}


# Module-level factory so RPC's _import_factory can resolve it by dotted path.
def make_mock_backend() -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            '{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}',
            '{"tool": "done", "args": {"answer": "hi"}, "reasoning": "finish"}',
        ]
    )


def _drive(commands: list[dict]) -> list[dict]:
    in_buf = io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n")
    out_buf = io.StringIO()
    RPCServer(in_stream=in_buf, out_stream=out_buf).serve_forever()
    out_buf.seek(0)
    return [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]


def test_legacy_step_done_only_client_unaffected(tmp_path: Path) -> None:
    """A pre-foundation client reading ONLY the legacy event kinds still
    completes a run correctly, despite the new additive frames."""
    ws = scaffold_cartridge(tmp_path / "w.workspace", name="w", tools=["greet"])
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name):\n    return {'greeting': f'Hello, {name}!'}\n"
    )
    events = _drive(
        [
            {"cmd": "load_workspace", "path": str(ws), "runtime": {"workspace": str(tmp_path)}},
            {"cmd": "set_backend", "factory": "tests.test_rpc_backcompat:make_mock_backend"},
            {"cmd": "run", "task": {"goal": "greet world"}, "max_steps": 5},
            {"cmd": "quit"},
        ]
    )
    legacy = [e for e in events if e["event"] in LEGACY_EVENTS]
    steps = [e for e in legacy if e["event"] == "step"]
    done = [e for e in legacy if e["event"] == "done"]
    assert steps, "legacy client must still receive step events"
    assert any(s["step"]["tool_call"]["tool"] == "done" for s in steps)
    assert len(done) == 1 and done[0]["steps"] >= 2
    assert not any(e["event"] == "error" for e in legacy)


@pytest.mark.parametrize("name", LOCAL_FIRST_EXAMPLES)
def test_example_cartridge_still_loads(name: str) -> None:
    """Every shipped local-first example cartridge still materialises
    through the v1 loader unchanged (additive-only proof)."""
    path = EXAMPLES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    preset = cartridge_to_preset(path)
    assert preset.config is not None
    assert preset.tools is not None
