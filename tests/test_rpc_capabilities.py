"""Tests for the capability handshake on ``load_workspace`` (RPC §1.1).

The ``ready`` event emitted after ``load_workspace`` must carry a
``capabilities`` dict per the frozen contract::

    {events, cancel, checkpoint, cost, permission_authority, stop_reasons[]}

Values are derived from what the loaded :class:`AgentPreset` actually
supports — e.g. ``permission_authority`` is true only when a permission
hook (PermissionEngine-backed or LEP) is present. The bare ``ready``
emitted for every other command is unchanged (additive only).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from looplet.cartridge import cartridge_to_preset
from looplet.cartridge.scaffold import scaffold_cartridge
from looplet.loop import LoopConfig
from looplet.rpc import STOP_REASONS, RPCServer, _capabilities
from looplet.testing import MockLLMBackend

EXPECTED_KEYS = {
    "events",
    "cancel",
    "checkpoint",
    "cost",
    "permission_authority",
    "stop_reasons",
}

# The frozen stop-reason enum (formalised as a StopReason enum in §1.3).
EXPECTED_STOP_REASONS = [
    "done",
    "max_steps",
    "budget",
    "stagnated",
    "cancelled",
    "error",
]


# Module-level factory so RPC's _import_factory can resolve it by dotted path.
def make_mock_backend() -> MockLLMBackend:
    return MockLLMBackend(responses=['{"tool": "done", "args": {"answer": "x"}, "reasoning": "f"}'])


def _drive(commands: list[dict]) -> list[dict]:
    """Run RPCServer over an in-memory pipe; return parsed events."""
    in_buf = io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n")
    out_buf = io.StringIO()
    RPCServer(in_stream=in_buf, out_stream=out_buf).serve_forever()
    out_buf.seek(0)
    return [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]


def _scaffold(tmp_path: Path) -> Path:
    ws = scaffold_cartridge(tmp_path / "w.workspace", name="w", tools=["greet"])
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name):\n    return {'greeting': f'Hi {name}'}\n"
    )
    return ws


class _FakePreset:
    """Minimal preset-like object for unit-testing ``_capabilities``."""

    def __init__(
        self,
        *,
        hooks: list | None = None,
        resources: dict | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self.hooks = hooks or []
        self.resources = resources or {}
        self.config = config or LoopConfig(max_steps=5)


# ── AC-1: load_workspace ready carries the capabilities dict ─────────


def test_load_workspace_ready_carries_capabilities(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path)
    events = _drive(
        [
            {"cmd": "load_workspace", "path": str(ws), "runtime": {"workspace": str(tmp_path)}},
            {"cmd": "quit"},
        ]
    )
    ready = next(e for e in events if e["event"] == "ready" and "loaded" in e)
    assert "capabilities" in ready
    caps = ready["capabilities"]
    assert set(caps) == EXPECTED_KEYS
    # The bare load fields remain untouched.
    assert ready["loaded"] == str(ws)
    assert isinstance(ready["tools"], list)


def test_capabilities_stop_reasons_enum(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path)
    events = _drive(
        [
            {"cmd": "load_workspace", "path": str(ws), "runtime": {"workspace": str(tmp_path)}},
            {"cmd": "quit"},
        ]
    )
    ready = next(e for e in events if e["event"] == "ready" and "loaded" in e)
    assert ready["capabilities"]["stop_reasons"] == EXPECTED_STOP_REASONS


def test_plain_cartridge_capability_defaults(tmp_path: Path) -> None:
    """A vanilla scaffolded cartridge has no permission hook or cost sink,
    but the server always offers events + cancel + checkpoint."""
    ws = _scaffold(tmp_path)
    preset = cartridge_to_preset(str(ws), runtime={"workspace": str(tmp_path)})
    caps = _capabilities(preset)
    assert caps["permission_authority"] is False
    assert caps["cost"] is False
    assert caps["checkpoint"] is True
    assert caps["events"] is True
    assert caps["cancel"] is True


# ── AC-2: capabilities reflect the loaded preset ────────────────────


def test_permission_authority_true_with_permission_hook() -> None:
    from looplet.permissions import PermissionEngine, PermissionHook

    assert _capabilities(_FakePreset())["permission_authority"] is False
    hook = PermissionHook(PermissionEngine())
    caps = _capabilities(_FakePreset(hooks=[hook]))
    assert caps["permission_authority"] is True


def test_permission_authority_true_with_lep_hook() -> None:
    from looplet.lep import LEPHookAdapter

    adapter = LEPHookAdapter(["python", "-c", "pass"])
    caps = _capabilities(_FakePreset(hooks=[adapter]))
    assert caps["permission_authority"] is True


def test_cost_true_with_cost_hook() -> None:
    from looplet.cost import CostHook, CostTracker

    assert _capabilities(_FakePreset())["cost"] is False
    caps = _capabilities(_FakePreset(hooks=[CostHook(CostTracker())]))
    assert caps["cost"] is True


def test_cost_true_with_cost_tracker_resource() -> None:
    from looplet.cost import CostTracker

    caps = _capabilities(_FakePreset(resources={"cost": CostTracker()}))
    assert caps["cost"] is True


def test_checkpoint_is_a_server_capability(tmp_path: Path) -> None:
    # checkpoint is offered by the RPC server for ANY run (via a per-call
    # checkpoint_dir on run/resume), so it is advertised unconditionally —
    # whether or not the cartridge sets config.checkpoint_dir.
    assert _capabilities(_FakePreset())["checkpoint"] is True
    cfg = LoopConfig(max_steps=5, checkpoint_dir=str(tmp_path))
    assert _capabilities(_FakePreset(config=cfg))["checkpoint"] is True


# ── AC-3: other commands' ready event unchanged (no capabilities) ────


def test_set_backend_ready_has_no_capabilities() -> None:
    events = _drive(
        [
            {"cmd": "set_backend", "factory": "tests.test_rpc_capabilities:make_mock_backend"},
            {"cmd": "quit"},
        ]
    )
    backend_ready = next(e for e in events if e["event"] == "ready" and e.get("backend"))
    assert "capabilities" not in backend_ready


def test_quit_ready_has_no_capabilities() -> None:
    events = _drive([{"cmd": "quit"}])
    quit_ready = next(e for e in events if e["event"] == "ready" and e.get("quit"))
    assert "capabilities" not in quit_ready


def test_stop_reasons_constant_matches_contract() -> None:
    assert list(STOP_REASONS) == EXPECTED_STOP_REASONS
