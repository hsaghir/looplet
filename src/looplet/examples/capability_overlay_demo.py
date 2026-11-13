"""Capability overlay demo - layer and reload tools at step boundaries.

Demonstrates composing a read-only base cartridge with a restricted writable
overlay directory using ``WorkspaceWatcher``, ``ReloadHook``, ``CachePolicy``,
``PermissionEngine``, ``ProvenanceSink``, and ``replay_loop``.

    python -m looplet.examples.capability_overlay_demo --scripted
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from looplet import (
    CachePolicy,
    DefaultState,
    EvalContext,
    EvalHook,
    MockLLMBackend,
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    ProvenanceSink,
    ReloadHook,
    RunResult,
    WorkspaceWatcher,
    composable_loop,
    preset_to_cartridge,
    replay_loop,
    scaffold_cartridge,
    tool,
)
from looplet.cache import CacheBreakDetector, CacheControl

LEDGER_ROWS: tuple[dict[str, str], ...] = (
    {"id": "inv-101", "region": "NA", "amount": "$1,250.50", "status": "settled"},
    {"id": "inv-102", "region": "EU", "amount": "($200.25)", "status": "refund"},
    {"id": "inv-103", "region": "NA", "amount": "$49.75", "status": "settled"},
    {"id": "inv-104", "region": "APAC", "amount": "$999.00", "status": "void"},
)
EXPECTED_ACTIVE_CENTS = 110_000  # 125050 - 20025 + 4975


def build_base_cartridge(directory: Path) -> Path:
    """Create a minimal schema-v2 base cartridge with read_ledger and done."""
    scaffold_cartridge(directory, name="ledger_analyst", tools=["read_ledger"])
    (directory / "prompts" / "system.md").write_text(
        "You reconcile ledger records. Base tools are read-only; if repeated "
        "parsing is needed, author a focused helper tool in the overlay and "
        "verify its output before calling done.",
        encoding="utf-8",
    )
    (directory / "tools" / "read_ledger" / "tool.yaml").write_text(
        "name: read_ledger\n"
        "description: Return the raw ledger rows for reconciliation.\n"
        "parameters: {}\n"
        "concurrent_safe: true\n",
        encoding="utf-8",
    )
    (directory / "tools" / "read_ledger" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n"
        "    rows = ctx.resources['ledger_rows']\n"
        "    return {'rows': list(rows), 'count': len(rows)}\n",
        encoding="utf-8",
    )
    (directory / "tools" / "done" / "tool.yaml").write_text(
        "name: done\n"
        "description: Submit reconciled active ledger total in integer cents.\n"
        "parameters:\n"
        "  total_cents:\n"
        "    type: integer\n"
        "    description: Net settled minus refund total in cents, excluding void rows.\n"
        "output_schema:\n"
        "  strict: true\n"
        "  fields:\n"
        "    total_cents:\n"
        "      type: int\n"
        "      required: true\n",
        encoding="utf-8",
    )
    (directory / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx, *, total_cents: int) -> dict:\n    return {'total_cents': total_cents}\n",
        encoding="utf-8",
    )
    return directory


def make_overlay_writer(overlay_dir: Path):
    """Return a narrow host tool that writes single-tool definitions into ``overlay_dir``."""

    @tool(
        name="write_overlay_tool",
        description="Write a Python tool into the capability overlay for activation next turn.",
        origin="host",
    )
    def write_overlay_tool(*, tool_name: str, description: str, code: str) -> dict[str, Any]:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,48}", tool_name):
            return {"written": False, "error": f"invalid tool name: {tool_name!r}"}
        tool_dir = overlay_dir / "tools" / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "tool.yaml").write_text(
            f"name: {tool_name}\ndescription: {json.dumps(description)}\nparameters: {{}}\n",
            encoding="utf-8",
        )
        (tool_dir / "execute.py").write_text(code, encoding="utf-8")
        return {"written": True, "tool_name": tool_name}

    return write_overlay_tool


def _native_call(call_id: str, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "id": call_id, "name": name, "input": args}]


def scripted_native_responses() -> list[list[dict[str, Any]]]:
    broken_code = (
        "def execute(ctx -> dict:\n"  # Intentional SyntaxError on first attempt
        "    return {'net_cents': 0}\n"
    )
    fixed_code = (
        "def _parse_cents(raw: str) -> int:\n"
        "    negative = raw.startswith('(') and raw.endswith(')')\n"
        "    cleaned = raw.strip('()$').replace(',', '')\n"
        "    whole, frac = cleaned.split('.')\n"
        "    cents = int(whole) * 100 + int(frac)\n"
        "    return -cents if negative else cents\n\n"
        "def execute(ctx) -> dict:\n"
        "    rows = ctx.resources['ledger_rows']\n"
        "    active = [r for r in rows if r['status'] != 'void']\n"
        "    total = sum(_parse_cents(r['amount']) for r in active)\n"
        "    return {'active_count': len(active), 'net_cents': total}\n"
    )
    return [
        _native_call("c1", "read_ledger", {}),
        _native_call(
            "c2",
            "write_overlay_tool",
            {
                "tool_name": "sum_active_ledger",
                "description": "Parse accounting currency strings and sum non-void rows in cents.",
                "code": broken_code,
            },
        ),
        _native_call(
            "c3",
            "write_overlay_tool",
            {
                "tool_name": "sum_active_ledger",
                "description": "Parse accounting currency strings and sum non-void rows in cents.",
                "code": fixed_code,
            },
        ),
        _native_call("c4", "sum_active_ledger", {}),
        _native_call("c5", "done", {"total_cents": EXPECTED_ACTIVE_CENTS}),
    ]


class _NativeScriptedBackend:
    """Deterministic native tool-calling backend for zero-network mechanism demos."""

    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.observed_tool_lists: list[list[str]] = []

    def generate(self, prompt: str, **_: Any) -> str:
        raise AssertionError("demo uses native tool calling")

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        **_: Any,
    ) -> list[dict[str, Any]]:
        del prompt
        self.observed_tool_lists.append([entry["name"] for entry in tools])
        response = self._responses[self._index]
        self._index += 1
        return response


def eval_exact_total(ctx: EvalContext) -> bool:
    return ctx.completed and ctx.final_output == {"total_cents": EXPECTED_ACTIVE_CENTS}


def run_demo(work_dir: Path) -> dict[str, Any]:
    base_dir = build_base_cartridge(work_dir / "base.cartridge")
    overlay_dir = work_dir / "run_overlay"
    trace_dir = work_dir / "trace"
    graduated_dir = work_dir / "graduated.cartridge"

    watcher = WorkspaceWatcher(
        base_dir,
        overlays=[overlay_dir],
        allowed_overlay_slots={"tools", "helpers"},
        strict=True,
    )
    with watcher:
        preset = watcher.preset()
        preset.config.max_steps = 8
        preset.resources["ledger_rows"] = LEDGER_ROWS
        preset.tools.set_resources(preset.resources)
        preset.tools.register(make_overlay_writer(overlay_dir))
        preset.config.cache_policy = CachePolicy(
            system_prompt=CacheControl(),
            tool_schemas=CacheControl(),
        )

        permissions = (
            PermissionEngine(default=PermissionDecision.DENY)
            .allow_origin(preset.tools, "cartridge")
            .allow_origin(preset.tools, "host")
            .allow_origin(preset.tools, "overlay")
        )
        cache_detector = CacheBreakDetector(preset.config.cache_policy)
        reload_hook = ReloadHook(watcher)
        sink = ProvenanceSink(trace_dir)
        backend = _NativeScriptedBackend(scripted_native_responses())
        llm = sink.wrap_llm(backend)

        state = DefaultState(max_steps=8)
        with closing(
            composable_loop(
                llm=llm,
                tools=preset.tools,
                state=state,
                config=preset.config,
                hooks=[
                    *preset.hooks,
                    reload_hook,
                    PermissionHook(permissions),
                    cache_detector,
                    sink.trajectory_hook(),
                    EvalHook(evaluators=[eval_exact_total]),
                ],
                task={"goal": "Compute active ledger total in cents, excluding void rows."},
            )
        ) as steps:
            for _ in steps:
                pass
        sink.flush()

        # Graduate the validated overlay preset into a standalone v2 cartridge.
        preset_to_cartridge(watcher.preset(), graduated_dir, strict=True)

    # Replay captured responses against a fresh blank overlay directory.
    replay_overlay = work_dir / "replay_overlay"
    replay_watcher = WorkspaceWatcher(
        base_dir,
        overlays=[replay_overlay],
        allowed_overlay_slots={"tools", "helpers"},
        strict=True,
    )
    with replay_watcher:
        replay_preset = replay_watcher.preset()
        replay_preset.config.max_steps = 8
        replay_preset.resources["ledger_rows"] = LEDGER_ROWS
        replay_preset.tools.set_resources(replay_preset.resources)
        replay_preset.tools.register(make_overlay_writer(replay_overlay))
        replay_reload = ReloadHook(replay_watcher)
        replay_state = DefaultState(max_steps=8)
        with closing(
            replay_loop(
                trace_dir,
                tools=replay_preset.tools,
                state=replay_state,
                config=replay_preset.config,
                hooks=[*replay_preset.hooks, replay_reload],
            )
        ) as replay_steps:
            for _ in replay_steps:
                pass

    trajectory = json.loads((trace_dir / "trajectory.json").read_text(encoding="utf-8"))
    result = RunResult.from_state(state)
    replay_result = RunResult.from_state(replay_state)
    return {
        "completed": result.status == "completed",
        "output": result.output,
        "reloads": [obs.to_dict() for obs in reload_hook.observations],
        "cache_breaks": len(cache_detector.breaks),
        "harness_epochs": trajectory["metadata"]["harness_epochs"],
        "observed_tool_lists": backend.observed_tool_lists,
        "graduated_tools": sorted(p.name for p in (graduated_dir / "tools").iterdir()),
        "replay_completed": replay_result.status == "completed",
        "replay_output": replay_result.output,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the capability overlay mechanism demo.")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run the deterministic zero-network demo.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output directory.")
    args = parser.parse_args(argv)
    if not args.scripted:
        # Keep CLI consistent with hello_world; non-scripted mode uses MockLLMBackend fallback hint.
        _ = MockLLMBackend
        parser.error("Pass --scripted to run the deterministic capability overlay demo.")

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        summary = run_demo(args.output)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    else:
        with tempfile.TemporaryDirectory(prefix="looplet-overlay-demo-") as tmp:
            summary = run_demo(Path(tmp))

    print(json.dumps(summary, indent=2))
    return 0 if summary["completed"] and summary["replay_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
