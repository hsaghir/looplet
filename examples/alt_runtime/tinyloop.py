#!/usr/bin/env python3
"""``tinyloop`` — a 200-line second runtime for Cartridge Spec v1.0.

This file is intentionally a stand-alone script that does NOT import
:mod:`looplet`. Its only job is to demonstrate that the cartridge
format is genuinely portable: a different loader written from scratch
in dependency-free stdlib Python can load the same cartridge directory
and produce a comparable conformance summary.

Why this exists
---------------

The Cartridge Spec v1.0 (see :file:`SPEC.md` at the repository root)
claims that any conformant runtime can execute a cartridge. That
claim is structurally defensible because the cartridge declares no
runtime, but the only way to *prove* it is to write a second loader
that doesn't share code with the first. This file is that second
loader, kept deliberately small.

What it implements
------------------

* Manifest parsing (``workspace.json`` / ``cartridge.json``).
* Minimal config parsing (a hand-rolled tiny YAML reader covering
  the ``max_steps:`` / ``max_tokens:`` / ``temperature:`` /
  ``done_tool:`` keys the conformance fixtures actually use).
* Tool discovery: scans ``tools/<name>/`` for ``tool.yaml`` +
  ``execute.py`` pairs and dynamically imports the bodies.
* A trivial loop that calls a scripted backend (no LLM provider
  required) and dispatches tool calls to the loaded bodies.
* A ``conformance_summary()`` matching the v1.0 spec-pinned subset
  produced by :file:`tests/conformance/test_conformance.py`.

What it does NOT implement
--------------------------

By design, ``tinyloop`` does not implement: the full LoopHook
protocol, ``extends:`` inheritance, ``resources/<name>.py`` shared
singletons, ``hooks/<name>/``, ``permissions:``, ``model:`` blocks,
``memory/``, ``output_schema:``, hot-reload, native tool calling,
provenance, recovery, or compaction. These are documented loader
extension points in SPEC.md; they are NOT preconditions for the
``identity / shape / portability`` properties that this script is
demonstrating. A v2 of ``tinyloop`` would add them one at a time.

Usage
-----

Run conformance against a fixture::

    python examples/alt_runtime/tinyloop.py conform \\
        tests/conformance/fixtures/01_minimal/cartridge

Run the trivial scripted loop on the same fixture::

    python examples/alt_runtime/tinyloop.py run \\
        tests/conformance/fixtures/01_minimal/cartridge \\
        '{"tool": "done", "args": {"summary": "ok"}}'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ── Minimal YAML reader ─────────────────────────────────────────


_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "" or text in ("null", "~"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def _parse_tiny_yaml(source: str) -> dict[str, Any]:
    """Parse the subset of YAML the conformance fixtures use.

    Supports:
        key: value
        key: { subkey: value, ... }   # inline flow style only
        # comments, blank lines

    Does NOT support: nested block-style mappings, lists, anchors,
    multi-line strings. Sufficient for ``config.yaml`` and
    ``tool.yaml`` in the bundled fixtures; insufficient for the
    full looplet workspace format.
    """
    out: dict[str, Any] = {}
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1]
            sub: dict[str, Any] = {}
            for pair in inner.split(","):
                pair = pair.strip()
                if not pair or ":" not in pair:
                    continue
                k, _, v = pair.partition(":")
                sub[k.strip()] = _parse_scalar(v)
            out[key] = sub
        else:
            out[key] = _parse_scalar(value)
    return out


# ── Cartridge model ─────────────────────────────────────────────


@dataclass
class TinyTool:
    name: str
    description: str
    parameters: dict[str, Any]
    body: Callable[..., dict]


@dataclass
class TinyCartridge:
    name: str
    schema_version: int
    system_prompt: str
    config: dict[str, Any]
    tools: dict[str, TinyTool] = field(default_factory=dict)


# ── Loader ──────────────────────────────────────────────────────


_MANIFEST_NAMES = ("workspace.json", "cartridge.json")


def load_cartridge(root: Path) -> TinyCartridge:
    """Load a cartridge from ``root`` using only stdlib.

    Raises ``FileNotFoundError`` with a structured message naming the
    offending path on missing required files, mirroring the
    ``Loader contract`` clause 7 of SPEC.md ("fail loudly with a
    structured error that names the offending file path").
    """
    if not root.is_dir():
        raise FileNotFoundError(f"cartridge directory not found: {root}")

    manifest_path = next(
        (root / name for name in _MANIFEST_NAMES if (root / name).is_file()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(
            f"cartridge manifest not found at {root / _MANIFEST_NAMES[0]} (or {_MANIFEST_NAMES[1]})"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    config_path = root / "config.yaml"
    config = (
        _parse_tiny_yaml(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    )

    # Cartridge Spec v2: runtime-tier knobs live in a sibling
    # ``runtime.yaml``. Load it (if present) and merge under
    # ``config`` — the resulting flat dict matches v1.x callers'
    # expectations, while v2 cartridges keep the contract / runtime
    # separation on disk.
    runtime_path = root / "runtime.yaml"
    if runtime_path.is_file():
        runtime_kwargs = _parse_tiny_yaml(runtime_path.read_text(encoding="utf-8"))
        # runtime.yaml wins on conflict (it's the host-side override).
        for k, v in runtime_kwargs.items():
            config[k] = v

    system_prompt_path = root / "prompts" / "system.md"
    system_prompt = (
        system_prompt_path.read_text(encoding="utf-8") if system_prompt_path.is_file() else ""
    )

    tools_dir = root / "tools"
    tools: dict[str, TinyTool] = {}
    if tools_dir.is_dir():
        for tool_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
            yaml_path = tool_dir / "tool.yaml"
            exec_path = tool_dir / "execute.py"
            if not yaml_path.is_file() or not exec_path.is_file():
                continue
            meta = _parse_tiny_yaml(yaml_path.read_text(encoding="utf-8"))
            body = _import_execute(exec_path)
            name = str(meta.get("name", tool_dir.name))
            tools[name] = TinyTool(
                name=name,
                description=str(meta.get("description", "")),
                parameters=meta.get("parameters", {}) or {},
                body=body,
            )

    return TinyCartridge(
        name=str(manifest.get("name", root.name)),
        schema_version=int(manifest.get("schema_version", 1)),
        system_prompt=system_prompt,
        config=config,
        tools=tools,
    )


def _import_execute(path: Path) -> Callable[..., dict]:
    """Import ``execute.py`` and return its ``execute`` function.

    Uses a synthetic module name so two cartridges with the same
    tool name don't collide in :data:`sys.modules`.
    """
    mod_name = f"_tinyloop_{path.parent.parent.parent.name}_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "execute"):
        raise AttributeError(f"{path} defines no `execute` function")
    return module.execute  # type: ignore[no-any-return]


# ── Conformance summary ─────────────────────────────────────────


def conformance_summary(cart: TinyCartridge) -> dict[str, Any]:
    """Produce the spec-pinned summary subset for a loaded cartridge.

    Mirrors :func:`tests.conformance.test_conformance._summarise_preset`
    so a v1.0 cartridge produces the same summary on both runtimes.
    Fields ``tinyloop`` does not implement (permissions, output schema,
    model binding, memory) are emitted as ``null`` / ``0`` to match
    the spec's "slot is empty, never ambiguous" stance.
    """
    cfg = cart.config
    return {
        "max_steps": int(cfg.get("max_steps", 15)),
        "max_tokens": int(cfg.get("max_tokens", 2000)),
        "temperature": float(cfg.get("temperature", 0.2)),
        "done_tool": str(cfg.get("done_tool", "done")),
        "tools": sorted(
            ({"name": name, "requires": []} for name in cart.tools),
            key=lambda d: d["name"],
        ),
        "permissions": None,
        "output_schema_fields": None,
        "model": None,
        "memory_source_count": 0,
    }


# ── Trivial loop ────────────────────────────────────────────────


@dataclass
class TinyContext:
    """Stand-in for ``looplet.types.ToolContext``.

    Tools that only read ``ctx.resources`` work unchanged; tools
    that need cancellation or the wider context surface need a
    runtime upgrade. The conformance fixture tools don't.
    """

    resources: dict[str, Any] = field(default_factory=dict)


def run_scripted(cart: TinyCartridge, scripted_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the cartridge against a hard-coded list of tool calls.

    Returns a list of step records: ``{"tool": str, "args": dict, "result": dict, "error": str|None}``.
    Stops on the cartridge's done_tool or when scripted_calls is exhausted.
    """
    done_tool = str(cart.config.get("done_tool", "done"))
    max_steps = int(cart.config.get("max_steps", 15))
    ctx = TinyContext()
    steps: list[dict[str, Any]] = []
    for call in scripted_calls[:max_steps]:
        name = call["tool"]
        args = call.get("args", {})
        if name not in cart.tools:
            steps.append(
                {"tool": name, "args": args, "result": None, "error": f"unknown tool: {name}"}
            )
            continue
        try:
            result = cart.tools[name].body(ctx, **args)
            steps.append({"tool": name, "args": args, "result": result, "error": None})
        except Exception as e:  # noqa: BLE001
            steps.append(
                {"tool": name, "args": args, "result": None, "error": f"{type(e).__name__}: {e}"}
            )
        if name == done_tool:
            break
    return steps


# ── CLI ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tinyloop", description="A second cartridge runtime in stdlib Python."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conform = sub.add_parser(
        "conform", help="Print this loader's conformance summary for a cartridge."
    )
    p_conform.add_argument("cartridge", type=Path)
    p_conform.add_argument("--expected", type=Path, help="Compare against an expected.json")

    p_run = sub.add_parser(
        "run", help="Run a scripted JSON list of tool calls against the cartridge."
    )
    p_run.add_argument("cartridge", type=Path)
    p_run.add_argument("calls", help="JSON object or list: [{tool, args}, ...]")

    args = parser.parse_args(argv)

    cart = load_cartridge(args.cartridge)

    if args.cmd == "conform":
        summary = conformance_summary(cart)
        print(json.dumps(summary, indent=2))
        if args.expected:
            expected = json.loads(args.expected.read_text(encoding="utf-8"))
            if summary != expected:
                print(f"\nMISMATCH against {args.expected}", file=sys.stderr)
                return 1
            print("\nMATCH", file=sys.stderr)
        return 0

    if args.cmd == "run":
        raw = json.loads(args.calls)
        calls = raw if isinstance(raw, list) else [raw]
        steps = run_scripted(cart, calls)
        print(json.dumps(steps, indent=2, default=str))
        return 0

    parser.error("no command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
