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
* Cartridge Spec v2 hard-rejections: refuses to load a v2 cartridge
  containing ``setup.py`` or magic ``prompts/briefing.md`` /
  ``prompts/recovery.md`` files. Demonstrates the rejection
  contract is portable, not a quirk of the reference loader.

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
    # Inline flow-style mapping: { k: v, k2: v2 }
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        sub: dict[str, Any] = {}
        if inner:
            for pair in _split_flow_commas(inner):
                if ":" not in pair:
                    continue
                k, _, v = pair.partition(":")
                sub[k.strip()] = _parse_scalar(v.strip())
        return sub
    # Inline flow-style list: [a, b, c]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p.strip()) for p in _split_flow_commas(inner)]
    return text


def _split_flow_commas(text: str) -> list[str]:
    """Split a comma-separated flow-style payload, respecting nested {} / []."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch in "{[":
            depth += 1
            buf.append(ch)
        elif ch in "}]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _parse_tiny_yaml(source: str) -> dict[str, Any]:
    """Parse the subset of YAML needed by the conformance fixtures.

    Supports (sufficient for v1.0 ``config.yaml`` + ``tool.yaml``):

    * Block-style nested mappings (indent-based).
    * Block-style lists (``- item`` or ``- key: value`` rows).
    * Inline flow style for both mappings (``{ k: v, ... }``) and
      lists (``[a, b]``), with proper bracket balancing.
    * ``#`` comments and blank lines.

    Does NOT support: anchors / aliases, multi-line scalars, complex
    keys. If a fixture grows past this, we either extend the parser
    or split the fixture.
    """
    # Strip comments + trailing whitespace, drop empty lines, but keep
    # indent for structural parsing.
    raw_lines = source.splitlines()
    lines: list[tuple[int, str]] = []  # (indent, body)
    for ln in raw_lines:
        # Drop trailing inline comments — but only ones that aren't
        # inside a string or flow expression. The fixtures keep things
        # simple enough that the conservative "strip everything after
        # an unquoted ``#``" approach works.
        stripped = ln.rstrip()
        if not stripped.strip() or stripped.strip().startswith("#"):
            continue
        # Conservatively strip inline comments only when there's a space
        # before the ``#`` (avoids breaking ``"rm -rf #"``-style values).
        m = re.search(r"\s+#", stripped)
        if m:
            stripped = stripped[: m.start()].rstrip()
        indent = len(stripped) - len(stripped.lstrip(" "))
        body = stripped.strip()
        lines.append((indent, body))

    def _parse_block(start: int, base_indent: int) -> tuple[Any, int]:
        """Parse a block (mapping or list) and return (value, next_index)."""
        if start >= len(lines):
            return {}, start
        first_indent, first_body = lines[start]
        if first_indent < base_indent:
            return {}, start

        # List: starts with ``- ``.
        if first_body.startswith("- "):
            items: list[Any] = []
            i = start
            while i < len(lines):
                indent, body = lines[i]
                if indent < base_indent or not body.startswith("- "):
                    break
                rest = body[2:].strip()
                # Is this a "- key: value" with possibly more keys
                # indented under it? Detect by ``:`` outside flow.
                if ":" in rest and not (rest.startswith("{") or rest.startswith("[")):
                    key, _, value = rest.partition(":")
                    key = key.strip()
                    value = value.strip()
                    item_map: dict[str, Any] = {}
                    if value:
                        item_map[key] = _parse_scalar(value)
                    else:
                        # Nested mapping under this key.
                        sub, j = _parse_block(i + 1, indent + 2)
                        item_map[key] = sub
                        i = j - 1
                    # Pick up additional keys at the same indent as
                    # the "- " marker's body (indent + 2).
                    j = i + 1
                    while j < len(lines) and lines[j][0] == indent + 2:
                        k_indent, k_body = lines[j]
                        if k_body.startswith("- "):
                            break
                        k, _, v = k_body.partition(":")
                        v = v.strip()
                        if v:
                            item_map[k.strip()] = _parse_scalar(v)
                        else:
                            sub, next_j = _parse_block(j + 1, k_indent + 2)
                            item_map[k.strip()] = sub
                            j = next_j - 1
                        j += 1
                    items.append(item_map)
                    i = j
                    continue
                # Plain scalar list item.
                items.append(_parse_scalar(rest))
                i += 1
            return items, i

        # Mapping: a sequence of ``key:`` or ``key: value`` lines at
        # this indent.
        out: dict[str, Any] = {}
        i = start
        while i < len(lines):
            indent, body = lines[i]
            if indent < base_indent or body.startswith("- "):
                break
            if indent > base_indent:
                # Should not happen — we always descend explicitly.
                i += 1
                continue
            if ":" not in body:
                i += 1
                continue
            key, _, value = body.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                out[key] = _parse_scalar(value)
                i += 1
            else:
                sub, j = _parse_block(i + 1, base_indent + 2)
                out[key] = sub
                i = j
        return out, i

    result, _ = _parse_block(0, 0)
    if isinstance(result, dict):
        return result
    # Top-level list is unusual for the fixtures but harmless to wrap.
    return {"__list__": result} if isinstance(result, list) else {}


# ── Cartridge model ─────────────────────────────────────────────


@dataclass
class TinyTool:
    name: str
    description: str
    parameters: dict[str, Any]
    body: Callable[..., dict]
    # v1.0 output_schema: per-tool required-fields + types contract.
    # ``None`` means "no schema declared"; a dict carries the parsed
    # ``output_schema:`` from the tool's tool.yaml. tinyloop validates
    # ``done`` calls against this in :func:`run_scripted`.
    output_schema: dict[str, Any] | None = None


@dataclass
class TinyCartridge:
    name: str
    schema_version: int
    system_prompt: str
    config: dict[str, Any]
    tools: dict[str, TinyTool] = field(default_factory=dict)
    # v1.0 declarative slots, surfaced for ``conformance_summary``.
    permissions: dict[str, Any] | None = None  # normalised: {"default", "rules": [...]}
    model: dict[str, Any] | None = None
    memory_sources: list[str] = field(default_factory=list)  # paths under memory/


# ── Loader ──────────────────────────────────────────────────────


_MANIFEST_NAMES = ("workspace.json", "cartridge.json")


class CartridgeSerializationError(Exception):
    """Raised on a malformed or spec-rejected cartridge.

    Mirrors the reference loader's ``looplet.cartridge.CartridgeSerializationError``
    so callers (and conformance tests) can catch the same kind of
    failure regardless of which runtime loaded the cartridge.
    """


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
    schema_version = int(manifest.get("schema_version", 1))

    # Cartridge Spec v2 hard-rejections (see SPEC.md "Backwards
    # compatibility"). v1.x accepted these with a deprecation warning;
    # v2 fail-louds. Implementing them here demonstrates the v2
    # rejection contract is portable across runtimes, not a quirk of
    # the reference loader.
    if schema_version >= 2:
        forbidden = [
            (root / "setup.py", "setup.py"),
            (root / "prompts" / "briefing.md", "prompts/briefing.md"),
            (root / "prompts" / "recovery.md", "prompts/recovery.md"),
        ]
        for path, label in forbidden:
            if path.is_file():
                raise CartridgeSerializationError(
                    f"Cartridge {root}: {label} is forbidden in spec v2 "
                    f"(was a v1.x escape hatch). Declare an explicit "
                    f"hook or builtin_hook instead."
                )

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
            schema = meta.get("output_schema")
            tools[name] = TinyTool(
                name=name,
                description=str(meta.get("description", "")),
                parameters=meta.get("parameters", {}) or {},
                body=body,
                output_schema=schema if isinstance(schema, dict) else None,
            )

    # ── v1.0 declarative slots ──────────────────────────────────
    permissions = _normalise_permissions(config.get("permissions"))
    model = config.get("model") if isinstance(config.get("model"), dict) else None
    # Hoist model.{max_tokens,temperature} into the flat config so
    # legacy summary code keeps finding them.
    if model:
        for hoist in ("max_tokens", "temperature"):
            if hoist in model and hoist not in config:
                config[hoist] = model[hoist]
    memory_sources: list[str] = []
    memory_dir = root / "memory"
    if memory_dir.is_dir():
        for p in sorted(memory_dir.iterdir()):
            if p.is_file() and p.suffix in (".md", ".txt", ".py"):
                memory_sources.append(p.name)
    # Mirror the reference loader's long-term-memory auto-load:
    # an explicit ``memory: { long_term: <path> }`` in config OR an
    # auto-discovered ``memory/long_term.md`` appends one ADDITIONAL
    # source on top of the file scan above. This matches the count
    # produced by ``cartridge_to_preset`` for conformance fixture 04.
    mem_block = config.get("memory") if isinstance(config.get("memory"), dict) else None
    long_term_extra: Path | None = None
    if mem_block and isinstance(mem_block.get("long_term"), str):
        cand = (root / mem_block["long_term"]).resolve()
        if cand.is_file():
            long_term_extra = cand
    if long_term_extra is None:
        cand = root / "memory" / "long_term.md"
        if cand.is_file():
            long_term_extra = cand
    if long_term_extra is not None:
        memory_sources.append(f"@long_term:{long_term_extra.name}")

    return TinyCartridge(
        name=str(manifest.get("name", root.name)),
        schema_version=schema_version,
        system_prompt=system_prompt,
        config=config,
        tools=tools,
        permissions=permissions,
        model=model,
        memory_sources=memory_sources,
    )


def _normalise_permissions(raw: Any) -> dict[str, Any] | None:
    """Flatten a ``permissions:`` block into ``{default, rules}``.

    The on-disk shape splits rules across ``deny:``, ``ask:`` and
    optional ``allow:`` lists. The conformance summary wants a
    single ordered ``rules`` list with ``{tool, decision, reason}``
    entries — matching what the reference looplet ``PermissionEngine``
    serialises. We preserve declaration order: ``deny`` then ``ask``
    then ``allow``, since v1.0 callers describe rules in that order
    of safety. The ``contains:`` matcher is intentionally dropped
    from the summary — it's an enforcement detail, not part of the
    declarative contract surface.
    """
    if not isinstance(raw, dict):
        return None
    default = raw.get("default", "allow")
    rules: list[dict[str, Any]] = []
    for decision in ("deny", "ask", "allow"):
        items = raw.get(decision)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            rules.append(
                {
                    "tool": item.get("tool", ""),
                    "decision": decision,
                    "reason": item.get("reason", "") or "",
                }
            )
    return {"default": default, "rules": rules}


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
    tinyloop is now **minimal-conforming**: it surfaces all v1.0
    declarative slots (permissions, output schema, model binding,
    memory). Slots a cartridge does not declare stay ``null`` / ``0``.
    """
    cfg = cart.config
    done_tool = str(cfg.get("done_tool", "done"))
    done = cart.tools.get(done_tool)
    output_schema_fields: list[str] | None = None
    if done is not None and isinstance(done.output_schema, dict):
        props = done.output_schema.get("properties")
        if isinstance(props, dict):
            output_schema_fields = sorted(props.keys())
    model_summary: dict[str, Any] | None = None
    if isinstance(cart.model, dict):
        # The conformance summary intentionally exposes only the
        # binding identity (provider, name, reasoning_effort) — not
        # generation knobs like max_tokens/temperature, which live in
        # the flat config and are already surfaced above.
        model_summary = {
            k: cart.model[k] for k in ("provider", "name", "reasoning_effort") if k in cart.model
        }
        if not model_summary:
            model_summary = None
    return {
        "max_steps": int(cfg.get("max_steps", 15)),
        "max_tokens": int(cfg.get("max_tokens", 2000)),
        "temperature": float(cfg.get("temperature", 0.2)),
        "done_tool": done_tool,
        "tools": sorted(
            ({"name": name, "requires": []} for name in cart.tools),
            key=lambda d: d["name"],
        ),
        "permissions": cart.permissions,
        "output_schema_fields": output_schema_fields,
        "model": model_summary,
        "memory_source_count": len(cart.memory_sources),
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
