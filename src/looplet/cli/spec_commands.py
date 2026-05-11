"""``looplet conform`` and ``looplet diff`` and ``looplet describe`` CLIs.

Three small commands that operationalise three of the SPEC.md / paper
promises into something a reader can type and see.

* ``looplet conform [path-to-fixtures]``
    Run cartridge-spec v1.0 conformance fixtures against this
    repository's loader. Prints per-fixture pass/fail and exits
    non-zero on any mismatch. Intended to be runnable against
    *any* loader implementation in the future by importing this
    module and overriding the loader callable.

* ``looplet describe <cartridge-path>``
    Print the structural anatomy of a cartridge: tool surface, hook
    surface, system prompt preview, key config knobs. The "what does
    this agent do?" answer in one screen.

* ``looplet diff <a> <b>``
    Categorical diff of two cartridges: prompt / tools / hooks /
    config / resources. Each change is shown with its category label
    so reviewers can triage by category before reading content.

These are deliberately small (~300 lines combined). The argument they
back up is that the artifact boundary makes these operations routine.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Callable

# ── shared helpers ───────────────────────────────────────────────


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if sys.stdout.isatty() else s


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if sys.stdout.isatty() else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if sys.stdout.isatty() else s


# ── conform ──────────────────────────────────────────────────────


def _default_fixtures_dir() -> Path:
    """Locate the bundled conformance fixtures.

    The fixtures live under ``tests/conformance/fixtures/`` in this
    repository. When looplet is installed via pip the tests directory
    is not shipped, so the fixtures path is required as an argument
    in that case (``looplet conform <path>``).
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "tests" / "conformance" / "fixtures"
        if candidate.is_dir():
            return candidate
    return Path("tests/conformance/fixtures")


def _summarise_preset(preset: Any) -> dict[str, Any]:
    """Reduce a loaded preset to the spec-pinned subset.

    Mirrors :func:`tests.conformance.test_conformance._summarise_preset`
    so the CLI and the in-tree tests stay aligned.
    """
    from looplet.permissions import PermissionHook  # noqa: PLC0415

    cfg = preset.config
    tools_summary = sorted(
        (
            {
                "name": name,
                "requires": list(getattr(preset.tools._tools[name], "requires", []) or []),
            }
            for name in preset.tools.tool_names
        ),
        key=lambda d: d["name"],
    )

    perm_hooks = [h for h in preset.hooks if isinstance(h, PermissionHook)]
    permissions: dict[str, Any] | None = None
    if perm_hooks:
        engine = perm_hooks[0].engine
        permissions = {
            "default": engine.default.value,
            "rules": [
                {
                    "tool": rule.tool,
                    "decision": rule.decision.value,
                    "reason": rule.reason,
                }
                for rule in engine.rules
            ],
        }

    output_schema_fields: list[str] | None = None
    if cfg.output_schema is not None:
        output_schema_fields = sorted(cfg.output_schema.fields)

    model_meta = (cfg.tool_metadata or {}).get("model")

    return {
        "max_steps": cfg.max_steps,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "done_tool": cfg.done_tool,
        "tools": tools_summary,
        "permissions": permissions,
        "output_schema_fields": output_schema_fields,
        "model": model_meta,
        "memory_source_count": len(cfg.memory_sources or []),
    }


def cmd_conform(args: argparse.Namespace) -> int:
    """Run conformance fixtures against the reference loader.

    Returns 0 if every fixture matches its ``expected.json`` summary,
    1 if any mismatched, 2 if no fixtures found.
    """
    from looplet.workspace import workspace_to_preset  # noqa: PLC0415

    fixtures_dir = Path(args.fixtures or _default_fixtures_dir())
    if not fixtures_dir.is_dir():
        print(_red(f"error: fixtures dir not found: {fixtures_dir}"), file=sys.stderr)
        print(
            _dim(
                "    pass a fixtures directory: `looplet conform <path>`. "
                "Sources are under tests/conformance/fixtures/ in the "
                "looplet repository."
            ),
            file=sys.stderr,
        )
        return 2

    fixtures = sorted(p for p in fixtures_dir.iterdir() if p.is_dir())
    if not fixtures:
        print(_red(f"error: no fixtures found under {fixtures_dir}"), file=sys.stderr)
        return 2

    loader: Callable[..., Any] = workspace_to_preset
    print(_bold(f"Cartridge Spec v1.0 conformance — {len(fixtures)} fixture(s)"))
    print(_dim(f"  fixtures: {fixtures_dir}"))
    print()

    failures: list[tuple[str, str]] = []
    for fix in fixtures:
        expected_path = fix / "expected.json"
        cartridge = fix / "cartridge"
        if not expected_path.is_file():
            print(f"  {_yellow('SKIP')} {fix.name} (no expected.json)")
            continue
        if not cartridge.is_dir():
            failures.append((fix.name, "missing cartridge/ subdir"))
            print(f"  {_red('FAIL')} {fix.name} — missing cartridge/")
            continue
        try:
            preset = loader(str(cartridge), strict=True)
        except Exception as e:  # noqa: BLE001
            failures.append((fix.name, f"{type(e).__name__}: {e}"))
            print(f"  {_red('FAIL')} {fix.name} — load error: {type(e).__name__}: {e}")
            continue
        actual = _summarise_preset(preset)
        expected = json.loads(expected_path.read_text())
        if actual != expected:
            failures.append((fix.name, "summary mismatch"))
            print(f"  {_red('FAIL')} {fix.name} — summary mismatch")
            if args.verbose:
                _print_summary_diff(expected, actual)
        else:
            print(f"  {_green('PASS')} {fix.name}")

    print()
    if failures:
        print(_red(f"{len(failures)} failure(s)"))
        for name, why in failures:
            print(f"  - {name}: {why}")
        return 1
    print(_green(f"all {len(fixtures)} fixture(s) passed"))
    return 0


def _print_summary_diff(expected: dict, actual: dict) -> None:
    exp = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    act = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    diff = difflib.unified_diff(exp, act, fromfile="expected", tofile="actual", lineterm="")
    for line in diff:
        if line.startswith("+"):
            print(_green(line))
        elif line.startswith("-"):
            print(_red(line))
        else:
            print(line)


# ── describe ─────────────────────────────────────────────────────


def cmd_describe(args: argparse.Namespace) -> int:
    """Print a one-screen structural summary of a cartridge."""
    from looplet.workspace import workspace_to_preset  # noqa: PLC0415

    cartridge_path = Path(args.cartridge)
    if not cartridge_path.is_dir():
        print(_red(f"error: not a directory: {cartridge_path}"), file=sys.stderr)
        return 2

    try:
        preset = workspace_to_preset(str(cartridge_path), strict=False)
    except Exception as e:  # noqa: BLE001
        print(_red(f"error loading cartridge: {type(e).__name__}: {e}"), file=sys.stderr)
        return 1

    cfg = preset.config
    print(_bold(f"{cartridge_path.name}"))
    if cfg.system_prompt:
        first_line = cfg.system_prompt.strip().splitlines()[0][:80]
        print(_dim(f"  {first_line}"))
    print()

    # Config knobs
    print(_bold("config"))
    print(f"  max_steps        {cfg.max_steps}")
    print(f"  max_tokens       {cfg.max_tokens}")
    print(f"  temperature      {cfg.temperature}")
    print(f"  done_tool        {cfg.done_tool}")
    if cfg.tool_metadata and "model" in cfg.tool_metadata:
        m = cfg.tool_metadata["model"]
        if isinstance(m, dict):
            print(f"  model            {m.get('provider', '?')}/{m.get('name', '?')}")
    print()

    # Tools
    tool_names = sorted(preset.tools.tool_names)
    print(_bold(f"tools ({len(tool_names)})"))
    for name in tool_names:
        spec = preset.tools._tools[name]  # type: ignore[attr-defined]
        desc = (getattr(spec, "description", "") or "").strip().splitlines()
        first = desc[0][:60] if desc else ""
        print(f"  {name:20s} {_dim(first)}")
    print()

    # Hooks
    print(_bold(f"hooks ({len(preset.hooks)})"))
    if not preset.hooks:
        print(_dim("  (none)"))
    for h in preset.hooks:
        print(f"  {type(h).__name__}")
    print()

    # System prompt preview
    if cfg.system_prompt:
        prompt_lines = cfg.system_prompt.strip().splitlines()
        preview = prompt_lines[:8]
        print(_bold("system prompt (first 8 lines)"))
        for line in preview:
            print(f"  {line[:90]}")
        if len(prompt_lines) > 8:
            print(_dim(f"  ... ({len(prompt_lines) - 8} more lines)"))

    return 0


# ── diff ─────────────────────────────────────────────────────────


_CATEGORY_PATHS = (
    ("prompt", ("prompts/",)),
    ("tool", ("tools/",)),
    ("hook", ("hooks/",)),
    ("resource", ("resources/",)),
    ("config", ("config.yaml", "workspace.json", "cartridge.json")),
    ("memory", ("memory/",)),
    ("setup", ("setup.py",)),
)


def _categorize(rel: str) -> str:
    for label, prefixes in _CATEGORY_PATHS:
        for prefix in prefixes:
            if rel == prefix or rel.startswith(prefix):
                return label
    return "other"


def _walk_files(root: Path) -> dict[str, str]:
    """Return ``{relative_path: file_content}`` for every regular file.

    Skips ``__pycache__/`` and ``*.pyc`` so byte-compiled output
    doesn't show up as a diff entry.
    """
    out: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        try:
            out[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            out[rel] = "<binary>"
    return out


def cmd_diff(args: argparse.Namespace) -> int:
    """Categorical diff of two cartridges."""
    a_root = Path(args.a)
    b_root = Path(args.b)
    if not a_root.is_dir() or not b_root.is_dir():
        print(_red("error: both arguments must be directories"), file=sys.stderr)
        return 2

    a_files = _walk_files(a_root)
    b_files = _walk_files(b_root)
    all_paths = sorted(set(a_files) | set(b_files))

    by_category: dict[str, list[tuple[str, str, str]]] = {}
    for rel in all_paths:
        a_text = a_files.get(rel)
        b_text = b_files.get(rel)
        if a_text == b_text:
            continue
        if a_text is None:
            change = "added"
        elif b_text is None:
            change = "removed"
        else:
            change = "modified"
        cat = _categorize(rel)
        by_category.setdefault(cat, []).append((rel, change, _line_delta(a_text, b_text)))

    if not by_category:
        print(_dim("(no changes)"))
        return 0

    print(_bold(f"diff: {a_root.name} -> {b_root.name}"))
    for cat, _ in _CATEGORY_PATHS + (("other", ()),):
        entries = by_category.get(cat, [])
        if not entries:
            continue
        print()
        print(_bold(f"{cat} ({len(entries)})"))
        for rel, change, delta in entries:
            mark = {"added": _green("+"), "removed": _red("-"), "modified": _yellow("~")}[change]
            print(f"  {mark} {rel:50s} {_dim(delta)}")

    if args.show:
        print()
        print(_bold("─── full diff ───"))
        for cat in by_category:
            for rel, change, _delta in by_category[cat]:
                a_text = a_files.get(rel, "")
                b_text = b_files.get(rel, "")
                print()
                print(_bold(f"# {rel} ({change})"))
                if change == "modified":
                    diff = difflib.unified_diff(
                        a_text.splitlines(),
                        b_text.splitlines(),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                        lineterm="",
                    )
                    for line in diff:
                        if line.startswith("+"):
                            print(_green(line))
                        elif line.startswith("-"):
                            print(_red(line))
                        else:
                            print(line)

    return 0


def _line_delta(a: str | None, b: str | None) -> str:
    a_n = len((a or "").splitlines())
    b_n = len((b or "").splitlines())
    if a is None:
        return f"+{b_n}"
    if b is None:
        return f"-{a_n}"
    delta = b_n - a_n
    if delta == 0:
        return "(content)"
    return f"{'+' if delta > 0 else ''}{delta} lines"


# ── argparse wiring (called from looplet.__main__) ───────────────


def add_subparsers(sub: "argparse._SubParsersAction") -> None:
    """Register ``conform``, ``describe``, ``diff`` on the top-level parser."""
    conform_p = sub.add_parser(
        "conform",
        help="Run Cartridge Spec v1.0 conformance fixtures against the loader",
        description=(
            "Run the bundled conformance fixtures (or any directory of them) "
            "against the reference loader and print per-fixture pass/fail. "
            "Exits non-zero on any mismatch."
        ),
    )
    conform_p.add_argument(
        "fixtures",
        nargs="?",
        type=str,
        help="Directory of conformance fixtures (default: bundled tests/conformance/fixtures/)",
    )
    conform_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="On mismatch, print a unified diff between expected and actual",
    )
    conform_p.set_defaults(_handler=cmd_conform)

    describe_p = sub.add_parser(
        "describe",
        help="Print a one-screen structural summary of a cartridge",
        description=(
            "Load a cartridge and print its anatomy: tools, hooks, "
            "config knobs, system prompt preview. Answers 'what does "
            "this agent do?' from a directory listing."
        ),
    )
    describe_p.add_argument("cartridge", type=str, help="Path to a cartridge directory")
    describe_p.set_defaults(_handler=cmd_describe)

    diff_p = sub.add_parser(
        "diff",
        help="Categorical diff between two cartridges (prompt / tool / hook / config / ...)",
        description=(
            "Compare two cartridges and group changes by category. "
            "Each change is labelled by its category before any "
            "content is shown so reviewers can triage by path."
        ),
    )
    diff_p.add_argument("a", type=str, help="Path to the first (baseline) cartridge")
    diff_p.add_argument("b", type=str, help="Path to the second (modified) cartridge")
    diff_p.add_argument(
        "--show",
        action="store_true",
        help="Print the full unified diff under each modified file",
    )
    diff_p.set_defaults(_handler=cmd_diff)


__all__ = ["add_subparsers", "cmd_conform", "cmd_describe", "cmd_diff"]
