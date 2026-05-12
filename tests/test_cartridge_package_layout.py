"""Pin the cartridge package layout and dependency direction.

The cartridge moved from sibling modules to a self-contained
:mod:`looplet.cartridge` package. These tests pin two invariants
that protect that work:

1. **Layout**: the package re-exports every previously-public
   symbol from its old import path.
2. **Dependency direction**: the cartridge package depends only on
   *public* looplet types (LoopConfig, AgentPreset, ToolSpec,
   LoopHook, OutputSchema, MemorySource, ...) — never on internal
   machinery (loop body, scaffolding helpers, recovery internals,
   async loop, context).

If a future change adds a dep from the cartridge package to e.g.
``looplet.scaffolding`` (the LLM-call retry helpers), this test
fails — keeping the cartridge separable from the loop runtime in
preparation for a future ``looplet-cartridge`` package extraction.
"""

from __future__ import annotations

import pkgutil
from pathlib import Path

import pytest

import looplet
import looplet.cartridge

# ── Layout: package re-exports the historical surface ────────────


def test_cartridge_package_exports_canonical_names() -> None:
    """``from looplet.cartridge import X`` works for every X that was
    previously importable from ``looplet.cartridge`` as a module."""
    for name in (
        "Cartridge",
        "CartridgeLayout",
        "CartridgeSerializationError",
        "cartridge_to_preset",
        "preset_to_cartridge",
        "resource_ref_for",
        "SCHEMA_VERSION",
        # back-compat aliases
        "Workspace",
        "WorkspaceLayout",
        "WorkspaceSerializationError",
        "workspace_to_preset",
        "preset_to_workspace",
    ):
        assert hasattr(looplet.cartridge, name), f"looplet.cartridge no longer exports {name!r}"


def test_cartridge_subpackages_importable() -> None:
    """The four submodules moved into the cartridge package are
    importable from their canonical paths."""
    import looplet.cartridge.hot_reload
    import looplet.cartridge.prompt_files
    import looplet.cartridge.scaffold
    import looplet.cartridge.spec_slots

    assert hasattr(looplet.cartridge.scaffold, "scaffold_cartridge")
    assert hasattr(looplet.cartridge.spec_slots, "compile_model_block")
    assert hasattr(looplet.cartridge.spec_slots, "compile_permissions_block")
    assert hasattr(looplet.cartridge.spec_slots, "compile_output_schema")
    assert hasattr(looplet.cartridge.prompt_files, "StaticBriefingHook")
    assert hasattr(looplet.cartridge.prompt_files, "RecoveryHintHook")
    assert hasattr(looplet.cartridge.hot_reload, "WorkspaceWatcher")


def test_back_compat_shim_paths_still_work() -> None:
    """The historical ``looplet.scaffold`` etc. paths must keep
    re-exporting their public symbols so existing user code doesn't
    break."""
    import looplet.hot_reload
    import looplet.prompt_files
    import looplet.scaffold
    import looplet.spec_slots

    # Same objects (``is``-equal), not different copies.
    from looplet.cartridge.hot_reload import WorkspaceWatcher as canonical_ww
    from looplet.cartridge.prompt_files import StaticBriefingHook as canonical_sb
    from looplet.cartridge.scaffold import scaffold_cartridge as canonical_scaffold
    from looplet.cartridge.spec_slots import (
        compile_model_block as canonical_compile_model,
    )

    assert looplet.scaffold.scaffold_cartridge is canonical_scaffold
    assert looplet.spec_slots.compile_model_block is canonical_compile_model
    assert looplet.prompt_files.StaticBriefingHook is canonical_sb
    assert looplet.hot_reload.WorkspaceWatcher is canonical_ww


# ── Dependency direction: cartridge → public types only ──────────


# Allowed: cartridge submodules + the small set of looplet public
# types the cartridge needs to construct an AgentPreset.
ALLOWED_LOOPLET_DEPS = frozenset(
    {
        "looplet.cartridge",  # internal
        # public types / registries the cartridge MAY depend on:
        "looplet.builtin_hooks",
        "looplet.builtin_tools",
        "looplet.compact",  # public CompactService types
        "looplet.hook_decision",  # public HookDecision factories
        "looplet.loop",  # public LoopConfig / hook protocol
        "looplet.memory",  # public MemorySource types
        "looplet.permissions",  # public PermissionEngine etc.
        "looplet.presets",  # public AgentPreset
        "looplet.refs",  # resource ref registry
        "looplet.tools",  # public ToolSpec / BaseToolRegistry
        "looplet.types",  # public DefaultState etc.
        "looplet.validation",  # public OutputSchema / FieldSpec
    }
)

# Deps that would indicate the cartridge has crept into runtime
# internals and the separation is broken:
FORBIDDEN_LOOPLET_DEPS = frozenset(
    {
        "looplet.async_loop",  # async loop body (runtime)
        "looplet.context",  # session-entry trimming (runtime helper)
        "looplet.recovery",  # recovery registry internals
        "looplet.recovery_strategies",  # recovery internals
        "looplet.scaffolding",  # LLM-call retry / truncation (runtime)
        "looplet.budget",  # context-budget runtime helpers
        "looplet.streaming",  # streaming hook (runtime)
        "looplet.telemetry",  # tracer (runtime)
        "looplet.evals",  # adjacent artifact, NOT the cartridge
        "looplet.provenance",  # adjacent artifact, NOT the cartridge
        "looplet.session",  # runtime data structure
        "looplet.checkpoint",  # runtime persistence
        "looplet.cache",  # runtime caching
        "looplet.router",  # runtime routing
        "looplet.subagent",  # runtime sub-loop
        "looplet.skills",  # adjacent artifact (per paper)
        "looplet.bundles",  # adjacent artifact
        "looplet.rpc",  # JSONL stdio (runtime)
        "looplet.parse",  # LLM response parsing (runtime)
        "looplet.history",  # turn/step history (runtime)
        "looplet.conversation",  # message thread (runtime)
    }
)


def _walk_cartridge_imports() -> set[tuple[str, str]]:
    """Return ``{(consumer_module, imported_module), ...}`` for every
    ``from looplet.X import ...`` or ``import looplet.X`` line in the
    cartridge package, including module-level and lazy/inside-function
    imports.
    """
    import re

    cartridge_root = Path(looplet.cartridge.__file__).parent
    pattern = re.compile(r"(?:from\s+(looplet\.[\w.]+)\s+import|import\s+(looplet\.[\w.]+))")

    edges: set[tuple[str, str]] = set()
    for py in cartridge_root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        rel = "looplet.cartridge"
        if py.stem != "__init__":
            rel = f"looplet.cartridge.{py.stem}"
        for m in pattern.finditer(text):
            target = m.group(1) or m.group(2)
            # Normalise to top-level looplet.X (drop deeper attrs).
            top2 = ".".join(target.split(".")[:2])
            edges.add((rel, top2))
    return edges


def test_cartridge_package_depends_only_on_public_types() -> None:
    """Every external looplet dep of the cartridge package must be on
    the public-type allowlist."""
    edges = _walk_cartridge_imports()
    bad: list[tuple[str, str]] = []
    for consumer, target in edges:
        if target == "looplet":
            continue  # umbrella import; usually re-exports
        if target.startswith("looplet.cartridge"):
            continue  # internal
        if target not in ALLOWED_LOOPLET_DEPS:
            bad.append((consumer, target))
    assert not bad, (
        f"cartridge package has {len(bad)} dep(s) outside the allowed public-type "
        f"surface: {bad}. Either the new dep belongs on ALLOWED_LOOPLET_DEPS "
        f"(if it's a public type) or it's a runtime-internal dep that breaks "
        f"the cartridge/runtime separation."
    )


def test_cartridge_package_does_not_depend_on_runtime_internals() -> None:
    """Belt-and-suspenders: explicitly check we don't depend on any
    of the known runtime-internal modules."""
    edges = _walk_cartridge_imports()
    bad: list[tuple[str, str]] = []
    for consumer, target in edges:
        if target in FORBIDDEN_LOOPLET_DEPS:
            bad.append((consumer, target))
    assert not bad, (
        f"cartridge package depends on runtime-internal module(s) it shouldn't: "
        f"{bad}. The cartridge/runtime separation requires the cartridge to "
        f"depend only on public looplet types."
    )


def test_cartridge_package_does_not_self_import_via_umbrella() -> None:
    """The cartridge package must not import ``from looplet import X``
    at module top level (creates a circular dep with the package
    surface). Lazy/inside-function umbrella imports are fine."""
    import re

    cartridge_root = Path(looplet.cartridge.__file__).parent
    bad: list[tuple[str, int, str]] = []
    for py in cartridge_root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 0:
                continue  # inside a function or class — fine
            if re.match(r"from\s+looplet\s+import\b", stripped):
                bad.append(
                    (str(py.relative_to(cartridge_root.parent.parent)), lineno, line.strip())
                )
    assert not bad, (
        f"cartridge submodule(s) import from the looplet umbrella at module top "
        f"level: {bad}. Use the canonical submodule path "
        f"(``from looplet.hook_decision import InjectContext``, etc.) instead."
    )
