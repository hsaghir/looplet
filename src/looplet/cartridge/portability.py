"""Whole-cartridge portability report.

``looplet.hook_contract.classify`` answers the question *"is this one
hook portable across runtimes?"*. This module lifts that question to the
**whole cartridge**: *"can this agent package run on a non-Python loader
(Rust/Go/TypeScript), and if not, exactly which components pin it to a
Python host?"*

It is a **static** analyser — it reads the cartridge directory and its
``config.yaml`` / ``runtime.yaml`` rather than importing any Python
bodies. That keeps it dependency-free and lets it grade cartridges whose
tool/resource code can't even be imported in the current environment
(e.g. ``looplet-tax`` with its own deps).

Four portability tiers, mirroring the conformance model in
``HOOK_CARTRIDGE_DESIGN.md``:

* :data:`PROTOCOL` — pure data or out-of-process protocol. Runs on *any*
  conforming loader with no shared code: ``config.yaml``, ``prompts/``,
  ``mcp_servers:`` tools, ``kind: lep`` hooks.
* :data:`STDLIB` — declarative reference to a looplet-shipped archetype
  (``builtin_tools:`` / ``builtin_hooks:``). No Python body lives in the
  cartridge; portable to any runtime that ships the same named stdlib.
* :data:`RUNTIME` — a ``resources/*.py`` whose only job is to wrap a
  host-provided builtin *service* (compaction, the skill manager, …) via
  a looplet factory such as ``default_compact_service``. The service is a
  host responsibility every conforming loader ships its own equivalent
  of, so it does NOT pin the cartridge to Python — not a blocker.
* :data:`INPROCESS` — a Python body or shared in-process object that
  pins the cartridge to a Python host: ``tools/<n>/execute.py``,
  single-file ``tools/<n>.py``, ``hooks/<n>/`` class hooks, and
  author-owned ``resources/*.py`` (``@ref`` shared mutable state).

A cartridge is in the **portable profile** when it has *no* INPROCESS
components (everything is PROTOCOL, STDLIB, or RUNTIME host services);
otherwise it is in the **python-host profile** and the INPROCESS
components are the exact blockers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from looplet.cartridge._yaml import _load_yaml

__all__ = [
    "INPROCESS",
    "PROTOCOL",
    "RUNTIME",
    "STDLIB",
    "ComponentReport",
    "CartridgePortabilityReport",
    "analyse_cartridge",
]

# Portability tiers (best → worst for cross-runtime portability).
PROTOCOL = "protocol"
STDLIB = "stdlib"
# RUNTIME: a host-provided builtin service (compaction, skill manager, …)
# wrapped declaratively. Any conforming loader ships its own equivalent,
# so it does NOT pin the cartridge to a Python host — not a blocker.
RUNTIME = "runtime"
INPROCESS = "inprocess"

# Builtin host-service factories. A resource whose only job is to call
# one of these is a RUNTIME-tier component (host responsibility), not an
# author-owned shared-state blocker.
_RUNTIME_SERVICE_FACTORIES = frozenset(
    {
        "default_compact_service",
        "build_skill_manager_for_workspace",
    }
)

# Profile verdicts.
PROFILE_PORTABLE = "portable"
PROFILE_PYTHON_HOST = "python-host"


@dataclass(frozen=True)
class ComponentReport:
    """Portability classification of one cartridge component."""

    kind: str  # "tool" | "hook" | "resource" | "config" | "prompts"
    name: str
    tier: str  # PROTOCOL | STDLIB | INPROCESS
    detail: str = ""  # e.g. "mcp", "lep", "python-execute", "class", "shared-ref"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_blocker(self) -> bool:
        """True when this component pins the cartridge to a Python host."""
        return self.tier == INPROCESS


@dataclass(frozen=True)
class CartridgePortabilityReport:
    """Whole-cartridge portability verdict + per-component breakdown."""

    root: Path
    name: str
    components: tuple[ComponentReport, ...] = field(default_factory=tuple)

    @property
    def profile(self) -> str:
        """``portable`` if nothing pins it to Python, else ``python-host``."""
        return (
            PROFILE_PYTHON_HOST if any(c.is_blocker for c in self.components) else PROFILE_PORTABLE
        )

    @property
    def blockers(self) -> tuple[ComponentReport, ...]:
        """The INPROCESS components that force the python-host profile."""
        return tuple(c for c in self.components if c.is_blocker)

    def by_tier(self, tier: str) -> tuple[ComponentReport, ...]:
        return tuple(c for c in self.components if c.tier == tier)

    def counts(self) -> dict[str, int]:
        out = {PROTOCOL: 0, STDLIB: 0, RUNTIME: 0, INPROCESS: 0}
        for c in self.components:
            out[c.tier] = out.get(c.tier, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "profile": self.profile,
            "counts": self.counts(),
            "components": [
                {
                    "kind": c.kind,
                    "name": c.name,
                    "tier": c.tier,
                    "detail": c.detail,
                    "reasons": list(c.reasons),
                }
                for c in self.components
            ],
        }

    def render(self) -> str:
        """Human-readable multi-line report."""
        counts = self.counts()
        lines = [
            f"Cartridge: {self.name}  ({self.root})",
            f"Profile:   {self.profile.upper()}",
            (
                f"Tiers:     protocol={counts[PROTOCOL]}  "
                f"stdlib={counts[STDLIB]}  runtime={counts[RUNTIME]}  "
                f"inprocess={counts[INPROCESS]}"
            ),
            "",
        ]
        if self.profile == PROFILE_PORTABLE:
            lines.append(
                "  ✔ No Python-pinned components — runs on any conforming loader (Rust/Go/TS)."
            )
        else:
            lines.append("  Python-host blockers (pin the cartridge to Python):")
            for c in self.blockers:
                why = f" — {c.reasons[0]}" if c.reasons else ""
                lines.append(f"    ✗ {c.kind}:{c.name} [{c.detail}]{why}")
        lines.append("")
        lines.append("  Components:")
        symbol = {PROTOCOL: "●", STDLIB: "◐", RUNTIME: "◈", INPROCESS: "○"}
        for c in self.components:
            lines.append(
                f"    {symbol.get(c.tier, '?')} {c.tier:<9} {c.kind}:{c.name} [{c.detail}]"
            )
        return "\n".join(lines)


# ── Static walkers ────────────────────────────────────────────


def _read_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = _load_yaml(path.read_text(encoding="utf-8"), source_path=path)
    except Exception:  # noqa: BLE001 — malformed config shouldn't crash analysis
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _analyse_tools(root: Path, mcp_servers: dict[str, Any]) -> list[ComponentReport]:
    out: list[ComponentReport] = []

    # MCP servers → protocol-tier tool providers (out-of-process, any runtime).
    for srv_name in mcp_servers:
        out.append(
            ComponentReport(
                kind="tool",
                name=f"mcp:{srv_name}",
                tier=PROTOCOL,
                detail="mcp",
                reasons=(
                    "out-of-process MCP server — tool body runs over the "
                    "stdio protocol, no Python required by the loader",
                ),
            )
        )

    tools_dir = root / "tools"
    if tools_dir.is_dir():
        # Single-file tools: tools/<name>.py (Python body → inprocess).
        for p in sorted(tools_dir.iterdir()):
            if p.is_file() and p.suffix == ".py" and not p.name.startswith("_"):
                out.append(
                    ComponentReport(
                        kind="tool",
                        name=p.stem,
                        tier=INPROCESS,
                        detail="python-single-file",
                        reasons=(
                            "single-file Python tool body — pinned to a "
                            "Python host (port to an MCP server for "
                            "cross-runtime portability)",
                        ),
                    )
                )
        # Multi-file tools: tools/<name>/execute.py (Python body → inprocess).
        for d in sorted(tools_dir.iterdir()):
            if d.is_dir() and (d / "execute.py").is_file():
                out.append(
                    ComponentReport(
                        kind="tool",
                        name=d.name,
                        tier=INPROCESS,
                        detail="python-execute",
                        reasons=(
                            "Python execute.py tool body — pinned to a "
                            "Python host (port to an MCP server for "
                            "cross-runtime portability)",
                        ),
                    )
                )
    return out


def _analyse_hooks(root: Path, builtin_hooks: list[Any]) -> list[ComponentReport]:
    out: list[ComponentReport] = []

    # builtin_hooks: declarative stdlib references → stdlib tier.
    for entry in builtin_hooks:
        if isinstance(entry, str):
            hook_name = entry
        elif isinstance(entry, dict) and entry:
            hook_name = next(iter(entry))
        else:
            hook_name = str(entry)
        out.append(
            ComponentReport(
                kind="hook",
                name=hook_name,
                tier=STDLIB,
                detail="builtin",
                reasons=(
                    "declarative looplet stdlib hook — portable to any "
                    "runtime that ships the same named archetype",
                ),
            )
        )

    hooks_dir = root / "hooks"
    if hooks_dir.is_dir():
        for d in sorted(hooks_dir.iterdir()):
            if not d.is_dir():
                continue
            cfg = _read_yaml_file(d / "config.yaml")
            kind = str(cfg.get("kind", "")).lower()
            if kind == "lep":
                out.append(
                    ComponentReport(
                        kind="hook",
                        name=d.name,
                        tier=PROTOCOL,
                        detail="lep",
                        reasons=(
                            "out-of-process LEP hook — portable by "
                            "construction over line-delimited JSON-RPC",
                        ),
                    )
                )
            else:
                detail = "class" if kind in ("", "class", "inprocess") else kind
                out.append(
                    ComponentReport(
                        kind="hook",
                        name=d.name,
                        tier=INPROCESS,
                        detail=detail,
                        reasons=(
                            "in-process Python hook class — reads live loop "
                            "state; port to a kind: lep hook for portability",
                        ),
                    )
                )
    return out


def _resource_is_runtime_builtin(path: Path) -> bool:
    """True if a resource only wraps a looplet builtin host service.

    Such resources (e.g. ``compact_service.py`` → ``default_compact_service``)
    are a host responsibility, not author-owned shared state: any conforming
    loader provides its own equivalent, so they do not pin the cartridge to a
    Python host.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("looplet"):
            for alias in node.names:
                if alias.name in _RUNTIME_SERVICE_FACTORIES:
                    return True
    return False


def _analyse_resources(
    root: Path, state_services: dict[str, Any] | None = None
) -> list[ComponentReport]:
    out: list[ComponentReport] = []
    state_services = state_services or {}
    resources_dir = root / "resources"
    if resources_dir.is_dir():
        for p in sorted(resources_dir.iterdir()):
            if p.is_file() and p.suffix == ".py" and not p.name.startswith("_"):
                if p.stem in state_services:
                    # A resource whose name matches a declared state
                    # service is backed by the out-of-process
                    # StateServiceClient the loader injects — portable.
                    out.append(
                        ComponentReport(
                            kind="resource",
                            name=p.stem,
                            tier=PROTOCOL,
                            detail="state-service",
                            reasons=(
                                "shared state served out-of-process by a "
                                "state_services: entry — the loader injects a "
                                "StateServiceClient proxy, so no Python body "
                                "is required by the host",
                            ),
                        )
                    )
                    continue
                if _resource_is_runtime_builtin(p):
                    # A thin declarative wrapper around a looplet builtin
                    # host service (compaction, skill manager, …). Any
                    # conforming loader ships its own equivalent, so this
                    # does not pin the cartridge to a Python host.
                    out.append(
                        ComponentReport(
                            kind="resource",
                            name=p.stem,
                            tier=RUNTIME,
                            detail="host-service",
                            reasons=(
                                "builtin host service (compaction / skill "
                                "manager / …) — a host responsibility, not "
                                "author-owned state; any conforming loader "
                                "provides its own equivalent",
                            ),
                        )
                    )
                    continue
                out.append(
                    ComponentReport(
                        kind="resource",
                        name=p.stem,
                        tier=INPROCESS,
                        detail="shared-ref",
                        reasons=(
                            "Python resource (@ref shared object) — shared "
                            "mutable state in one address space; port to a "
                            "state_services: entry for cross-runtime "
                            "portability",
                        ),
                    )
                )
    return out


def _analyse_state_services(
    state_services: dict[str, Any],
) -> list[ComponentReport]:
    """State Service Protocol servers → protocol-tier shared state.

    Each ``state_services:`` entry is an out-of-process server that owns
    a piece of shared mutable state behind a Unix socket. Portable by
    construction: any conforming loader can spawn the ``command:`` and
    speak the line-delimited JSON wire protocol.
    """
    out: list[ComponentReport] = []
    for svc_name, svc_cfg in state_services.items():
        out.append(
            ComponentReport(
                kind="resource",
                name=f"state:{svc_name}",
                tier=PROTOCOL,
                detail="ssp",
                reasons=(
                    "out-of-process state service (SSP) — shared mutable "
                    "state lives behind a Unix socket, addressable by any "
                    "runtime over line-delimited JSON; no Python required "
                    "by the loader",
                ),
            )
        )
    return out


def analyse_cartridge(path: str | Path) -> CartridgePortabilityReport:
    """Statically classify a cartridge directory's portability.

    Args:
        path: Path to the cartridge root (the directory containing
            ``config.yaml`` / ``cartridge.json``).

    Returns:
        A :class:`CartridgePortabilityReport` with a per-component
        breakdown and an overall ``portable`` / ``python-host`` verdict.
    """
    return _analyse_cartridge(path)


def _analyse_cartridge(
    path: str | Path, *, _seen: set[Path] | None = None
) -> CartridgePortabilityReport:
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"cartridge root is not a directory: {root}")

    config = _read_yaml_file(root / "config.yaml")
    mcp_servers = config.get("mcp_servers")
    mcp_servers = mcp_servers if isinstance(mcp_servers, dict) else {}
    builtin_hooks = config.get("builtin_hooks")
    builtin_hooks = builtin_hooks if isinstance(builtin_hooks, list) else []
    builtin_tools = config.get("builtin_tools")
    builtin_tools = builtin_tools if isinstance(builtin_tools, list) else []
    state_services = config.get("state_services")
    state_services = state_services if isinstance(state_services, dict) else {}

    components: list[ComponentReport] = []

    # config + prompts are always pure portable data.
    components.append(
        ComponentReport(
            kind="config",
            name="config.yaml",
            tier=PROTOCOL,
            detail="data",
            reasons=("declarative configuration — pure data",),
        )
    )
    if (root / "prompts").is_dir():
        components.append(
            ComponentReport(
                kind="prompts",
                name="prompts/",
                tier=PROTOCOL,
                detail="text",
                reasons=("prompt text — pure data",),
            )
        )

    # builtin_tools: declarative stdlib references → stdlib tier.
    for entry in builtin_tools:
        tool_name = entry if isinstance(entry, str) else str(entry)
        components.append(
            ComponentReport(
                kind="tool",
                name=tool_name,
                tier=STDLIB,
                detail="builtin",
                reasons=(
                    "declarative looplet stdlib tool — portable to any "
                    "runtime that ships the same named archetype",
                ),
            )
        )

    components.extend(_analyse_tools(root, mcp_servers))
    components.extend(_analyse_hooks(root, builtin_hooks))
    components.extend(_analyse_state_services(state_services))
    components.extend(_analyse_resources(root, state_services))

    # ``extends:`` inheritance — the loader merges the parent workspace
    # under the child, so the child inherits every parent component. A
    # report that ignored this would silently under-count blockers (e.g.
    # a one-tool child that ``extends:`` a 27-blocker parent). Resolve the
    # parent recursively and fold in its components, tagged ``inherited``.
    extends_val = config.get("extends")
    if isinstance(extends_val, str) and extends_val.strip():
        parent = Path(extends_val)
        if not parent.is_absolute():
            parent = (root / parent).resolve()
        parent = parent.resolve()
        if parent.is_dir() and parent not in (_seen or set()):
            seen = set(_seen or set())
            seen.add(root)
            parent_report = _analyse_cartridge(parent, _seen=seen)
            own_names = {(c.kind, c.name) for c in components}
            for c in parent_report.components:
                # The child always supplies its own config/prompts; skip
                # the parent's so they aren't double-counted.
                if c.kind in ("config", "prompts"):
                    continue
                if (c.kind, c.name) in own_names:
                    continue  # child overlay wins for same-named components
                components.append(
                    ComponentReport(
                        kind=c.kind,
                        name=f"{c.name} (inherited)",
                        tier=c.tier,
                        detail=c.detail,
                        reasons=(
                            f"inherited via extends: {extends_val} — "
                            + (c.reasons[0] if c.reasons else ""),
                        ),
                    )
                )

    name = root.name
    return CartridgePortabilityReport(
        root=root,
        name=name,
        components=tuple(components),
    )
