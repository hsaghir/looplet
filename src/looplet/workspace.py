"""Composable Harness Workspace (CHW) — bidirectional cartridge ↔ preset.

A *workspace* is a directory layout that round-trips with an
:class:`AgentPreset` losslessly for the JSON-able subset of the harness
and provides a clean code-escape hatch for the rest. It is the missing
inverse of :class:`looplet.bundles.SkillBundle`, which can be loaded
from disk but not written back from a live preset.

Design goal
-----------

Make the agent harness an editable artifact on disk so external tools
(harness search, GEPA-style evolution, diff/review workflows) can
mutate components by file diff, version-control the result with git,
and re-materialise an :class:`AgentPreset` for execution — without
anyone forking the loop or replacing the cartridge mechanism.

Layout
------

::

    my_workspace/
    ├── workspace.json           # schema_version, name, description, version bookkeeping
    ├── prompts/
    │   └── system.md            # config.system_prompt (file body)
    ├── config.yaml              # LoopConfig JSON-able subset (max_steps, etc.)
    ├── tools/
    │   └── grep/
    │       ├── tool.yaml        # name, description, parameters, concurrent_safe, free, timeout_s
    │       └── execute.py       # def execute(*, ...) -> Any
    ├── hooks/
    │   └── 00_done_gate/        # leading number = sort order = hook list order
    │       ├── hook.py          # exposes either `class HookClass` or `def build()`
    │       └── config.yaml      # optional kwargs for HookClass(**kwargs)
    └── memory/
        └── lessons.md           # one StaticMemorySource per file; filename = source name

What is round-trippable
-----------------------

* ``LoopConfig``: every primitive scalar field (``max_steps``,
  ``max_tokens``, ``temperature``, ``recovery_temperature``,
  ``done_tool``, ``max_turn_continuations``, ``use_native_tools``,
  ``concurrent_dispatch``, ``reactive_recovery``, ``context_window``,
  ``max_briefing_tokens``, ``checkpoint_dir``); ``acceptance_criteria``;
  ``tool_metadata`` and ``generate_kwargs`` (JSON-able dicts).
* Every :class:`ToolSpec` whose ``execute`` is a top-level function
  (closures cannot be re-imported from disk).
* Every hook that either: (a) implements an opt-in
  ``to_config() -> dict`` returning JSON-able kwargs for its
  constructor, OR (b) is a top-level class importable from a written
  ``hook.py`` module, OR (c) ships its own ``hook.py`` source via the
  code-escape hatch.
* :class:`StaticMemorySource` instances; other memory sources land
  under the code-escape hatch.

What is NOT round-trippable (raises ``WorkspaceSerializationError``
when ``preset_to_workspace`` is called with ``strict=True``)
-----------------------------------------------------------------------

Callable / opaque ``LoopConfig`` fields (``build_briefing``,
``router``, ``tracer``, ``compact_service``, ``recovery_registry``,
``output_schema``, ``initial_checkpoint``, ``cache_policy``,
``cancel_token``, ``approval_handler``, ``render_messages_override``,
``domain``). When ``strict=False`` (default), they are silently
omitted from the serialized config and a list of skipped fields is
returned in the resulting :class:`Workspace.serialization_warnings`.

These fields can still be wired **declaratively on load** by
hand-authoring ``config.yaml`` with ``"@<name>"`` references that
resolve against ``resources/<name>.py`` builders — the same
mechanism hook kwargs use. Example::

    # config.yaml
    max_steps: 20
    compact_service: "@compact_service"
    tracer: "@tracer"

    # resources/compact_service.py
    from looplet.compact import PruneToolResults, TruncateCompact, compact_chain
    def build(runtime=None):
        return compact_chain(PruneToolResults(), TruncateCompact())

This eliminates the ``setup.py`` detour for the common case of
attaching callable LoopConfig services. ``setup.py`` is still
required for: (a) injecting shared resources into top-level tool
function module globals, and (b) live-state callables that close
over ``state`` per turn.

Why this is in Looplet (not in a research extension)
----------------------------------------------------

The disk format is generic infrastructure: anyone can use it for
cartridge editing, agent diffing, code review, packaging, or
between-round harness search. The research-specific layer
(manifests with ``predicted_fixes``/``predicted_regressions``,
the evolve agent, the search loop) lives in downstream packages
that consume :class:`Workspace`.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from looplet.memory import PersistentMemorySource, StaticMemorySource

if TYPE_CHECKING:
    from looplet.presets import AgentPreset
    from looplet.tools import BaseToolRegistry

__all__ = [
    "WorkspaceLayout",
    "Workspace",
    "WorkspaceSerializationError",
    "preset_to_workspace",
    "workspace_to_preset",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ── Layout constants ────────────────────────────────────────────


class WorkspaceLayout:
    """Fixed mount points inside a workspace directory."""

    WORKSPACE_JSON = "workspace.json"
    CONFIG_YAML = "config.yaml"
    PROMPTS_DIR = "prompts"
    SYSTEM_PROMPT_MD = "prompts/system.md"
    TOOLS_DIR = "tools"
    HOOKS_DIR = "hooks"
    MEMORY_DIR = "memory"
    RESOURCES_DIR = "resources"
    SETUP_PY = "setup.py"

    # ``LoopConfig`` field names that round-trip via ``config.yaml``.
    SERIALIZABLE_CONFIG_FIELDS: tuple[str, ...] = (
        "max_steps",
        "max_tokens",
        "temperature",
        "recovery_temperature",
        "done_tool",
        "max_turn_continuations",
        "use_native_tools",
        "concurrent_dispatch",
        "reactive_recovery",
        "context_window",
        "max_briefing_tokens",
        "checkpoint_dir",
        "acceptance_criteria",
        "tool_metadata",
        "generate_kwargs",
    )

    # ``LoopConfig`` callable / opaque fields that cannot round-trip.
    NON_SERIALIZABLE_CONFIG_FIELDS: tuple[str, ...] = (
        "build_briefing",
        "extract_entities",
        "build_trace",
        "build_prompt",
        "extract_step_metadata",
        "domain",
        "router",
        "tracer",
        "recovery_registry",
        "compact_service",
        "output_schema",
        "initial_checkpoint",
        "cache_policy",
        "cancel_token",
        "approval_handler",
        "render_messages_override",
    )


# ── Errors ──────────────────────────────────────────────────────


class WorkspaceSerializationError(RuntimeError):
    """Raised when a workspace component cannot be round-tripped.

    Use ``strict=False`` on :func:`preset_to_workspace` to demote these
    into recorded warnings on the resulting :class:`Workspace`.
    """


# ── Data class ──────────────────────────────────────────────────


@dataclass
class Workspace:
    """A loaded composable harness workspace.

    Serves both as the in-memory representation of an on-disk workspace
    and as the structured target of :func:`preset_to_workspace`.
    """

    path: Path
    name: str = ""
    description: str = ""
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    serialization_warnings: list[str] = field(default_factory=list)

    # ── classmethod builders ───────────────────────────────────

    @classmethod
    def from_directory(cls, path: str | Path) -> "Workspace":
        """Load workspace metadata from a CHW directory.

        Use :func:`workspace_to_preset` to materialise the
        :class:`AgentPreset` from the loaded workspace.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"workspace directory not found: {root}")
        meta_path = root / WorkspaceLayout.WORKSPACE_JSON
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"workspace metadata not found at {meta_path}; "
                f"is this a Composable Harness Workspace?"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            path=root,
            name=str(meta.get("name", root.name)),
            description=str(meta.get("description", "")),
            schema_version=int(meta.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(meta.get("metadata", {})),
        )

    # ── instance API ───────────────────────────────────────────

    def write_metadata(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / WorkspaceLayout.WORKSPACE_JSON).write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "name": self.name,
                    "description": self.description,
                    "metadata": dict(self.metadata),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def to_preset(self) -> "AgentPreset":
        """Materialise the :class:`AgentPreset` described by this workspace."""
        return workspace_to_preset(self.path)


# ── helpers: minimal YAML (key: value, lists, nested dicts) ────


def _dump_yaml(value: Any, indent: int = 0) -> str:
    """Dependency-free YAML emitter for the JSON subset we need.

    Looplet has no third-party dependencies; we hand-emit the limited
    YAML subset we use (scalars, lists of scalars/dicts, nested dicts).
    """
    pad = "  " * indent
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if not value or any(c in value for c in ":#\n'\"") or value.strip() != value:
            return json.dumps(value)
        return value
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = _dump_yaml(item, indent + 1).rstrip()
                if "\n" in rendered:
                    lines.append(f"{pad}-")
                    lines.append(rendered)
                else:
                    lines.append(f"{pad}- {rendered.strip()}")
            else:
                lines.append(f"{pad}- {_dump_yaml(item, 0)}")
        return "\n".join(lines)
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for key, val in value.items():
            rendered = _dump_yaml(val, indent + 1)
            if isinstance(val, (dict, list)) and rendered not in ("{}", "[]"):
                lines.append(f"{pad}{key}:")
                lines.append(rendered)
            else:
                lines.append(f"{pad}{key}: {rendered}")
        return "\n".join(lines)
    raise WorkspaceSerializationError(
        f"cannot serialize value of type {type(value).__name__!r} to workspace YAML"
    )


def _load_yaml(text: str) -> Any:
    """Parse the YAML subset emitted by :func:`_dump_yaml`.

    Supports key: value lines, nested dicts (indent 2), lists with ``- ``,
    and JSON-style scalars (true/false/null/numbers/quoted strings). For
    anything beyond this subset we fall back to JSON parsing of the line
    value.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    pos = 0

    def parse_block(min_indent: int) -> Any:
        nonlocal pos
        # Detect whether the block is a list (lines starting with "- ")
        # or a dict (lines with "key: value"). Empty block → empty dict.
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        if pos >= len(lines):
            return {}
        first = lines[pos]
        first_indent = len(first) - len(first.lstrip())
        if first_indent < min_indent:
            return {}
        is_list = first.lstrip().startswith("- ") or first.lstrip() == "-"
        if is_list:
            return parse_list(first_indent)
        return parse_dict(first_indent)

    def parse_dict(indent: int) -> dict[str, Any]:
        nonlocal pos
        out: dict[str, Any] = {}
        while pos < len(lines):
            line = lines[pos]
            if not line.strip():
                pos += 1
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent < indent:
                break
            stripped = line.strip()
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                raise WorkspaceSerializationError(f"unparseable workspace YAML line: {line!r}")
            key, _, raw_val = stripped.partition(":")
            raw_val = raw_val.strip()
            pos += 1
            if not raw_val:
                # Nested block follows.
                out[key.strip()] = parse_block(indent + 2)
            else:
                out[key.strip()] = _scalar(raw_val)
        return out

    def parse_list(indent: int) -> list[Any]:
        nonlocal pos
        out: list[Any] = []
        while pos < len(lines):
            line = lines[pos]
            if not line.strip():
                pos += 1
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent < indent:
                break
            stripped = line.strip()
            if not stripped.startswith("-"):
                break
            after = stripped[1:].strip()
            pos += 1
            if not after:
                out.append(parse_block(indent + 2))
            else:
                out.append(_scalar(after))
        return out

    def _scalar(raw: str) -> Any:
        if raw in ("null", "~", ""):
            return None
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw.startswith(("[", "{", '"')):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        try:
            if "." in raw or "e" in raw or "E" in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    return parse_block(0)


# ── helpers: hook + tool source loading ────────────────────────


def _import_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkspaceSerializationError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_resources(root: Path, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the shared-resource registry from ``resources/<name>.py`` files.

    Each resource module must define a builder named ``build``. The
    loader inspects the signature: ``build()`` (zero-arg) keeps the
    legacy contract; ``build(runtime)`` lets the resource read the
    host-supplied ``runtime`` dict (e.g. ``runtime['workspace']`` for
    the coder cartridge). Resources are shared by every ``"@<name>"``
    reference in hook / tool kwargs.
    """
    import inspect as _inspect  # noqa: PLC0415

    resources_dir = root / WorkspaceLayout.RESOURCES_DIR
    if not resources_dir.is_dir():
        return {}
    runtime_dict = dict(runtime or {})
    resources: dict[str, Any] = {}
    for resource_file in sorted(resources_dir.glob("*.py")):
        name = resource_file.stem
        module = _import_module_from_path(resource_file, f"_chw_resource_{name}")
        builder = getattr(module, "build", None)
        if not callable(builder):
            raise WorkspaceSerializationError(
                f"resource {name!r} ({resource_file}) must define `def build() -> Any`"
            )
        # Pass runtime only when the builder accepts it, so legacy
        # zero-arg builders keep working unchanged.
        try:
            sig = _inspect.signature(builder)
            accepts_runtime = "runtime" in sig.parameters or any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_runtime = False
        resources[name] = builder(runtime=runtime_dict) if accepts_runtime else builder()
    return resources


_REF_PREFIX = "@"


def _resolve_refs(value: Any, resources: dict[str, Any]) -> Any:
    """Replace any ``"@<name>"`` strings in ``value`` with their
    resolved resource. Recurses into dicts and lists. Other types pass
    through unchanged.

    Raises :class:`WorkspaceSerializationError` when a reference points
    at a missing resource so loading fails loud.
    """
    if isinstance(value, str) and value.startswith(_REF_PREFIX):
        name = value[len(_REF_PREFIX) :]
        if name not in resources:
            raise WorkspaceSerializationError(
                f"unresolved resource reference {value!r}; known resources: {sorted(resources)}"
            )
        return resources[name]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, resources) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(item, resources) for item in value]
    return value


def _safe_filename(name: str) -> str:
    """Sanitise an arbitrary string into a directory-safe filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "unnamed"


class _DataclassReprFailed(Exception):
    """Raised by ``_render_dataclass_kwargs`` when a field cannot be
    reproduced in source form (closure, lambda, opaque object, …).

    The auto-emit machinery catches this and falls through to the
    generic class branch which writes the safer ``Cls(...)`` shell.
    """


def _render_value_literal(value: Any, imports: set[str]) -> str:
    """Render ``value`` as a Python source expression usable in a builder.

    Supports JSON-able scalars/lists/dicts, top-level importable
    callables (emitted as ``from M import F`` + bare name), and
    nested dataclasses (recurses). Mutates ``imports`` so the caller
    can collect every needed import line.

    Raises :class:`_DataclassReprFailed` for closures, lambdas, opaque
    instances, or anything else that can't be re-emitted in source.
    """
    import dataclasses as _dc  # noqa: PLC0415
    import enum as _enum  # noqa: PLC0415

    # Enum check first — string-backed enums (``class X(str, Enum)``)
    # would otherwise match the scalar branch and ``repr()`` would
    # emit invalid ``<EnumClass.MEMBER: 'value'>`` source.
    if isinstance(value, _enum.Enum):
        ecls = type(value)
        emod = ecls.__module__
        ename = ecls.__name__
        if emod and emod not in ("builtins",) and not emod.startswith("_chw_"):
            imports.add(f"from {emod} import {ename}")
            return f"{ename}.{value.name}"
        raise _DataclassReprFailed(f"non-importable enum: {ecls!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        parts = [_render_value_literal(v, imports) for v in value]
        if isinstance(value, list):
            return "[" + ", ".join(parts) + "]"
        # tuple: keep trailing comma for single-element tuples
        if len(parts) == 1:
            return "(" + parts[0] + ",)"
        return "(" + ", ".join(parts) + ")"
    if isinstance(value, dict):
        parts = [
            f"{_render_value_literal(k, imports)}: {_render_value_literal(v, imports)}"
            for k, v in value.items()
        ]
        return "{" + ", ".join(parts) + "}"
    # Top-level importable callable / class → emit bare name + import.
    if callable(value):
        mod = getattr(value, "__module__", "") or ""
        name = getattr(value, "__name__", "")
        qual = getattr(value, "__qualname__", "<lambda>")
        if (
            mod
            and mod not in ("builtins",)
            and not mod.startswith("_chw_")
            and name
            and qual != "<lambda>"
            and "<locals>" not in qual
        ):
            imports.add(f"from {mod} import {name}")
            return name
        raise _DataclassReprFailed(f"non-importable callable: {qual!r} from {mod!r}")
    # Nested dataclass instance → recurse.
    if _dc.is_dataclass(value):
        v_mod = type(value).__module__
        v_name = type(value).__name__
        if v_mod and v_mod not in ("builtins", "__main__") and not v_mod.startswith("_chw_"):
            imports.add(f"from {v_mod} import {v_name}")
            inner_kwargs = _render_dataclass_kwargs(value, imports)
            return f"{v_name}({inner_kwargs})"
    raise _DataclassReprFailed(f"unrenderable value of type {type(value).__name__!r}")


def _render_dataclass_kwargs(instance: Any, imports: set[str]) -> str:
    """Return ``"k1=v1, k2=v2, ..."`` reproducing ``instance``'s fields.

    Skips fields whose current value equals the dataclass-declared
    default (or default_factory output) so the rendered builder stays
    compact and matches the source preset's expressed configuration.
    """
    import dataclasses as _dc  # noqa: PLC0415

    parts: list[str] = []
    for f in _dc.fields(instance):
        val = getattr(instance, f.name)
        # Skip fields holding their default — keeps builder readable
        # and matches dataclass __repr__ semantics.
        if f.default is not _dc.MISSING and val == f.default:
            continue
        if f.default_factory is not _dc.MISSING:  # type: ignore[misc]
            try:
                if val == f.default_factory():  # type: ignore[misc]
                    continue
            except Exception:  # noqa: BLE001
                pass
        rendered = _render_value_literal(val, imports)
        parts.append(f"{f.name}={rendered}")
    return ", ".join(parts)


_RUNTIME_PLACEHOLDER = re.compile(r"\$\{runtime\.([A-Za-z_][A-Za-z0-9_]*)\}")


def _apply_runtime_substitutions(text: str, runtime: dict[str, Any]) -> str:
    """Replace ``${runtime.<key>}`` placeholders in ``text`` with the
    string form of ``runtime[<key>]``.

    Used on ``config.yaml`` (and any other plain-text workspace file the
    loader passes through this) so workspace authors can parameterise
    paths and other host-supplied values without writing a setup.py.

    Unknown keys raise :class:`WorkspaceSerializationError` so a typo
    fails loudly at load time rather than silently leaving the
    placeholder string in the running config.
    """

    def _sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in runtime:
            raise WorkspaceSerializationError(
                f"unresolved ${{runtime.{key}}} placeholder; known runtime keys: {sorted(runtime)}"
            )
        return str(runtime[key])

    return _RUNTIME_PLACEHOLDER.sub(_sub, text)


def _hook_class(hook: Any) -> type:
    return hook if inspect.isclass(hook) else type(hook)


# ── Serialise: AgentPreset → directory ─────────────────────────


def preset_to_workspace(
    preset: "AgentPreset",
    out_dir: str | Path,
    *,
    name: str | None = None,
    description: str = "",
    overwrite: bool = False,
    strict: bool = False,
) -> Workspace:
    """Write an :class:`AgentPreset` to a CHW directory.

    Args:
        preset: The harness to serialise.
        out_dir: Target directory. Created if missing. If it already
            exists and is non-empty, ``overwrite=True`` is required.
        name: Workspace name. Defaults to the directory basename.
        description: Free-form description stored in
            ``workspace.json``.
        overwrite: Allow writing into a non-empty existing directory
            (its CHW-managed subdirectories are wiped first).
        strict: When ``True``, raise
            :class:`WorkspaceSerializationError` on any non-round-trippable
            component. When ``False`` (default), record warnings on the
            returned workspace and skip the offending field.

    Returns:
        The :class:`Workspace` describing the newly-written directory.
    """
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} is not empty; pass overwrite=True to wipe and rewrite")
    if root.exists() and overwrite:
        for sub in (
            WorkspaceLayout.PROMPTS_DIR,
            WorkspaceLayout.TOOLS_DIR,
            WorkspaceLayout.HOOKS_DIR,
            WorkspaceLayout.MEMORY_DIR,
            WorkspaceLayout.RESOURCES_DIR,
        ):
            sub_path = root / sub
            if sub_path.is_dir():
                shutil.rmtree(sub_path)
        for stale in (WorkspaceLayout.WORKSPACE_JSON, WorkspaceLayout.CONFIG_YAML):
            stale_path = root / stale
            if stale_path.is_file():
                stale_path.unlink()
    root.mkdir(parents=True, exist_ok=True)

    workspace = Workspace(
        path=root,
        name=name or root.name,
        description=description,
    )
    warnings: list[str] = []

    # 1. config — write JSON-able subset; emit warnings for the rest.
    cfg = preset.config
    serialized_cfg: dict[str, Any] = {}
    for fname in WorkspaceLayout.SERIALIZABLE_CONFIG_FIELDS:
        if fname == "system_prompt":
            continue  # written as a separate prompts/system.md file
        if not hasattr(cfg, fname):
            continue
        value = getattr(cfg, fname)
        if value is None and fname in ("acceptance_criteria",):
            continue
        try:
            json.dumps(value)
        except TypeError:
            msg = f"config.{fname} ({type(value).__name__!r}) is not JSON-able; skipping"
            if strict:
                raise WorkspaceSerializationError(msg)
            warnings.append(msg)
            continue
        serialized_cfg[fname] = value

    # Auto-emit ``@ref`` strings into config.yaml for any non-serializable
    # LoopConfig field that is set, mirroring how hook kwargs ride the
    # resource-builder machinery. The actual instances get collected here
    # and passed to ``_write_resources_for_refs`` below so the writer
    # auto-generates ``resources/<field>.py`` placeholders the loader
    # resolves declaratively.
    config_field_refs: dict[str, Any] = {}
    for fname in WorkspaceLayout.NON_SERIALIZABLE_CONFIG_FIELDS:
        if not hasattr(cfg, fname):
            continue
        value = getattr(cfg, fname)
        if value is None:
            continue
        # Round-trip via ``@<fname>`` ref + auto-generated resource stub.
        # The user can replace the stub with a real builder later; for
        # in-process snapshot+reload (harness search / GEPA evolution)
        # the auto-emitted builder rebuilds the type from its module
        # the same way hook resources do.
        serialized_cfg[fname] = f"{_REF_PREFIX}{fname}"
        config_field_refs[fname] = value

    if serialized_cfg:
        (root / WorkspaceLayout.CONFIG_YAML).write_text(
            _dump_yaml(serialized_cfg) + "\n",
            encoding="utf-8",
        )

    # 2. system prompt
    prompts_dir = root / WorkspaceLayout.PROMPTS_DIR
    prompts_dir.mkdir(exist_ok=True)
    (root / WorkspaceLayout.SYSTEM_PROMPT_MD).write_text(
        getattr(cfg, "system_prompt", "") or "",
        encoding="utf-8",
    )

    # 3. tools — one subdir per tool with tool.yaml + execute.py
    tools_root = root / WorkspaceLayout.TOOLS_DIR
    tools_root.mkdir(exist_ok=True)
    for spec in _iter_tool_specs(preset.tools):
        _write_tool(spec, tools_root, warnings, strict)

    # 4. hooks — one subdir per hook, ordered by index for deterministic load
    hooks_root = root / WorkspaceLayout.HOOKS_DIR
    hooks_root.mkdir(exist_ok=True)
    for idx, hook in enumerate(preset.hooks):
        _write_hook(hook, hooks_root, idx, warnings, strict)

    # 5. memory sources — StaticMemorySource → markdown file
    memory_root = root / WorkspaceLayout.MEMORY_DIR
    memory_root.mkdir(exist_ok=True)
    for idx, source in enumerate(getattr(cfg, "memory_sources", []) or []):
        _write_memory(source, memory_root, idx, warnings, strict)

    # 6. resources — for any @<name> ref found in written hook configs,
    # emit a placeholder ``resources/<name>.py`` so the snapshot loads
    # without unresolved-reference errors. The placeholder returns the
    # actual instance the hook is currently bound to so the reload
    # behaves identically to the source preset (within process — across
    # processes the user must replace the placeholder with a real
    # builder).
    _write_resources_for_refs(
        hooks_root,
        root,
        preset.hooks,
        warnings,
        strict,
        extra_refs=config_field_refs,
    )

    workspace.serialization_warnings = warnings
    workspace.write_metadata()
    return workspace


def _write_resources_for_refs(
    hooks_root: Path,
    root: Path,
    hooks: list[Any],
    warnings: list[str],
    strict: bool,
    *,
    extra_refs: dict[str, Any] | None = None,
) -> None:
    """Emit ``resources/<name>.py`` placeholders for every ``@<name>``
    ref found in hook configs, plus any caller-supplied
    ``extra_refs`` (used by the writer to auto-emit callable
    LoopConfig fields like ``compact_service``).

    The placeholder's ``build()`` returns a stashed module-level
    reference to the actual object the hook is bound to in the
    source preset. This makes in-process snapshot+reload
    round-trip cleanly (the same FileCache instance gets re-used).
    For cross-process distribution the user must replace the
    placeholder with a real builder (the comment header explains
    how).
    """
    # Walk every hook's to_config() output and collect unique @<name> refs
    # paired with the actual constructor-arg value the hook holds.
    refs_seen: dict[str, Any] = {}
    for hook in hooks:
        if not hasattr(hook, "to_config") or not callable(hook.to_config):
            continue
        try:
            cfg = hook.to_config()
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(cfg, dict):
            continue
        for kwarg_name, kwarg_value in cfg.items():
            if not (isinstance(kwarg_value, str) and kwarg_value.startswith(_REF_PREFIX)):
                continue
            ref_name = kwarg_value[len(_REF_PREFIX) :]
            if ref_name in refs_seen:
                continue
            # Pull the actual instance from the hook's matching attribute
            # (try public attr name, then mangled private convention).
            actual = getattr(hook, kwarg_name, None)
            if actual is None:
                actual = getattr(hook, f"_{kwarg_name}", None)
            refs_seen[ref_name] = actual

    # Caller-supplied refs (e.g. config.compact_service) win over hook
    # refs when names collide — the LoopConfig field is the
    # authoritative live instance for that key.
    if extra_refs:
        for ref_name, instance in extra_refs.items():
            refs_seen[ref_name] = instance

    if not refs_seen:
        return

    resources_dir = root / WorkspaceLayout.RESOURCES_DIR
    resources_dir.mkdir(exist_ok=True)
    for ref_name, instance in refs_seen.items():
        if instance is None:
            msg = (
                f"resource {ref_name!r}: could not locate the live instance on "
                f"any hook; emitted a stub builder that returns None"
            )
            if strict:
                raise WorkspaceSerializationError(msg)
            warnings.append(msg)
            stub = (
                f"# AUTOGENERATED PLACEHOLDER for {ref_name!r}.\n"
                f"# No live instance was found on any hook. Replace this\n"
                f"# stub with a real ``def build(runtime=None)`` that\n"
                f"# constructs the resource the hooks expect.\n"
                "def build(runtime=None):\n"
                "    return None\n"
            )
            (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(stub, encoding="utf-8")
            continue

        # Special case: list / tuple of top-level callables. Common
        # for ``EvalHook(evaluators=[...])`` and similar collector
        # patterns. Emit a builder that re-imports each callable by
        # ``module:qualname``. ``__main__`` callables are accepted
        # (so script-driven dogfooding round-trips) but recorded as
        # a non-fatal warning in strict mode — cross-process loads
        # need the callables moved to a real module.
        if (
            isinstance(instance, (list, tuple))
            and instance
            and all(
                callable(item)
                and getattr(item, "__module__", "") not in ("", "builtins")
                and not getattr(item, "__module__", "").startswith("_chw_")
                and getattr(item, "__qualname__", "<lambda>") != "<lambda>"
                and "<locals>" not in getattr(item, "__qualname__", "<locals>")
                for item in instance
            )
        ):
            import_lines: list[str] = []
            ref_names: list[str] = []
            main_callables: list[str] = []
            for item in instance:
                mod = item.__module__
                fname = item.__name__
                import_lines.append(f"from {mod} import {fname}")
                ref_names.append(fname)
                if mod == "__main__":
                    main_callables.append(fname)
            if main_callables:
                # Non-fatal in strict mode: best-effort same-process
                # round-trip works, cross-process needs editing.
                warnings.append(
                    f"resource {ref_name!r}: contains __main__ callables "
                    f"{main_callables} — re-imports from ``__main__`` for "
                    f"same-process round-trip but cross-process loads will "
                    f"fail until these are moved to an importable module"
                )
            joined_imports = "\n".join(import_lines)
            joined_refs = ", ".join(ref_names)
            container = "list" if isinstance(instance, list) else "tuple"
            if container == "list":
                stub = (
                    f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                    f"# Returns a fresh list of top-level callables\n"
                    f"# re-imported from their original modules. If the\n"
                    f"# source preset depended on closures or instance state\n"
                    f"# carried by the callables, replace each entry with\n"
                    f"# the real construction.\n"
                    f"{joined_imports}\n"
                    "\n"
                    "def build(runtime=None):\n"
                    f"    return [{joined_refs}]\n"
                )
            else:
                stub = (
                    f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                    f"# Returns a fresh tuple of top-level callables\n"
                    f"# re-imported from their original modules.\n"
                    f"{joined_imports}\n"
                    "\n"
                    "def build(runtime=None):\n"
                    f"    return ({joined_refs},)\n"
                )
            (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(stub, encoding="utf-8")
            continue
        cls_module = type(instance).__module__ or ""
        cls_name = type(instance).__name__

        # Dataclass auto-emit: when the live instance is a dataclass we
        # reproduce its full field state in the builder (not just the
        # required ctor args). Common shape: ``PermissionEngine(rules=[
        # PermissionRule(...), ...])`` — the rules list is JSON-able-ish
        # only after we re-emit each PermissionRule's class import.
        # Falls through to the generic class branch when reproduction
        # fails (e.g. a field holds a closure).
        import dataclasses as _dc  # noqa: PLC0415

        if (
            _dc.is_dataclass(instance)
            and cls_module
            and cls_module not in {"__main__", "builtins"}
            and not cls_module.startswith("_chw_")
        ):
            try:
                imports_set: set[str] = {f"from {cls_module} import {cls_name}"}
                kwargs_src = _render_dataclass_kwargs(instance, imports_set)
            except _DataclassReprFailed as exc:
                kwargs_src = None
                logger.debug("dataclass auto-emit fell through for %s: %s", ref_name, exc)
            if kwargs_src is not None:
                joined_imports = "\n".join(sorted(imports_set))
                stub = (
                    f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                    f"# Reproduces the live ``{cls_name}`` instance field-by-\n"
                    f"# field. Replace any value here with a real construction\n"
                    f"# if you need different behaviour at load time.\n"
                    f"{joined_imports}\n"
                    "\n"
                    "def build(runtime=None):\n"
                    f"    return {cls_name}({kwargs_src})\n"
                )
                (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(
                    stub, encoding="utf-8"
                )
                continue

        # Best-effort: for installed classes, emit a builder that
        # imports the class and returns a fresh instance. This loses
        # cross-process state — explain that in the header.
        if (
            cls_module
            and cls_module not in {"__main__", "builtins"}
            and not cls_module.startswith("_chw_")
        ):
            # Inspect the constructor so we know which kwargs to pass —
            # FileCache(workspace=...) needs ``workspace`` from runtime,
            # while StreamingHook(emitter=...) won't survive at all.
            try:
                ctor_sig = inspect.signature(type(instance).__init__)
                required: list[str] = []
                for p_name, p in ctor_sig.parameters.items():
                    if p_name == "self":
                        continue
                    if p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    if p.default is inspect.Parameter.empty:
                        required.append(p_name)
            except (TypeError, ValueError):
                required = []

            ctor_kwargs_parts: list[str] = []
            extra_imports: list[str] = []
            best_effort_warnings: list[str] = []
            for kw in required:
                # Try to derive the kwarg from the live instance's
                # matching attribute (public ``kw`` first, then mangled
                # private ``_kw``). When the attr is a top-level
                # importable callable, generate a real ``from M import
                # F`` so the builder reproduces the original wiring
                # exactly. Otherwise fall back to ``runtime.get(kw)``
                # so the user can still inject the value at load time.
                live = getattr(instance, kw, None)
                if live is None:
                    live = getattr(instance, f"_{kw}", None)
                live_mod = getattr(live, "__module__", "") or ""
                live_name = getattr(live, "__name__", "")
                live_qual = getattr(live, "__qualname__", "<lambda>")
                if (
                    callable(live)
                    and live_mod
                    and live_mod not in ("builtins", "_chw_")
                    and not live_mod.startswith("_chw_")
                    and live_name
                    and live_qual != "<lambda>"
                    and "<locals>" not in live_qual
                ):
                    extra_imports.append(f"from {live_mod} import {live_name}")
                    ctor_kwargs_parts.append(f"{kw}={live_name}")
                    if live_mod == "__main__":
                        best_effort_warnings.append(
                            f"resource {ref_name!r}: kwarg {kw!r} re-imports "
                            f"from ``__main__`` ({live_name!r}); cross-process "
                            f"loads will fail until moved to a real module"
                        )
                else:
                    ctor_kwargs_parts.append(f"{kw}=runtime.get({kw!r})")

            if best_effort_warnings:
                warnings.extend(best_effort_warnings)
            ctor_kwargs = ", ".join(ctor_kwargs_parts)
            extra_import_block = ("\n".join(extra_imports) + "\n") if extra_imports else ""
            stub = (
                f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                f"# Returns a FRESH {cls_name} instance from {cls_module} on\n"
                f"# every workspace load. Required ctor kwargs are derived from\n"
                f"# the live source instance when possible (top-level callables\n"
                f"# get re-imported); the rest fall back to ``runtime.get(<kw>)``.\n"
                f"# Replace any unresolved kwargs with real construction before\n"
                f"# distributing the workspace.\n"
                f"from {cls_module} import {cls_name}\n"
                f"{extra_import_block}"
                "\n"
                "def build(runtime=None):\n"
                "    runtime = runtime or {}\n"
                f"    return {cls_name}({ctor_kwargs})\n"
            )
        else:
            msg = (
                f"resource {ref_name!r}: live instance class {cls_name!r} from "
                f"module {cls_module!r} is not importable; emitted a None-stub "
                f"builder \u2014 replace ``resources/{_safe_filename(ref_name)}.py`` "
                f"with a real builder before loading in a new process"
            )
            if strict:
                raise WorkspaceSerializationError(msg)
            warnings.append(msg)
            stub = (
                f"# AUTOGENERATED PLACEHOLDER for {ref_name!r} (instance class\n"
                f"# {cls_name!r} from non-importable module {cls_module!r}).\n"
                f"# Replace with a real builder.\n"
                "def build(runtime=None):\n"
                "    return None\n"
            )
        (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(stub, encoding="utf-8")


def _iter_tool_specs(tools: "BaseToolRegistry") -> Iterable[Any]:
    if hasattr(tools, "_tools"):
        return list(tools._tools.values())  # type: ignore[attr-defined]
    if hasattr(tools, "_specs"):
        return list(tools._specs.values())  # type: ignore[attr-defined]
    if hasattr(tools, "specs"):
        return list(tools.specs())  # type: ignore[attr-defined,operator]
    raise WorkspaceSerializationError(
        f"tool registry {type(tools).__name__!r} does not expose tool specs"
    )


def _write_tool(spec: Any, tools_root: Path, warnings: list[str], strict: bool) -> None:
    name = spec.name
    tool_dir = tools_root / _safe_filename(name)
    tool_dir.mkdir(parents=True, exist_ok=True)

    yaml_payload: dict[str, Any] = {
        "name": name,
        "description": spec.description,
        "parameters": dict(spec.parameters or {}),
    }
    for opt in ("concurrent_safe", "free", "timeout_s"):
        if hasattr(spec, opt):
            val = getattr(spec, opt)
            if val is not None:
                yaml_payload[opt] = val
    (tool_dir / "tool.yaml").write_text(_dump_yaml(yaml_payload) + "\n", encoding="utf-8")

    fn = spec.execute
    qualname = getattr(fn, "__qualname__", "<lambda>")
    if "<locals>" in qualname or qualname == "<lambda>":
        msg = (
            f"tool {name!r} execute is a closure or lambda ({qualname}); cannot round-trip to disk"
        )
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)
        (tool_dir / "execute.py").write_text(
            "# AUTOGENERATED PLACEHOLDER\n"
            "# Original tool.execute was a closure/lambda and could not be\n"
            "# serialised. Re-implement here as a top-level ``execute`` function.\n"
            "def execute(**kwargs):\n"
            "    raise NotImplementedError('replace this stub')\n",
            encoding="utf-8",
        )
        return

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        msg = f"tool {name!r} execute has no retrievable source"
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)
        return

    # Prefer re-importing the original function so its enclosing module's
    # imports stay in scope (typing.Any, helper functions, etc.). Falls
    # back to source-dump only when the function isn't importable.
    fn_name = getattr(fn, "__name__", "")
    module_name = getattr(fn, "__module__", "") or ""
    if fn_name and module_name and module_name not in {"__main__", "builtins"}:
        try:
            mod = importlib.import_module(module_name)
            if getattr(mod, fn_name, None) is fn:
                (tool_dir / "execute.py").write_text(
                    "# AUTOGENERATED from preset_to_workspace.\n"
                    "# Re-imported from the original module so the function's\n"
                    "# closure (typing imports, helpers) stays available.\n"
                    f"from {module_name} import {fn_name} as execute\n",
                    encoding="utf-8",
                )
                return
        except Exception:  # noqa: BLE001
            pass

    # Fallback: dump source. Add an `execute = <fn_name>` alias so the
    # loader finds it under the canonical name regardless of what the
    # original function was called.
    alias_line = f"execute = {fn_name}\n" if fn_name and fn_name != "execute" else ""
    (tool_dir / "execute.py").write_text(
        f"# AUTOGENERATED from preset_to_workspace.\n{source}\n{alias_line}",
        encoding="utf-8",
    )


def _write_hook(hook: Any, hooks_root: Path, index: int, warnings: list[str], strict: bool) -> None:
    cls = _hook_class(hook)
    cls_name = cls.__name__
    dir_name = f"{index:02d}_{_safe_filename(cls_name)}"
    hook_dir = hooks_root / dir_name
    hook_dir.mkdir(parents=True, exist_ok=True)

    # Strategy: write a hook.py that **re-imports** the original class from
    # its module by default, so the class's full closure (typing imports,
    # helpers, sibling utilities) stays intact. ``inspect.getsource(cls)``
    # alone returns just the class body — names like ``Any`` and the
    # tool-call decision helpers vanish on reload.
    #
    # When the original module is not importable from disk (anonymous /
    # closure / dynamically-defined class), fall back to the full module
    # source — which is heavier but preserves correctness.
    src = _render_hook_source(cls, warnings, strict)
    (hook_dir / "hook.py").write_text(
        "# AUTOGENERATED from preset_to_workspace.\n"
        f"# Original class: {cls.__module__}.{cls_name}\n"
        f"{src}\n",
        encoding="utf-8",
    )

    # Constructor kwargs: prefer hook.to_config(); else dataclasses.asdict;
    # else empty (caller will supply via workspace edit).
    cfg_payload: dict[str, Any] = {"class_name": cls_name}
    if hasattr(hook, "to_config") and callable(hook.to_config):
        try:
            cfg_payload["kwargs"] = hook.to_config()
        except Exception as exc:  # noqa: BLE001
            msg = f"hook {cls_name!r}.to_config() raised: {exc!r}"
            if strict:
                raise WorkspaceSerializationError(msg) from exc
            warnings.append(msg)
            cfg_payload["kwargs"] = {}
    else:
        cfg_payload["kwargs"] = {}

    (hook_dir / "config.yaml").write_text(_dump_yaml(cfg_payload) + "\n", encoding="utf-8")


def _render_hook_source(cls: type, warnings: list[str], strict: bool) -> str:
    """Render a self-contained hook.py source for ``cls``.

    Preference order:
      1. Importable module → re-import the class by name.
      2. Loaded from a workspace ``_chw_hook_*`` dynamic module → walk
         the MRO to find an importable base class and re-import that.
      3. Module source available → dump full module (preserves imports).
      4. Class source only → dump source with a typing-import fallback.
    """
    cls_name = cls.__name__
    module_name = cls.__module__ or ""

    # 1. Try to re-import — works for installed packages, top-level classes
    #    in importable modules, and anything addressable by ``module:name``.
    if (
        module_name
        and module_name not in {"__main__", "builtins"}
        and not module_name.startswith("_chw_")
    ):
        try:
            mod = importlib.import_module(module_name)
            if getattr(mod, cls_name, None) is cls:
                return (
                    "# Re-imported from the original module so the class's full\n"
                    "# closure (typing imports, helpers) stays available.\n"
                    "# To customize, replace this import with a class definition.\n"
                    f"from {module_name} import {cls_name}\n"
                )
        except Exception:  # noqa: BLE001
            pass

    # 2. Class came from a workspace ``_chw_hook_*`` dynamic module
    #    (re-loaded from disk). Walk the MRO to find an importable
    #    parent class — workspace hook files commonly subclass an
    #    installed class to add ``to_config()``, so the ancestor is
    #    a stable re-import target.
    for ancestor in cls.__mro__[1:]:
        anc_module = ancestor.__module__ or ""
        anc_name = ancestor.__name__
        if (
            not anc_module
            or anc_module in {"__main__", "builtins", "object"}
            or anc_module.startswith("_chw_")
        ):
            continue
        try:
            mod = importlib.import_module(anc_module)
            if getattr(mod, anc_name, None) is ancestor:
                # Re-export under the original subclass name so config.yaml
                # ``class_name`` lookups still resolve.
                return (
                    f"# Re-imported via base class {anc_module}.{anc_name}\n"
                    f"# (subclass {cls_name!r} was loaded from a workspace\n"
                    f"# _chw_hook_* dynamic module that has no on-disk source).\n"
                    f"from {anc_module} import {anc_name} as {cls_name}\n"
                )
        except Exception:  # noqa: BLE001
            continue

    # 3. Fall back to dumping the FULL module source (captures imports,
    #    sibling helpers). Heavier but correct for hooks defined inline
    #    in a script the user might still want to edit on disk.
    try:
        mod = inspect.getmodule(cls)
        if mod is not None:
            mod_src = inspect.getsource(mod)
            return (
                "# Full module source preserved so all imports / helpers\n"
                "# the hook references stay available on reload.\n"
                f"{mod_src}"
            )
    except (OSError, TypeError):
        pass

    # 4. Last resort: just the class source. Likely needs hand-editing on
    #    reload to add `from typing import Any` etc.
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        # ``TypeError: <class X> is a built-in class`` is what
        # ``inspect.getsource`` raises for classes loaded from the
        # workspace's own ``_chw_hook_*`` dynamic modules — those
        # have no on-disk source ``inspect`` can find.
        msg = f"hook class {cls_name!r} has no retrievable source"
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)
        return f"# AUTOGENERATED PLACEHOLDER\nclass {cls_name}:\n    pass\n"


def _write_memory(
    source: Any, memory_root: Path, index: int, warnings: list[str], strict: bool
) -> None:
    if isinstance(source, StaticMemorySource):
        (memory_root / f"{index:02d}_static.md").write_text(source.text, encoding="utf-8")
        return
    # CallableMemorySource: if the wrapped callable is a top-level
    # importable function, emit a ``<index>_callable.py`` that re-imports
    # it. The loader recognises ``*.py`` files in memory/ and wraps the
    # exported ``load`` callable with ``CallableMemorySource``. Closures
    # / lambdas fall through to the generic warning path because they
    # cannot be re-imported.
    from looplet.memory import CallableMemorySource  # noqa: PLC0415

    if isinstance(source, CallableMemorySource):
        fn = source.fn
        fn_name = getattr(fn, "__name__", "")
        fn_mod = getattr(fn, "__module__", "") or ""
        fn_qual = getattr(fn, "__qualname__", "<lambda>")
        if (
            fn_name
            and fn_mod
            and fn_mod not in ("builtins",)
            and not fn_mod.startswith("_chw_")
            and fn_qual != "<lambda>"
            and "<locals>" not in fn_qual
        ):
            (memory_root / f"{index:02d}_callable.py").write_text(
                "# AUTOGENERATED CallableMemorySource builder.\n"
                "# The exported ``load`` callable receives the loop's\n"
                "# ``state`` on every turn and returns the memory text\n"
                "# (or ``None`` to skip). Re-imported from the source\n"
                "# module so its closure stays intact.\n"
                f"from {fn_mod} import {fn_name} as load\n",
                encoding="utf-8",
            )
            if fn_mod == "__main__":
                warnings.append(
                    f"memory source {index!r}: CallableMemorySource wraps "
                    f"a ``__main__`` callable {fn_name!r}; cross-process "
                    f"loads will fail until it is moved to a real module"
                )
            return
        msg = (
            f"memory source 'CallableMemorySource' wraps a non-importable "
            f"callable ({fn_qual!r} from {fn_mod!r}); skipping"
        )
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)
        return

    name = type(source).__name__
    msg = f"memory source {name!r} is not a StaticMemorySource; skipping"
    if strict:
        raise WorkspaceSerializationError(msg)
    warnings.append(msg)


# ── Deserialise: directory → AgentPreset ───────────────────────


def workspace_to_preset(
    workspace_dir: str | Path,
    *,
    state_factory: Callable[[int], Any] | None = None,
    strict: bool = False,
    runtime: dict[str, Any] | None = None,
) -> "AgentPreset":
    """Read a CHW directory and materialise an :class:`AgentPreset`.

    Args:
        workspace_dir: Path to the workspace root.
        state_factory: Builds the runtime ``state`` from ``max_steps``.
            Defaults to ``DefaultState(max_steps=...)``.
        strict: When ``True``, raise :class:`WorkspaceSerializationError`
            on any tool / hook that fails to load (e.g. a hook whose
            ``config.yaml`` lacks the kwargs its constructor needs).
            When ``False`` (default), drop the offender, log a warning,
            and continue. Use ``strict=True`` for round-trip
            verification and CI lint.
        runtime: Optional dict of host-supplied runtime values
            (e.g. ``{"workspace": "/tmp/myrepo"}`` for the coder
            cartridge). Three integration points read it:
              * ``${runtime.<key>}`` placeholders in ``config.yaml``
                are substituted before constructing ``LoopConfig``.
              * ``resources/<name>.py`` builders that declare
                ``def build(runtime)`` (or ``**kwargs``) receive it.
              * ``setup.py``'s ``setup(...)`` receives it via the
                ``runtime`` kwarg when its signature accepts it.
    """
    from looplet.loop import LoopConfig  # noqa: PLC0415
    from looplet.presets import AgentPreset  # noqa: PLC0415
    from looplet.tools import BaseToolRegistry, ToolSpec  # noqa: PLC0415
    from looplet.types import DefaultState  # noqa: PLC0415

    root = Path(workspace_dir)
    if not (root / WorkspaceLayout.WORKSPACE_JSON).is_file():
        raise FileNotFoundError(
            f"workspace metadata not found at "
            f"{root / WorkspaceLayout.WORKSPACE_JSON}; "
            f"is this a Composable Harness Workspace?"
        )

    # Shared-resource registry — built once, referenced by ``@<name>``
    # strings throughout hook / tool kwargs. Lets two hooks share the
    # same live object (e.g. a FileCache) on reload, instead of
    # silently splitting into two independent instances.
    runtime_dict = dict(runtime or {})
    resources = _load_resources(root, runtime_dict)

    # Config
    cfg_kwargs: dict[str, Any] = {}
    cfg_path = root / WorkspaceLayout.CONFIG_YAML
    if cfg_path.is_file():
        raw_cfg_text = cfg_path.read_text(encoding="utf-8")
        # Apply ``${runtime.<key>}`` substitution before YAML parsing so
        # workspace authors can parameterise config.yaml without needing
        # a setup.py for the common cases.
        raw_cfg_text = _apply_runtime_substitutions(raw_cfg_text, runtime_dict)
        cfg_kwargs.update(_load_yaml(raw_cfg_text) or {})

    sys_prompt_path = root / WorkspaceLayout.SYSTEM_PROMPT_MD
    if sys_prompt_path.is_file():
        cfg_kwargs["system_prompt"] = sys_prompt_path.read_text(encoding="utf-8")

    # Memory sources — ``*.md`` → StaticMemorySource, ``*.py`` →
    # CallableMemorySource (the module's ``load`` attr is wrapped). Files
    # are loaded in lexicographic order so the writer's ``00_``, ``01_``
    # prefix preserves source order.
    memory_sources: list[PersistentMemorySource] = []
    memory_dir = root / WorkspaceLayout.MEMORY_DIR
    if memory_dir.is_dir():
        from looplet.memory import CallableMemorySource  # noqa: PLC0415

        memory_files = sorted(
            p for p in memory_dir.iterdir() if p.is_file() and p.suffix in (".md", ".py")
        )
        for memory_file in memory_files:
            if memory_file.suffix == ".md":
                memory_sources.append(
                    StaticMemorySource(text=memory_file.read_text(encoding="utf-8"))
                )
            else:
                module = _import_module_from_path(memory_file, f"_chw_memory_{memory_file.stem}")
                load_fn = getattr(module, "load", None)
                if not callable(load_fn):
                    msg = (
                        f"memory module {memory_file.name!r} must export a ``load(state)`` callable"
                    )
                    if strict:
                        raise WorkspaceSerializationError(msg)
                    logger.warning("%s; skipping", msg)
                    continue
                memory_sources.append(CallableMemorySource(fn=load_fn))  # type: ignore[arg-type]
    if memory_sources:
        cfg_kwargs["memory_sources"] = memory_sources

    # Resolve ``"@<name>"`` references in config kwargs against the
    # shared-resource registry so callable / opaque LoopConfig fields
    # (tracer, router, compact_service, recovery_registry, cache_policy,
    # approval_handler, domain, build_briefing, output_schema, …) can be
    # wired declaratively from ``resources/<name>.py`` builders instead
    # of forcing every workspace into a ``setup.py`` detour. Symmetric
    # with the hook-kwargs ref resolution below.
    cfg_kwargs = _resolve_refs(cfg_kwargs, resources)

    config = LoopConfig(**cfg_kwargs)

    # Track tool + hook modules so setup.py can wire shared resources
    # into them after the declarative load (see ``setup.py`` block below).
    tool_modules: dict[str, Any] = {}
    hook_modules: dict[str, Any] = {}

    # Tools
    registry = BaseToolRegistry()
    tools_dir = root / WorkspaceLayout.TOOLS_DIR
    if tools_dir.is_dir():
        for tool_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
            spec_path = tool_dir / "tool.yaml"
            execute_path = tool_dir / "execute.py"
            if not spec_path.is_file() or not execute_path.is_file():
                msg = f"malformed tool dir {tool_dir} (missing tool.yaml or execute.py)"
                if strict:
                    raise WorkspaceSerializationError(msg)
                logger.warning("skipping %s", msg)
                continue
            yaml_payload = (
                _load_yaml(
                    _apply_runtime_substitutions(
                        spec_path.read_text(encoding="utf-8"), runtime_dict
                    )
                )
                or {}
            )
            module = _import_module_from_path(execute_path, f"_chw_tool_{tool_dir.name}")
            tool_modules[tool_dir.name] = module
            execute_fn = getattr(module, "execute", None)
            if execute_fn is None:
                # Fall back to the function whose name matches the YAML name.
                execute_fn = getattr(module, str(yaml_payload.get("name", "")), None)
            if not callable(execute_fn):
                msg = (
                    f"tool {tool_dir.name!r} has no callable execute "
                    f"(looked for `execute` and `{yaml_payload.get('name', '')}` in {execute_path})"
                )
                if strict:
                    raise WorkspaceSerializationError(msg)
                logger.warning("%s; skipping", msg)
                continue
            spec = ToolSpec(
                name=str(yaml_payload.get("name", tool_dir.name)),
                description=str(yaml_payload.get("description", "")),
                parameters=dict(yaml_payload.get("parameters", {}) or {}),
                execute=execute_fn,
                concurrent_safe=bool(yaml_payload.get("concurrent_safe", False)),
                free=bool(yaml_payload.get("free", False)),
                timeout_s=yaml_payload.get("timeout_s"),
            )
            registry.register(spec)

    # Hooks (alphabetical-by-dirname → list order).
    hooks: list[Any] = []
    hooks_dir = root / WorkspaceLayout.HOOKS_DIR
    if hooks_dir.is_dir():
        for hook_dir in sorted(p for p in hooks_dir.iterdir() if p.is_dir()):
            hook_py = hook_dir / "hook.py"
            cfg_yaml = hook_dir / "config.yaml"
            if not hook_py.is_file():
                msg = f"malformed hook dir {hook_dir} (missing hook.py)"
                if strict:
                    raise WorkspaceSerializationError(msg)
                logger.warning("skipping %s", msg)
                continue
            module = _import_module_from_path(hook_py, f"_chw_hook_{hook_dir.name}")
            hook_modules[hook_dir.name] = module
            hook_cfg = (
                _load_yaml(
                    _apply_runtime_substitutions(cfg_yaml.read_text(encoding="utf-8"), runtime_dict)
                )
                if cfg_yaml.is_file()
                else {}
            ) or {}
            class_name = str(hook_cfg.get("class_name") or "")
            if not class_name:
                # Pick the first class defined in the module.
                classes = [
                    obj
                    for name, obj in inspect.getmembers(module, inspect.isclass)
                    if obj.__module__ == module.__name__
                ]
                if not classes:
                    msg = f"hook {hook_dir.name!r} has no class in {hook_py}"
                    if strict:
                        raise WorkspaceSerializationError(msg)
                    logger.warning("%s; skipping", msg)
                    continue
                cls = classes[0]
            else:
                cls = getattr(module, class_name, None)
                if cls is None:
                    msg = (
                        f"hook {hook_dir.name!r} declares class_name={class_name!r} "
                        f"but module has no such class"
                    )
                    if strict:
                        raise WorkspaceSerializationError(msg)
                    logger.warning("%s; skipping", msg)
                    continue
            kwargs = dict(hook_cfg.get("kwargs", {}) or {})
            # Resolve ``"@<name>"`` references against the shared-resource
            # registry so hooks can share live objects on reload.
            kwargs = _resolve_refs(kwargs, resources)
            try:
                hooks.append(cls(**kwargs))
            except TypeError as exc:
                msg = (
                    f"hook {hook_dir.name!r} ({class_name or cls.__name__}) could not be "
                    f"instantiated with kwargs={kwargs}: {exc}. "
                    f"Implement to_config(self) -> dict on the hook class so the "
                    f"workspace round-trip can capture its constructor kwargs."
                )
                if strict:
                    raise WorkspaceSerializationError(msg) from exc
                logger.warning("%s; skipping hook", msg)

    # State
    max_steps = int(getattr(config, "max_steps", 15))
    state = (
        state_factory(max_steps) if state_factory is not None else DefaultState(max_steps=max_steps)
    )

    preset = AgentPreset(config=config, hooks=hooks, tools=registry, state=state)

    # ``setup.py`` escape hatch — runs after the declarative load to
    # let the cartridge attach callable / opaque fields that don't
    # round-trip via JSON (e.g. ``LoopConfig.tracer``,
    # ``LoopConfig.compact_service``, custom domain adapters), or
    # inject shared resources into top-level tool/hook modules.
    setup_path = root / WorkspaceLayout.SETUP_PY
    if setup_path.is_file():
        module = _import_module_from_path(setup_path, "_chw_setup")
        setup_fn = getattr(module, "setup", None)
        if not callable(setup_fn):
            raise WorkspaceSerializationError(
                f"workspace setup.py at {setup_path} must define "
                f"`def setup(preset, resources, tool_modules, hook_modules)`"
            )
        # Modern signature accepts (preset, resources, tool_modules,
        # hook_modules); the older 2-arg signature still works for
        # forward compatibility — inspect.signature picks the right one.
        import inspect as _i  # noqa: PLC0415

        sig_params = _i.signature(setup_fn).parameters
        kwargs: dict[str, Any] = {}
        if "tool_modules" in sig_params:
            kwargs["tool_modules"] = tool_modules
        if "hook_modules" in sig_params:
            kwargs["hook_modules"] = hook_modules
        if "runtime" in sig_params:
            kwargs["runtime"] = runtime_dict
        result = setup_fn(preset, resources, **kwargs)
        if isinstance(result, AgentPreset):
            preset = result

    return preset
