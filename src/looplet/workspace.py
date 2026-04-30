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


def _safe_filename(name: str) -> str:
    """Sanitise an arbitrary string into a directory-safe filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "unnamed"


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

    for fname in WorkspaceLayout.NON_SERIALIZABLE_CONFIG_FIELDS:
        if not hasattr(cfg, fname):
            continue
        if getattr(cfg, fname) is None:
            continue
        msg = f"config.{fname!r} is set but not round-trippable; skipping"
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)

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

    workspace.serialization_warnings = warnings
    workspace.write_metadata()
    return workspace


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
    except OSError:
        msg = f"tool {name!r} execute has no retrievable source"
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)
        return

    # Add an `execute = <fn_name>` alias so the loader finds it under the
    # canonical name regardless of what the original function was called.
    fn_name = getattr(fn, "__name__", "")
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

    # Source code of the hook class.
    try:
        source = inspect.getsource(cls)
    except OSError:
        msg = f"hook class {cls_name!r} has no retrievable source"
        if strict:
            raise WorkspaceSerializationError(msg)
        warnings.append(msg)
        source = f"# AUTOGENERATED PLACEHOLDER\nclass {cls_name}:\n    pass\n"
    (hook_dir / "hook.py").write_text(
        "# AUTOGENERATED from preset_to_workspace.\n"
        "# The class below is loaded by name from this module.\n"
        f"{source}\n",
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


def _write_memory(
    source: Any, memory_root: Path, index: int, warnings: list[str], strict: bool
) -> None:
    if isinstance(source, StaticMemorySource):
        (memory_root / f"{index:02d}_static.md").write_text(source.text, encoding="utf-8")
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
) -> "AgentPreset":
    """Read a CHW directory and materialise an :class:`AgentPreset`.

    Args:
        workspace_dir: Path to the workspace root.
        state_factory: Builds the runtime ``state`` from ``max_steps``.
            Defaults to ``DefaultState(max_steps=...)``.
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

    # Config
    cfg_kwargs: dict[str, Any] = {}
    cfg_path = root / WorkspaceLayout.CONFIG_YAML
    if cfg_path.is_file():
        cfg_kwargs.update(_load_yaml(cfg_path.read_text(encoding="utf-8")) or {})

    sys_prompt_path = root / WorkspaceLayout.SYSTEM_PROMPT_MD
    if sys_prompt_path.is_file():
        cfg_kwargs["system_prompt"] = sys_prompt_path.read_text(encoding="utf-8")

    # Memory sources (StaticMemorySource per file).
    memory_sources: list[PersistentMemorySource] = []
    memory_dir = root / WorkspaceLayout.MEMORY_DIR
    if memory_dir.is_dir():
        for memory_file in sorted(memory_dir.glob("*.md")):
            memory_sources.append(StaticMemorySource(text=memory_file.read_text(encoding="utf-8")))
    if memory_sources:
        cfg_kwargs["memory_sources"] = memory_sources

    config = LoopConfig(**cfg_kwargs)

    # Tools
    registry = BaseToolRegistry()
    tools_dir = root / WorkspaceLayout.TOOLS_DIR
    if tools_dir.is_dir():
        for tool_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
            spec_path = tool_dir / "tool.yaml"
            execute_path = tool_dir / "execute.py"
            if not spec_path.is_file() or not execute_path.is_file():
                logger.warning("skipping malformed tool dir %s", tool_dir)
                continue
            yaml_payload = _load_yaml(spec_path.read_text(encoding="utf-8")) or {}
            module = _import_module_from_path(execute_path, f"_chw_tool_{tool_dir.name}")
            execute_fn = getattr(module, "execute", None)
            if execute_fn is None:
                # Fall back to the function whose name matches the YAML name.
                execute_fn = getattr(module, str(yaml_payload.get("name", "")), None)
            if not callable(execute_fn):
                logger.warning("tool %s has no callable execute; skipping", tool_dir)
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
                logger.warning("skipping malformed hook dir %s", hook_dir)
                continue
            module = _import_module_from_path(hook_py, f"_chw_hook_{hook_dir.name}")
            hook_cfg = (
                _load_yaml(cfg_yaml.read_text(encoding="utf-8")) if cfg_yaml.is_file() else {}
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
                    logger.warning("hook %s has no class; skipping", hook_dir)
                    continue
                cls = classes[0]
            else:
                cls = getattr(module, class_name, None)
                if cls is None:
                    logger.warning(
                        "hook %s declares class_name=%s but module has no such class",
                        hook_dir,
                        class_name,
                    )
                    continue
            kwargs = dict(hook_cfg.get("kwargs", {}) or {})
            try:
                hooks.append(cls(**kwargs))
            except TypeError as exc:
                logger.warning(
                    "hook %s could not be instantiated with kwargs=%s: %r",
                    hook_dir,
                    kwargs,
                    exc,
                )

    # State
    max_steps = int(getattr(config, "max_steps", 15))
    state = (
        state_factory(max_steps) if state_factory is not None else DefaultState(max_steps=max_steps)
    )

    return AgentPreset(config=config, hooks=hooks, tools=registry, state=state)
