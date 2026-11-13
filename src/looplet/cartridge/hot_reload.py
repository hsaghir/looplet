"""Hot-reload and capability-overlay synchronization for cartridge presets.

Supports two workflows without third-party filesystem-watch dependencies:

1. **Between runs** - call :meth:`WorkspaceWatcher.reload_if_changed` after
   a :func:`~looplet.loop.composable_loop` invocation finishes.
2. **At step boundaries inside a run** - attach :class:`ReloadHook` to
   ``hooks=[...]``. Before each prompt turn (after prior tool calls finish
   and before tool schemas and prompt cache breakpoints are built), the hook
   checks whether watched base or overlay directories changed, validates a
   staging preset with ``strict=True``, and atomically updates active
   overlay capabilities or surfaces a structured failure observation while
   retaining the last valid preset.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from looplet.cartridge._layout import CartridgeLayout
from looplet.cartridge._load import cartridge_to_preset
from looplet.hook_decision import HookDecision, InjectContext
from looplet.validation import ValidatingToolRegistry

__all__ = [
    "ReloadHook",
    "ReloadObservation",
    "WorkspaceWatcher",
    "fingerprint_workspace",
]


# Files whose contents can change agent behaviour in a base cartridge.
_WATCHED_SUFFIXES: tuple[str, ...] = (".py", ".yaml", ".yml", ".md", ".json")


def fingerprint_workspace(root: str | Path) -> dict[str, str]:
    """Return ``{relpath: fingerprint}`` for every watched file under ``root``.

    Each fingerprint combines ``mtime_ns``, ``size``, and a 64-bit BLAKE2b
    content digest so rapid writes on coarse-timestamp filesystems are still
    detected. Returns ``{}`` if ``root`` does not exist.
    """
    base = Path(root)
    if base.is_symlink() or not base.is_dir():
        return {}
    fp: dict[str, str] = {}
    for p in sorted(base.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        if p.suffix.lower() not in _WATCHED_SUFFIXES:
            continue
        rel = p.relative_to(base)
        parts = rel.parts
        if any(part.startswith("__pycache__") or part.startswith(".") for part in parts):
            continue
        try:
            stat = p.stat()
            data = p.read_bytes()
        except OSError:
            continue
        digest = hashlib.blake2b(data, digest_size=8).hexdigest()
        fp[rel.as_posix()] = f"{stat.st_mtime_ns}:{stat.st_size}:{digest}"
    return fp


def _fingerprint_overlay_layer(root: Path) -> dict[str, str]:
    """Fingerprint an overlay path including structural or invalid entries.

    Unlike :func:`fingerprint_workspace`, overlay fingerprinting also records
    symlinks, hidden files, unrecognized files, and empty directories so
    invalid overlay edits still trigger validation and surface an explicit
    rejection rather than being silently ignored.
    """
    if root.is_symlink():
        return {"__root__": f"symlink:{root.readlink()}"}
    if not root.exists():
        return {}
    if not root.is_dir():
        return {"__root__": "non_directory"}
    fp: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        posix = rel.as_posix()
        if p.is_symlink():
            try:
                target = str(p.readlink())
            except OSError:
                target = "unreadable"
            fp[posix] = f"symlink:{target}"
        elif p.is_dir():
            fp[posix] = "dir"
        elif p.is_file():
            try:
                stat = p.stat()
                data = p.read_bytes()
            except OSError:
                fp[posix] = "unreadable_file"
                continue
            digest = hashlib.blake2b(data, digest_size=8).hexdigest()
            fp[posix] = f"{stat.st_mtime_ns}:{stat.st_size}:{digest}"
        else:
            fp[posix] = "special"
    return fp


def _content_digest_layer(root: Path, *, include_unwatched: bool) -> str:
    """Compute an mtime-independent SHA-256 digest of a base or overlay layer.

    Content digests remain stable when identical files are restored into a
    fresh container or temporary workspace with new modification timestamps,
    while still changing on any byte, path, directory, or symlink edit.
    """
    h = hashlib.sha256()
    if root.is_symlink():
        try:
            target = str(root.readlink())
        except OSError:
            target = "unreadable"
        h.update(f"ROOT_SYMLINK:{target}\n".encode("utf-8"))
        return h.hexdigest()
    if not root.exists():
        h.update(b"ABSENT\n")
        return h.hexdigest()
    if not root.is_dir():
        h.update(b"NON_DIR\n")
        return h.hexdigest()
    h.update(b"DIR\n")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        parts = rel.parts
        if "__pycache__" in parts or p.suffix == ".pyc":
            continue
        if not include_unwatched and any(part.startswith(".") for part in parts):
            continue
        posix_bytes = rel.as_posix().encode("utf-8")
        if p.is_symlink():
            if not include_unwatched:
                continue
            try:
                target = str(p.readlink())
            except OSError:
                target = "unreadable"
            h.update(b"L")
            h.update(len(posix_bytes).to_bytes(4, "big"))
            h.update(posix_bytes)
            target_bytes = target.encode("utf-8")
            h.update(len(target_bytes).to_bytes(4, "big"))
            h.update(target_bytes)
        elif p.is_dir():
            if not include_unwatched:
                continue
            h.update(b"D")
            h.update(len(posix_bytes).to_bytes(4, "big"))
            h.update(posix_bytes)
        elif p.is_file():
            if not include_unwatched and p.suffix.lower() not in _WATCHED_SUFFIXES:
                continue
            try:
                data = p.read_bytes()
            except OSError:
                h.update(b"E")
                h.update(len(posix_bytes).to_bytes(4, "big"))
                h.update(posix_bytes)
                continue
            h.update(b"F")
            h.update(len(posix_bytes).to_bytes(4, "big"))
            h.update(posix_bytes)
            h.update(len(data).to_bytes(8, "big"))
            h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


@dataclass(frozen=True)
class ReloadObservation:
    """Structured step-boundary reload outcome for provenance and model context."""

    step_num: int
    status: Literal["reloaded", "rejected"]
    active_digest: str
    candidate_digest: str
    added_tools: tuple[str, ...] = ()
    updated_tools: tuple[str, ...] = ()
    removed_tools: tuple[str, ...] = ()
    active_tools: tuple[str, ...] = ()
    prompt_changed: bool = False
    memory_changed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_num": self.step_num,
            "status": self.status,
            "active_digest": self.active_digest,
            "candidate_digest": self.candidate_digest,
            "added_tools": list(self.added_tools),
            "updated_tools": list(self.updated_tools),
            "removed_tools": list(self.removed_tools),
            "active_tools": list(self.active_tools),
            "prompt_changed": self.prompt_changed,
            "memory_changed": self.memory_changed,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def format_briefing(self) -> str:
        """Render a concise factual observation for the upcoming model turn."""
        if self.status == "rejected":
            return (
                "[CAPABILITY RELOAD REJECTED]\n"
                f"status: rejected\n"
                f"error: {self.error}\n"
                f"active_digest: {self.active_digest[:16]}\n"
                f"candidate_digest: {self.candidate_digest[:16]}\n"
                f"active_tools: {', '.join(self.active_tools) or '(none)'}"
            )
        changes: list[str] = []
        if self.added_tools:
            changes.append(f"added_tools={list(self.added_tools)}")
        if self.updated_tools:
            changes.append(f"updated_tools={list(self.updated_tools)}")
        if self.removed_tools:
            changes.append(f"removed_tools={list(self.removed_tools)}")
        if self.prompt_changed:
            changes.append("prompt_changed=true")
        if self.memory_changed:
            changes.append("memory_changed=true")
        summary = ", ".join(changes) if changes else "no_effective_delta"
        return (
            "[CAPABILITY RELOAD APPLIED]\n"
            f"status: reloaded ({summary})\n"
            f"active_digest: {self.active_digest[:16]}\n"
            f"active_tools: {', '.join(self.active_tools) or '(none)'}"
        )


@dataclass
class WorkspaceWatcher:
    """Poll-based watcher over a base cartridge and optional overlay directories.

    Args:
        root: Path to the base cartridge directory.
        runtime: Forwarded to :func:`cartridge_to_preset` on every (re)load.
        strict: Forwarded to :func:`cartridge_to_preset` (default ``False``).
        overlays: Optional sequence of sparse overlay directories layered
            onto ``root`` in order. Missing overlay directories are treated
            as empty until created; once created, changes trigger reloads.
        allowed_overlay_slots: Optional subset of
            :attr:`CartridgeLayout.OVERLAY_SLOTS` enforced on ``overlays``.
        protected_tools: Tool names that ``overlays`` may not add or override.
    """

    root: str | Path
    runtime: dict[str, Any] | None = None
    strict: bool = False
    overlays: Sequence[str | Path] = ()
    allowed_overlay_slots: Collection[str] | None = None
    protected_tools: Collection[str] = ()
    _fp: dict[str, str] = field(default_factory=dict, init=False)
    _preset: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.overlays, (str, bytes)):
            raise TypeError("overlays must be a sequence of paths, not a bare string")
        self.overlays = tuple(Path(p) for p in self.overlays)
        if self.allowed_overlay_slots is not None:
            if isinstance(self.allowed_overlay_slots, (str, bytes)):
                raise TypeError("allowed_overlay_slots must be a collection of slot names")
            slots = frozenset(self.allowed_overlay_slots)
            unknown = sorted(slots - CartridgeLayout.OVERLAY_SLOTS)
            if unknown:
                raise ValueError(
                    f"unknown overlay slot(s) {unknown}; expected subset of "
                    f"{sorted(CartridgeLayout.OVERLAY_SLOTS)}"
                )
            self.allowed_overlay_slots = slots
        if isinstance(self.protected_tools, (str, bytes)):
            raise TypeError("protected_tools must be a collection of tool names")
        self.protected_tools = frozenset(self.protected_tools)

    def fingerprint(self) -> dict[str, str]:
        """Return a deterministic content fingerprint across base and overlays."""
        if not self.overlays:
            return fingerprint_workspace(self.root)
        combined = {f"base:{k}": v for k, v in fingerprint_workspace(self.root).items()}
        for index, overlay in enumerate(self.overlays):
            for key, value in _fingerprint_overlay_layer(Path(overlay)).items():
                combined[f"overlay[{index}]:{key}"] = value
        return combined

    def digest(self) -> str:
        """Return an mtime-independent SHA-256 content digest across base and overlays."""
        h = hashlib.sha256()
        h.update(b"base:")
        h.update(_content_digest_layer(Path(self.root), include_unwatched=False).encode("ascii"))
        h.update(b"\n")
        for index, overlay in enumerate(self.overlays):
            h.update(f"overlay[{index}]:".encode("ascii"))
            h.update(_content_digest_layer(Path(overlay), include_unwatched=True).encode("ascii"))
            h.update(b"\n")
        return h.hexdigest()

    def _active_overlay_paths(self) -> list[Path]:
        active: list[Path] = []
        for item in self.overlays:
            path = Path(item)
            if path.is_symlink() or path.exists():
                active.append(path)
        return active

    def _load_preset(
        self,
        *,
        strict: bool | None = None,
        extra_protected_tools: Collection[str] = (),
    ) -> Any:
        merged_protected = set(self.protected_tools) | set(extra_protected_tools)
        return cartridge_to_preset(
            self.root,
            overlays=self._active_overlay_paths(),
            allowed_overlay_slots=self.allowed_overlay_slots,
            protected_tools=merged_protected,
            strict=self.strict if strict is None else strict,
            runtime=self.runtime,
        )

    def validate(
        self,
        *,
        extra_protected_tools: Collection[str] = (),
    ) -> dict[str, Any]:
        """Validate current base + overlay files with ``strict=True`` without mutating state.

        Loads and immediately closes a staging preset so authoring tools can
        preflight edits and report structured errors before the step boundary.
        """
        staging = self._load_preset(
            strict=True,
            extra_protected_tools=extra_protected_tools,
        )
        try:
            tool_names = list(staging.tools.tool_names)
            overlay_tools = [spec.name for spec in staging.tools.tools_for_origin("overlay")]
            return {
                "valid": True,
                "digest": self.digest(),
                "tools": tool_names,
                "overlay_tools": overlay_tools,
            }
        finally:
            staging.close()

    def _commit_preset(
        self, replacement: Any, fp: dict[str, str], *, close_previous: bool = True
    ) -> Any:
        previous = self._preset
        self._preset = replacement
        self._fp = fp
        if close_previous and previous is not None and previous is not replacement:
            previous.close()
        return replacement

    def preset(self) -> Any:
        """Return the current preset, loading it on first call."""
        if self._preset is None:
            fp = self.fingerprint()
            self._preset = self._load_preset()
            self._fp = fp
        return self._preset

    def changed(self) -> bool:
        """True if the workspace or overlay fingerprint has shifted since last load."""
        if self._preset is None:
            return True
        return self.fingerprint() != self._fp

    def reload_if_changed(self) -> Any | None:
        """Return a fresh preset when watched files changed, else ``None``."""
        if not self.changed():
            return None
        fp = self.fingerprint()
        replacement = self._load_preset()
        return self._commit_preset(replacement, fp)

    def force_reload(self) -> Any:
        """Reload regardless of fingerprint (for tests / debug)."""
        fp = self.fingerprint()
        replacement = self._load_preset()
        return self._commit_preset(replacement, fp)

    def close(self) -> None:
        """Close the currently loaded preset, if any."""
        preset, self._preset = self._preset, None
        self._fp = {}
        if preset is not None:
            preset.close()

    def __enter__(self) -> "WorkspaceWatcher":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class ReloadHook:
    """Step-boundary capability reload hook backed by :class:`WorkspaceWatcher`.

    Checks for base/overlay changes in ``pre_prompt`` before tool schemas,
    prompts, and cache breakpoints are computed for the upcoming turn.
    Candidate changes are validated with ``strict=True`` before mutating live
    loop state; broken edits leave the previous valid capabilities active and
    emit a structured :class:`ReloadObservation`.
    """

    _checkpoint_schema = "looplet.reload-hook.v1"

    def __init__(
        self,
        watcher: WorkspaceWatcher,
        *,
        protect_initial_tools: bool = True,
        protected_tools: Collection[str] = (),
        tool_transform: Callable[[Any], Any] | None = None,
        notify_on_reload: bool = True,
        notify_on_error: bool = True,
    ) -> None:
        if isinstance(protected_tools, (str, bytes)):
            raise TypeError("protected_tools must be a collection of tool names")
        if tool_transform is not None and not callable(tool_transform):
            raise TypeError("tool_transform must be callable or None")
        self.watcher = watcher
        self.protect_initial_tools = protect_initial_tools
        self._explicit_protected_tools: frozenset[str] = frozenset(protected_tools)
        self._protected_tools: set[str] = set(watcher.protected_tools) | set(
            self._explicit_protected_tools
        )
        self.tool_transform = tool_transform
        self.notify_on_reload = notify_on_reload
        self.notify_on_error = notify_on_error
        self._loop_ctx: Any = None
        self._initial_run_preset: Any = None
        self._host_memory_sources: list[Any] = []
        self._cartridge_resource_keys: set[str] = set()
        self._active_fingerprint: dict[str, str] = {}
        self._active_digest: str = ""
        self._last_attempted_digest: str = ""
        self.reload_count: int = 0
        self.last_error: str | None = None
        self.observations: list[ReloadObservation] = []

    def provenance_config(self) -> dict[str, Any]:
        """Return JSON-safe configuration for :func:`~looplet.serialize_harness`."""
        slots = self.watcher.allowed_overlay_slots
        cfg: dict[str, Any] = {
            "overlay_count": len(self.watcher.overlays),
            "allowed_overlay_slots": sorted(slots) if slots is not None else None,
            "protect_initial_tools": self.protect_initial_tools,
            "protected_tools": sorted(self._explicit_protected_tools),
            "notify_on_reload": self.notify_on_reload,
            "notify_on_error": self.notify_on_error,
        }
        if self.tool_transform is not None:
            cfg["tool_transform"] = self.tool_transform
        return cfg

    def bind(self, loop_ctx: Any) -> None:
        self._loop_ctx = loop_ctx
        initial = self.watcher.preset()
        self._initial_run_preset = initial
        self._protected_tools = set(self.watcher.protected_tools) | set(
            self._explicit_protected_tools
        )
        done_tool = getattr(loop_ctx.config, "done_tool", "done")
        if isinstance(done_tool, str) and done_tool:
            self._protected_tools.add(done_tool)
        for secondary in getattr(loop_ctx.config, "done_tools", ()) or ():
            if isinstance(secondary, str) and secondary:
                self._protected_tools.add(secondary)
        if self.protect_initial_tools:
            for name, spec in loop_ctx.tools.tool_specs.items():
                if spec.origin != "overlay":
                    self._protected_tools.add(name)
        initial_mem_ids = {id(src) for src in initial.config.memory_sources}
        self._host_memory_sources = [
            src
            for src in getattr(loop_ctx.config, "memory_sources", ()) or ()
            if id(src) not in initial_mem_ids
        ]
        self._cartridge_resource_keys = set(initial.resources.keys())
        self._active_fingerprint = dict(self.watcher._fp)
        self._active_digest = self.watcher.digest()
        self._last_attempted_digest = self._active_digest

    @staticmethod
    def _record_on_state(state: Any, observation: ReloadObservation) -> None:
        metadata = getattr(state, "metadata", None)
        if isinstance(metadata, dict):
            reloads = metadata.setdefault("harness_reloads", [])
            if isinstance(reloads, list):
                reloads.append(observation.to_dict())

    def _apply_candidate(self, candidate: Any) -> tuple[dict[str, tuple[str, ...]], bool, bool]:
        loop_ctx = self._loop_ctx
        if loop_ctx is None:
            raise RuntimeError("ReloadHook must be bound to LoopContext before applying presets")

        tools = loop_ctx.tools
        host_resources = {
            k: v for k, v in tools.resources.items() if k not in self._cartridge_resource_keys
        }
        merged_resources = {**host_resources, **candidate.resources}
        self._cartridge_resource_keys = set(candidate.resources.keys())

        argument_schemas = (
            dict(candidate.tools._schemas)
            if isinstance(candidate.tools, ValidatingToolRegistry)
            else {}
        )

        if self.protect_initial_tools:
            overlay_specs = candidate.tools.tools_for_origin("overlay")
            if self.tool_transform is not None:
                overlay_specs = [self.tool_transform(spec) for spec in overlay_specs]
            if isinstance(tools, ValidatingToolRegistry):
                diff = tools.sync_origin(
                    "overlay",
                    overlay_specs,
                    protected=self._protected_tools,
                    resources=merged_resources,
                    argument_schemas=argument_schemas,
                )
            else:
                diff = tools.sync_origin(
                    "overlay",
                    overlay_specs,
                    protected=self._protected_tools,
                    resources=merged_resources,
                )
        else:
            # Full-cartridge developer hot-reload: sync both base cartridge
            # tools and overlay tools while preserving host-registered tools.
            base_specs = candidate.tools.tools_for_origin("cartridge")
            overlay_specs = candidate.tools.tools_for_origin("overlay")
            if self.tool_transform is not None:
                base_specs = [self.tool_transform(spec) for spec in base_specs]
                overlay_specs = [self.tool_transform(spec) for spec in overlay_specs]
            sync_kwargs: dict[str, Any] = {
                "protected": self._explicit_protected_tools,
                "allow_override_origins": ("cartridge", "overlay"),
                "resources": merged_resources,
            }
            if isinstance(tools, ValidatingToolRegistry):
                sync_kwargs["argument_schemas"] = argument_schemas
            base_diff = tools.sync_origin("cartridge", base_specs, **sync_kwargs)
            ov_diff = tools.sync_origin("overlay", overlay_specs, **sync_kwargs)
            diff = {
                "added": (*base_diff["added"], *ov_diff["added"]),
                "updated": (*base_diff["updated"], *ov_diff["updated"]),
                "removed": (*base_diff["removed"], *ov_diff["removed"]),
            }

        tools.set_resources(merged_resources)
        if isinstance(getattr(loop_ctx, "resources", None), dict):
            loop_ctx.resources.clear()
            loop_ctx.resources.update(merged_resources)

        slots = self.watcher.allowed_overlay_slots
        allow_prompts = slots is None or not self.watcher.overlays or "prompts" in slots
        prompt_changed = False
        if allow_prompts and loop_ctx.config.system_prompt != candidate.config.system_prompt:
            loop_ctx.config.system_prompt = candidate.config.system_prompt
            prompt_changed = True

        allow_memory = slots is None or not self.watcher.overlays or "memory" in slots
        memory_changed = False
        if allow_memory:
            prev_texts = [getattr(src, "text", id(src)) for src in loop_ctx.config.memory_sources]
            next_sources = [*candidate.config.memory_sources, *self._host_memory_sources]
            next_texts = [getattr(src, "text", id(src)) for src in next_sources]
            if prev_texts != next_texts:
                loop_ctx.config.memory_sources = next_sources
                memory_changed = True

        return diff, prompt_changed, memory_changed

    def pre_prompt(
        self, state: Any, session_log: Any, context: Any, step_num: int
    ) -> HookDecision | None:
        del session_log, context
        if self._loop_ctx is None:
            return None
        current_fp = self.watcher.fingerprint()
        current_digest = self.watcher.digest()
        if current_digest == self._active_digest:
            if self._last_attempted_digest != self._active_digest:
                self._last_attempted_digest = self._active_digest
                self.last_error = None
            return None
        if current_digest == self._last_attempted_digest:
            return None

        candidate: Any = None
        try:
            candidate = self.watcher._load_preset(
                strict=True,
                extra_protected_tools=self._protected_tools,
            )
            diff, prompt_changed, memory_changed = self._apply_candidate(candidate)
            self.watcher._commit_preset(
                candidate,
                current_fp,
                close_previous=self.watcher._preset is not self._initial_run_preset,
            )
            candidate = None
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            self._last_attempted_digest = current_digest
            error_text = f"{type(exc).__name__}: {exc}"
            self.last_error = error_text
            obs = ReloadObservation(
                step_num=step_num,
                status="rejected",
                active_digest=self._active_digest,
                candidate_digest=current_digest,
                active_tools=tuple(self._loop_ctx.tools.tool_names),
                error=error_text,
            )
            self.observations.append(obs)
            self._record_on_state(state, obs)
            if self.notify_on_error:
                return InjectContext(obs.format_briefing())
            return None

        self._active_fingerprint = current_fp
        self._active_digest = current_digest
        self._last_attempted_digest = current_digest
        self.last_error = None
        self.reload_count += 1
        obs = ReloadObservation(
            step_num=step_num,
            status="reloaded",
            active_digest=current_digest,
            candidate_digest=current_digest,
            added_tools=diff["added"],
            updated_tools=diff["updated"],
            removed_tools=diff["removed"],
            active_tools=tuple(self._loop_ctx.tools.tool_names),
            prompt_changed=prompt_changed,
            memory_changed=memory_changed,
        )
        self.observations.append(obs)
        self._record_on_state(state, obs)
        if self.notify_on_reload:
            return InjectContext(obs.format_briefing())
        return None

    def checkpoint_state(self) -> dict[str, Any]:
        """Capture active and on-disk workspace/overlay digests for :class:`CheckpointParticipant`.

        Checkpoints are captured after a step's tools run and before the next
        step's ``pre_prompt``. ``disk_digest`` records the post-step files on
        disk (including any newly authored overlay files awaiting the next
        ``pre_prompt`` reload), while ``active_digest`` records the preset
        already applied to the live loop.
        """
        return {
            "schema": self._checkpoint_schema,
            "active_digest": self._active_digest,
            "disk_digest": self.watcher.digest(),
            "last_attempted_digest": self._last_attempted_digest,
            "reload_count": self.reload_count,
        }

    def validate_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Verify saved reload state and ensure on-disk files match ``disk_digest``."""
        expected_keys = {
            "schema",
            "active_digest",
            "disk_digest",
            "last_attempted_digest",
            "reload_count",
        }
        if (
            not isinstance(state, dict)
            or set(state) != expected_keys
            or state.get("schema") != self._checkpoint_schema
            or not isinstance(state.get("active_digest"), str)
            or not state["active_digest"]
            or not isinstance(state.get("disk_digest"), str)
            or not state["disk_digest"]
            or not isinstance(state.get("last_attempted_digest"), str)
            or isinstance(state.get("reload_count"), bool)
            or not isinstance(state.get("reload_count"), int)
            or state["reload_count"] < 0
        ):
            raise ValueError("invalid ReloadHook checkpoint state")
        current_disk_digest = self.watcher.digest()
        if current_disk_digest != state["disk_digest"]:
            raise ValueError(
                f"workspace/overlay files on disk ({current_disk_digest[:16]}) do not match "
                f"checkpointed digest ({state['disk_digest'][:16]})"
            )

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore active capabilities or pending step-boundary reload state."""
        self.validate_checkpoint_state(state)
        self.reload_count = state["reload_count"]
        self.last_error = None
        if state["active_digest"] == state["disk_digest"]:
            current_fp = self.watcher.fingerprint()
            candidate = self.watcher._load_preset(
                strict=True,
                extra_protected_tools=self._protected_tools,
            )
            try:
                if self._loop_ctx is not None:
                    self._apply_candidate(candidate)
                self.watcher._commit_preset(
                    candidate,
                    current_fp,
                    close_previous=self.watcher._preset is not self._initial_run_preset,
                )
                candidate = None
            finally:
                if candidate is not None:
                    candidate.close()
            self._active_fingerprint = current_fp
            self._active_digest = state["active_digest"]
            self._last_attempted_digest = state["last_attempted_digest"]
        else:
            self._active_digest = state["active_digest"]
            self._last_attempted_digest = state["last_attempted_digest"]
