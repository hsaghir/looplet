"""Runnable skill bundles for loading domain bundles.

Bundles are deliberately a thin layer over existing looplet primitives.
A bundle entrypoint returns an :class:`AgentPreset`; the core loop still
only sees tools, hooks, config, and state.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import re
import sys
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from looplet.loop import composable_loop
from looplet.presets import AgentPreset
from looplet.skills import Skill, SkillCard
from looplet.types import LLMBackend, Step

logger = logging.getLogger(__name__)

__all__ = [
    "BundleCard",
    "BundleValidation",
    "SkillBundle",
    "SkillRuntime",
    "discover_skill_bundles",
    "load_skill_bundle",
    "run_skill_bundle",
    "validate_skill_bundle",
]

BuildSkillBundle = Callable[["SkillRuntime"], AgentPreset]


@dataclass(frozen=True)
class BundleCard:
    """Lightweight runnable bundle discovery record."""

    name: str
    description: str
    path: str
    entrypoint: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for CLI output and product UIs."""
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "entrypoint": self.entrypoint,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "ok": self.ok,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class SkillRuntime:
    """Runtime inputs passed to a runnable skill bundle."""

    workspace: str | Path = "."
    max_steps: int = 20
    options: Mapping[str, Any] = field(default_factory=dict)
    output_dir: str | Path | None = None

    def option(self, name: str, default: Any = None) -> Any:
        """Return a bundle-specific option value."""
        return self.options.get(name, default)


@dataclass(frozen=True)
class SkillBundle:
    """A runnable skill bundle loaded from disk."""

    skill: Skill
    root: Path
    build: BuildSkillBundle
    module: ModuleType
    import_roots: tuple[Path, ...] = ()

    @property
    def card(self) -> SkillCard:
        """Return the bundle's lightweight discovery card."""
        return self.skill.card()

    def build_preset(self, runtime: SkillRuntime | None = None) -> AgentPreset:
        """Build the normal looplet preset for this bundle."""
        with self.import_context():
            return self.build(runtime or SkillRuntime())

    def import_context(self) -> AbstractContextManager[None]:
        """Temporarily prioritize this bundle's project-local imports."""
        return _bundle_import_context(self.import_roots)


@dataclass(frozen=True)
class BundleValidation:
    """Result of validating a runnable skill bundle."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skill_name: str | None = None
    preset: AgentPreset | None = None


def discover_skill_bundles(
    *roots: str | Path | Iterable[str | Path],
    include_invalid: bool = False,
    on_duplicate: str = "raise",
) -> list[BundleCard]:
    """Discover runnable skill bundles without importing entrypoint code.

    Roots may be bundle directories, ``SKILL.md`` files, or directories
    containing many skill folders. Instruction-only skills are skipped by
    default because they are not runnable bundles until wrapped.

    Args:
        roots: Paths to scan for bundles.
        include_invalid: Include bundle-like folders with missing or
            invalid metadata. Their ``BundleCard`` will have ``ok=False``
            and a populated ``errors`` list.
        on_duplicate: How to handle two bundles claiming the same
            ``name``. ``"raise"`` (default, back-compat) raises
            ``ValueError`` listing both paths. ``"first_wins"`` keeps
            the first card discovered and silently drops subsequent
            duplicates. ``"warn"`` logs each collision via the standard
            logger and keeps the first card.
    """
    if on_duplicate not in {"raise", "first_wins", "warn"}:
        raise ValueError(
            f"on_duplicate must be 'raise', 'first_wins', or 'warn', got {on_duplicate!r}"
        )
    root_paths = _coerce_discovery_roots(roots)
    cards: dict[str, BundleCard] = {}
    for skill_file in _iter_skill_files(root_paths):
        try:
            card = _bundle_card_for_skill_file(skill_file)
        except Exception as exc:  # noqa: BLE001
            if not include_invalid:
                continue
            name = skill_file.parent.name
            card = BundleCard(
                name=name,
                description="",
                path=str(skill_file.parent.resolve()),
                entrypoint="looplet.py",
                ok=False,
                errors=[f"metadata failed: {type(exc).__name__}: {exc}"],
            )
        if not card.ok and not include_invalid:
            continue
        if card.name in cards:
            first = cards[card.name]
            if on_duplicate == "raise":
                raise ValueError(
                    f"Duplicate bundle name {card.name!r}: {first.path} and {card.path}"
                )
            if on_duplicate == "warn":
                logger.warning(
                    "duplicate bundle name %r — keeping %s, dropping %s",
                    card.name,
                    first.path,
                    card.path,
                )
            # 'first_wins' (and 'warn') both keep the first card.
            continue
        cards[card.name] = card
    return [cards[name] for name in sorted(cards)]


def load_skill_bundle(root: str | Path) -> SkillBundle:
    """Load a runnable skill bundle from a directory or ``SKILL.md`` path.

    The bundle must contain ``SKILL.md`` and an entrypoint file. The
    entrypoint defaults to ``looplet.py`` and must expose
    ``build(runtime: SkillRuntime) -> AgentPreset``.
    """
    skill_file = _skill_file_for(root)
    bundle_root = skill_file.parent.resolve()
    skill = Skill.from_markdown(
        skill_file.read_text(encoding="utf-8"),
        source_path=skill_file,
        default_name=bundle_root.name,
    )
    entrypoint_name = str(skill.metadata.get("entrypoint") or "looplet.py")
    entrypoint = (bundle_root / entrypoint_name).resolve()
    _ensure_inside(entrypoint, bundle_root)
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Skill bundle {skill.name!r} has no entrypoint: {entrypoint}")

    import_roots = tuple(_entrypoint_import_roots(entrypoint))
    module = _load_entrypoint(entrypoint, skill.name, import_roots)
    build = getattr(module, "build", None)
    if not callable(build):
        raise TypeError(
            f"Skill bundle {skill.name!r} entrypoint must define callable build(runtime)"
        )
    return SkillBundle(
        skill=skill,
        root=bundle_root,
        build=cast(BuildSkillBundle, build),
        module=module,
        import_roots=import_roots,
    )


def validate_skill_bundle(
    root: SkillBundle | str | Path,
    runtime: SkillRuntime | None = None,
) -> BundleValidation:
    """Validate that a bundle loads and builds a usable ``AgentPreset``."""
    errors: list[str] = []
    warnings: list[str] = []
    skill_name: str | None = None
    if isinstance(root, SkillBundle):
        bundle = root
        skill_name = bundle.skill.name
    else:
        try:
            bundle = load_skill_bundle(root)
            skill_name = bundle.skill.name
        except Exception as exc:  # noqa: BLE001
            return BundleValidation(
                ok=False,
                errors=[f"load failed: {type(exc).__name__}: {exc}"],
                warnings=warnings,
                skill_name=skill_name,
            )

    try:
        preset = bundle.build_preset(runtime)
    except Exception as exc:  # noqa: BLE001
        return BundleValidation(
            ok=False,
            errors=[f"build failed: {type(exc).__name__}: {exc}"],
            warnings=warnings,
            skill_name=skill_name,
        )

    contract_errors, contract_warnings = _validate_preset_contract(preset, runtime)
    errors.extend(contract_errors)
    warnings.extend(contract_warnings)

    return BundleValidation(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        skill_name=skill_name,
        preset=preset if isinstance(preset, AgentPreset) else None,
    )


def run_skill_bundle(
    bundle: SkillBundle | str | Path,
    *,
    llm: LLMBackend,
    task: str | Mapping[str, Any],
    runtime: SkillRuntime | None = None,
    extra_hooks: Iterable[Any] = (),
    session_log: Any | None = None,
    conversation: Any | None = None,
    provenance: bool = True,
    trace_dir: str | Path | None = None,
    preset: AgentPreset | None = None,
) -> Iterator[Step]:
    """Run a loaded bundle with ``composable_loop``.

    This helper is the console/bundle adapter: it loads a bundle,
    asks the bundle for normal looplet primitives, and delegates to the
    unchanged core loop.
    """
    loaded = load_skill_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    runtime = runtime or SkillRuntime()
    preset = preset or loaded.build_preset(runtime)
    errors, _ = _validate_preset_contract(preset, runtime)
    if errors:
        raise ValueError("invalid bundle preset: " + "; ".join(errors))
    preset = cast(AgentPreset, preset)
    loop_task = {"description": task} if isinstance(task, str) else dict(task)
    hooks = [*preset.hooks, *extra_hooks]
    run_llm: Any = llm
    sink = None
    if provenance:
        from looplet.provenance import ProvenanceSink  # noqa: PLC0415

        sink = ProvenanceSink(
            dir=trace_dir or runtime.output_dir or _default_trace_dir(loaded, runtime)
        )
        run_llm = sink.wrap_llm(llm)
        hooks.append(sink.trajectory_hook())

    def _steps() -> Iterator[Step]:
        try:
            with loaded.import_context():
                yield from composable_loop(
                    llm=run_llm,
                    task=loop_task,
                    tools=preset.tools,
                    state=preset.state,
                    config=preset.config,
                    hooks=hooks,
                    session_log=session_log,
                    conversation=conversation,
                )
        finally:
            if sink is not None:
                sink.flush()

    return _steps()


def _validate_preset_contract(
    preset: Any,
    runtime: SkillRuntime | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(preset, AgentPreset):
        errors.append(f"build returned {type(preset).__name__}, expected AgentPreset")
        return errors, warnings

    names = preset.tools.tool_names
    if len(names) != len(set(names)):
        errors.append("tool names must be unique")
    if "done" not in names:
        errors.append("bundle preset must register a done tool")
    if preset.config.max_steps != preset.state.max_steps:
        warnings.append("config.max_steps and state.max_steps differ")
    if runtime is not None and preset.config.max_steps != runtime.max_steps:
        warnings.append(
            "config.max_steps differs from runtime.max_steps "
            f"({preset.config.max_steps} != {runtime.max_steps})"
        )
    if preset.config.max_steps <= 0:
        errors.append("config.max_steps must be positive")
    return errors, warnings


def _default_trace_dir(bundle: SkillBundle, runtime: SkillRuntime) -> Path:
    workspace = Path(runtime.workspace).resolve()
    return workspace / ".looplet" / "traces" / f"{bundle.skill.name}-{uuid.uuid4().hex[:12]}"


def _skill_file_for(root: str | Path) -> Path:
    path = Path(root)
    if path.is_file():
        if path.name != "SKILL.md":
            raise ValueError(f"Expected SKILL.md, got {path}")
        return path.resolve()
    skill_file = path / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"Skill bundle root has no SKILL.md: {path}")
    return skill_file.resolve()


def _coerce_discovery_roots(
    roots: tuple[str | Path | Iterable[str | Path], ...],
) -> list[Path]:
    if len(roots) == 1 and not isinstance(roots[0], (str, Path)):
        root_iter = list(cast(Iterable[str | Path], roots[0]))
    else:
        root_iter = [cast(str | Path, root) for root in roots]
    if not root_iter:
        raise ValueError("discover_skill_bundles requires at least one root path")
    return [Path(root) for root in root_iter]


def _iter_skill_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "SKILL.md":
            files.add(root.resolve())
            continue
        direct = root / "SKILL.md"
        if direct.is_file():
            files.add(direct.resolve())
        if root.is_dir():
            files.update(path.resolve() for path in root.rglob("SKILL.md"))
    return sorted(files, key=lambda path: str(path))


def _bundle_card_for_skill_file(skill_file: Path) -> BundleCard:
    root = skill_file.parent.resolve()
    skill = Skill.from_markdown(
        skill_file.read_text(encoding="utf-8"),
        source_path=skill_file,
        default_name=root.name,
    )
    entrypoint = str(skill.metadata.get("entrypoint") or "looplet.py")
    errors: list[str] = []
    try:
        entrypoint_path = (root / entrypoint).resolve()
        _ensure_inside(entrypoint_path, root)
    except ValueError as exc:
        entrypoint_path = root / entrypoint
        errors.append(str(exc))
    if not entrypoint_path.is_file():
        errors.append(f"entrypoint not found: {entrypoint}")
    return BundleCard(
        name=skill.name,
        description=skill.description,
        path=str(root),
        entrypoint=entrypoint,
        tags=list(skill.tags),
        metadata=dict(skill.metadata),
        ok=not errors,
        errors=errors,
    )


def _ensure_inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Bundle entrypoint must stay inside bundle root: {path}") from exc


def _entrypoint_import_roots(path: Path) -> list[Path]:
    current = path.parent.resolve()

    project_root: Path | None = None
    for root in (current, *current.parents):
        if root == root.parent:
            break
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            project_root = root
            break

    if project_root is None:
        return [current]

    roots: list[Path] = []
    for root in (current, *current.parents):
        if root == root.parent:
            break
        roots.append(root)
        if root == project_root:
            break
    return roots


_BUNDLE_IMPORT_ROOTS: set[Path] = set()
_PROTECTED_MODULE_PREFIXES = ("_pytest", "looplet", "pytest", "tests")
_PROTECTED_MODULE_NAMES = {"conftest"}
_STDLIB_MODULE_NAMES = frozenset(getattr(sys, "stdlib_module_names", ()))


@contextmanager
def _bundle_import_context(import_roots: Iterable[Path]) -> Iterator[None]:
    roots = tuple(root.resolve() for root in import_roots)
    _preload_shadowed_stdlib_modules(roots)
    removed_modules: dict[str, ModuleType] = {}
    _merge_removed_modules(removed_modules, _purge_conflicting_bundle_modules(roots))
    _merge_removed_modules(removed_modules, _purge_shadowed_project_modules(roots))
    original_modules = dict(sys.modules)
    original_sys_path = list(sys.path)
    try:
        root_strings = [str(root) for root in roots]
        for root in root_strings:
            while root in sys.path:
                sys.path.remove(root)
        sys.path[:0] = root_strings
        _BUNDLE_IMPORT_ROOTS.update(roots)
        yield
    finally:
        _purge_context_modules(roots, original_modules)
        _restore_modules(removed_modules)
        sys.path[:] = original_sys_path


def _purge_conflicting_bundle_modules(current_roots: tuple[Path, ...]) -> dict[str, ModuleType]:
    removed_modules: dict[str, ModuleType] = {}
    for module_name, module in list(sys.modules.items()):
        if _is_protected_module_name(module_name):
            continue
        if not _module_is_inside_any(module, _BUNDLE_IMPORT_ROOTS):
            continue
        if _module_is_inside_any(module, current_roots):
            continue
        _merge_removed_modules(removed_modules, _remove_module(module_name, module))
    return removed_modules


def _is_protected_module_name(module_name: str) -> bool:
    if module_name in _PROTECTED_MODULE_NAMES:
        return True
    if module_name.partition(".")[0] in _STDLIB_MODULE_NAMES:
        return True
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in _PROTECTED_MODULE_PREFIXES
    )


def _purge_context_modules(
    current_roots: tuple[Path, ...],
    original_modules: Mapping[str, ModuleType],
) -> None:
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("_looplet_skill_bundle_"):
            continue
        if original_modules.get(module_name) is module:
            continue
        if _module_is_inside_any(module, current_roots):
            _remove_module(module_name, module)
            continue
        if _is_protected_module_name(module_name):
            continue


def _purge_shadowed_project_modules(current_roots: tuple[Path, ...]) -> dict[str, ModuleType]:
    removed_modules: dict[str, ModuleType] = {}
    for module_name, module in list(sys.modules.items()):
        if _is_protected_module_name(module_name):
            continue
        top_level_name = module_name.partition(".")[0]
        if top_level_name == "looplet":
            continue
        if not _module_exists_in_roots(top_level_name, current_roots):
            continue
        if _module_is_inside_any(module, current_roots):
            continue
        _merge_removed_modules(removed_modules, _remove_module(module_name, module))
    return removed_modules


def _module_exists_in_roots(module_name: str, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        if (root / f"{module_name}.py").is_file():
            return True
        if (root / module_name / "__init__.py").is_file():
            return True
        if (root / module_name).is_dir():
            return True
    return False


def _preload_shadowed_stdlib_modules(roots: tuple[Path, ...]) -> None:
    stdlib_names = _stdlib_module_names_in_roots(roots)
    if not stdlib_names:
        return

    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if not _path_entry_inside_any(entry, roots)]
        for module_name in sorted(stdlib_names):
            module = sys.modules.get(module_name)
            if module is not None and _module_is_inside_any(module, roots):
                _remove_module(module_name, module)
            if module_name not in sys.modules:
                try:
                    importlib.import_module(module_name)
                except ModuleNotFoundError:
                    continue
    finally:
        sys.path[:] = original_sys_path


def _stdlib_module_names_in_roots(roots: tuple[Path, ...]) -> set[str]:
    module_names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_file() and child.suffix == ".py":
                module_name = child.stem
            elif child.is_dir():
                module_name = child.name
            else:
                continue
            if module_name in _STDLIB_MODULE_NAMES:
                module_names.add(module_name)
    return module_names


def _path_entry_inside_any(path_entry: str, roots: tuple[Path, ...]) -> bool:
    try:
        path = Path.cwd().resolve() if path_entry == "" else Path(path_entry).resolve()
    except OSError:
        return False
    return _is_path_inside_any(path, roots)


def _module_file(module: ModuleType) -> Path | None:
    file_name = getattr(module, "__file__", None)
    if not file_name:
        return None
    try:
        return Path(file_name).resolve()
    except OSError:
        return None


def _module_search_locations(module: ModuleType) -> list[Path]:
    locations = getattr(module, "__path__", None)
    if locations is None:
        return []
    resolved: list[Path] = []
    try:
        raw_locations = list(locations)
    except (AttributeError, KeyError):
        return []
    for location in raw_locations:
        try:
            resolved.append(Path(location).resolve())
        except OSError:
            continue
    return resolved


def _module_is_inside_any(module: ModuleType, roots: Iterable[Path]) -> bool:
    roots_tuple = tuple(roots)
    module_path = _module_file(module)
    if module_path is not None and _is_path_inside_any(module_path, roots_tuple):
        return True
    return any(
        _is_path_inside_any(location, roots_tuple) for location in _module_search_locations(module)
    )


def _remove_module(module_name: str, module: ModuleType | None = None) -> dict[str, ModuleType]:
    removed_modules: dict[str, ModuleType] = {}
    for child_name, child_module in list(sys.modules.items()):
        if child_name.startswith(module_name + "."):
            _merge_removed_modules(removed_modules, _remove_module(child_name, child_module))
    removed = sys.modules.pop(module_name, None)
    target = module or removed
    if removed is not None:
        removed_modules[module_name] = removed
    parent_name, _, attribute = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if target is not None and parent is not None and getattr(parent, attribute, None) is target:
        delattr(parent, attribute)
    return removed_modules


def _merge_removed_modules(
    target: dict[str, ModuleType],
    source: Mapping[str, ModuleType],
) -> None:
    for module_name, module in source.items():
        target.setdefault(module_name, module)


def _restore_modules(modules: Mapping[str, ModuleType]) -> None:
    for module_name, module in sorted(modules.items(), key=lambda item: item[0].count(".")):
        sys.modules.setdefault(module_name, module)
    for module_name, module in sorted(modules.items(), key=lambda item: item[0].count(".")):
        parent_name, _, attribute = module_name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and sys.modules.get(module_name) is module:
            setattr(parent, attribute, module)


def _is_path_inside_any(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _load_entrypoint(path: Path, skill_name: str, import_roots: Iterable[Path]) -> ModuleType:
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", skill_name)
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module_name = f"_looplet_skill_bundle_{safe_name}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bundle entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _bundle_import_context(import_roots):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module
