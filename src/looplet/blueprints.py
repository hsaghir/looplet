"""Blueprints and conversion helpers for looplet bundles.

The blueprint layer is a small, serialisable description of an agent's
observable looplet structure. It is not a Python decompiler; it records
stable structure so bundles can be inspected, compared, exported as
library wrappers, and packaged back into runnable bundles.
"""

from __future__ import annotations

import json
import keyword
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from looplet.bundles import SkillBundle, SkillRuntime, load_skill_bundle
from looplet.presets import AgentPreset
from looplet.skills import Skill
from looplet.tools import ToolSpec

__all__ = [
    "AgentBlueprint",
    "BlueprintComparison",
    "ClaudeSkillCompatibility",
    "ComponentBlueprint",
    "SourceBlueprint",
    "ToolBlueprint",
    "blueprint_from_bundle",
    "blueprint_from_preset",
    "claude_skill_compatibility",
    "compare_blueprints",
    "export_bundle_to_library_code",
    "package_agent_factory_as_bundle",
    "wrap_claude_skill_as_bundle",
]

BLUEPRINT_SCHEMA_VERSION = "looplet.agent-blueprint.v1"


@dataclass(frozen=True)
class SourceBlueprint:
    """Where a blueprint came from."""

    kind: str
    path: str | None = None
    entrypoint: str | None = None
    factory_ref: str | None = None


@dataclass(frozen=True)
class ToolBlueprint:
    """Stable, non-executable description of one tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: str | None = None
    concurrent_safe: bool = False
    free: bool = False
    timeout_s: float | None = None


@dataclass(frozen=True)
class ComponentBlueprint:
    """Stable description of a hook, memory source, or other component."""

    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentBlueprint:
    """Versioned, serialisable recipe view of a looplet agent."""

    name: str
    description: str = ""
    schema_version: str = BLUEPRINT_SCHEMA_VERSION
    tags: list[str] = field(default_factory=list)
    source: SourceBlueprint = field(default_factory=lambda: SourceBlueprint(kind="unknown"))
    instructions: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolBlueprint] = field(default_factory=list)
    hooks: list[ComponentBlueprint] = field(default_factory=list)
    memory_sources: list[ComponentBlueprint] = field(default_factory=list)
    state: ComponentBlueprint | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return cast(dict[str, Any], asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentBlueprint":
        """Build a blueprint from :meth:`to_dict` data."""
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            schema_version=str(data.get("schema_version", BLUEPRINT_SCHEMA_VERSION)),
            tags=[str(tag) for tag in data.get("tags", [])],
            source=SourceBlueprint(**dict(data.get("source", {"kind": "unknown"}))),
            instructions=str(data.get("instructions", "")),
            config=dict(data.get("config", {})),
            tools=[ToolBlueprint(**dict(item)) for item in data.get("tools", [])],
            hooks=[ComponentBlueprint(**dict(item)) for item in data.get("hooks", [])],
            memory_sources=[
                ComponentBlueprint(**dict(item)) for item in data.get("memory_sources", [])
            ],
            state=(
                ComponentBlueprint(**dict(data["state"])) if data.get("state") is not None else None
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def fingerprint(self, *, include_metadata: bool = True) -> str:
        """Return a stable hash of the blueprint structure."""
        payload = _comparison_payload(self, include_metadata=include_metadata)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BlueprintComparison:
    """Result of comparing two blueprints."""

    ok: bool
    differences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClaudeSkillCompatibility:
    """Compatibility report for a Claude/Agent Skills-style folder."""

    level: str
    can_wrap: bool
    can_run_exactly: bool
    warnings: list[str] = field(default_factory=list)
    skill_name: str | None = None


def blueprint_from_bundle(
    bundle_root: SkillBundle | str | Path,
    runtime: SkillRuntime | None = None,
) -> AgentBlueprint:
    """Load a bundle and return its structural blueprint."""
    bundle = bundle_root if isinstance(bundle_root, SkillBundle) else load_skill_bundle(bundle_root)
    preset = bundle.build_preset(runtime or SkillRuntime())
    return blueprint_from_preset(
        preset,
        name=bundle.skill.name,
        description=bundle.skill.description,
        tags=bundle.skill.tags,
        instructions=bundle.skill.instructions,
        metadata=bundle.skill.metadata,
        source=SourceBlueprint(
            kind="bundle",
            path=str(bundle.root),
            entrypoint=str(bundle.skill.metadata.get("entrypoint") or "looplet.py"),
        ),
    )


def blueprint_from_preset(
    preset: AgentPreset,
    *,
    name: str,
    description: str = "",
    tags: Iterable[str] = (),
    instructions: str = "",
    metadata: Mapping[str, Any] | None = None,
    source: SourceBlueprint | None = None,
) -> AgentBlueprint:
    """Return a structural blueprint for an instantiated :class:`AgentPreset`."""
    tools = [_tool_blueprint(spec) for spec in _iter_tool_specs(preset)]
    return AgentBlueprint(
        name=name,
        description=description,
        tags=[str(tag) for tag in tags],
        source=source or SourceBlueprint(kind="preset"),
        instructions=instructions,
        config=_config_blueprint(preset),
        tools=tools,
        hooks=[_component_blueprint(hook) for hook in preset.hooks],
        memory_sources=[_component_blueprint(source) for source in preset.config.memory_sources],
        state=_component_blueprint(preset.state),
        metadata=_jsonable_mapping(metadata or {}),
    )


def compare_blueprints(
    left: AgentBlueprint,
    right: AgentBlueprint,
    *,
    ignore_metadata: bool = False,
) -> BlueprintComparison:
    """Compare two blueprints and return human-readable differences."""
    left_payload = _comparison_payload(left, include_metadata=not ignore_metadata)
    right_payload = _comparison_payload(right, include_metadata=not ignore_metadata)
    differences: list[str] = []
    for key in sorted(set(left_payload) | set(right_payload)):
        if left_payload.get(key) != right_payload.get(key):
            differences.append(f"{key} differs")
    return BlueprintComparison(ok=not differences, differences=differences)


def export_bundle_to_library_code(
    bundle_root: str | Path,
    out_file: str | Path,
    *,
    function_name: str = "build",
) -> Path:
    """Export a bundle as exact, editable Python library wrapper code.

    This mode preserves behavior by delegating to the source bundle. It is
    the reliable conversion path for arbitrary bundles, including those
    with closures or product-owned runtime shells.
    """
    if not function_name.isidentifier() or keyword.iskeyword(function_name):
        raise ValueError(f"function_name must be a valid Python function name: {function_name!r}")
    bundle = load_skill_bundle(bundle_root)
    blueprint = blueprint_from_bundle(bundle)
    target = Path(out_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    blueprint_json = json.dumps(blueprint.to_dict(), indent=2, sort_keys=True)
    target.write_text(
        f'''"""Generated local looplet library wrapper for the {bundle.skill.name} bundle.

This preserves exact behavior by loading the original bundle from the
absolute path recorded below. Keep that bundle available, or re-export
after moving it.
"""

from __future__ import annotations

import json
from pathlib import Path

from looplet import SkillRuntime, load_skill_bundle
from looplet.blueprints import AgentBlueprint

_BUNDLE_PATH = Path({str(bundle.root)!r})
_BLUEPRINT_JSON = r"""{blueprint_json}"""
BLUEPRINT = AgentBlueprint.from_dict(json.loads(_BLUEPRINT_JSON))


def {function_name}(runtime: SkillRuntime | None = None):
    """Build the same AgentPreset as the source bundle."""
    return load_skill_bundle(_BUNDLE_PATH).build_preset(runtime or SkillRuntime())
''',
        encoding="utf-8",
    )
    return target


def package_agent_factory_as_bundle(
    factory_ref: str,
    out_dir: str | Path,
    *,
    name: str,
    description: str,
    tags: Iterable[str] = (),
    instructions: str = "",
    entrypoint: str = "looplet.py",
) -> Path:
    """Package an importable looplet factory as a runnable bundle."""
    if ":" not in factory_ref:
        raise ValueError("factory_ref must use 'module:attribute' syntax")
    _resolve_factory(factory_ref)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    entrypoint_path = _entrypoint_path(root, entrypoint)
    skill_text = _render_skill_markdown(
        name=name,
        description=description,
        tags=list(tags),
        entrypoint=entrypoint,
        metadata={"factory": factory_ref},
        instructions=instructions,
    )
    (root / "SKILL.md").write_text(skill_text, encoding="utf-8")
    entrypoint_path.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_path.write_text(
        f'''"""Generated looplet bundle wrapper for {factory_ref}."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_FACTORY_REF = {factory_ref!r}


def _factory() -> Any:
    module_name, attr_name = _FACTORY_REF.split(":", 1)
    return getattr(import_module(module_name), attr_name)


def build(runtime: Any):
    """Build the packaged AgentPreset."""
    return _factory()(runtime)
''',
        encoding="utf-8",
    )
    return root


def claude_skill_compatibility(skill_root: str | Path) -> ClaudeSkillCompatibility:
    """Report how a Claude/Agent Skills-style folder can run in looplet."""
    skill_file = _skill_file_for(skill_root)
    skill = Skill.from_markdown(
        skill_file.read_text(encoding="utf-8"),
        source_path=skill_file,
        default_name=skill_file.parent.name,
    )
    if skill.metadata.get("entrypoint"):
        return ClaudeSkillCompatibility(
            level="looplet-bundle",
            can_wrap=True,
            can_run_exactly=True,
            skill_name=skill.name,
        )

    files = [path for path in skill_file.parent.rglob("*") if path.is_file()]
    payload_files = [path for path in files if path != skill_file]
    script_files = [path for path in payload_files if _looks_like_script(path, skill_file.parent)]
    if script_files:
        return ClaudeSkillCompatibility(
            level="scripts-present",
            can_wrap=True,
            can_run_exactly=False,
            warnings=["scripts require an explicit looplet tool adapter"],
            skill_name=skill.name,
        )
    if payload_files:
        return ClaudeSkillCompatibility(
            level="resources-present",
            can_wrap=True,
            can_run_exactly=False,
            warnings=["resources are copied but remain inert unless exposed as tools"],
            skill_name=skill.name,
        )
    return ClaudeSkillCompatibility(
        level="instruction-only",
        can_wrap=True,
        can_run_exactly=True,
        skill_name=skill.name,
    )


def wrap_claude_skill_as_bundle(skill_root: str | Path, out_dir: str | Path) -> Path:
    """Wrap a Claude Skill folder as an instruction-only looplet bundle."""
    skill_file = _skill_file_for(skill_root)
    source_root = skill_file.parent
    target = Path(out_dir)
    source_resolved = source_root.resolve()
    target_resolved = target.resolve()
    if source_resolved == target_resolved:
        raise ValueError("out_dir must be different from the source skill directory")
    if target_resolved.is_relative_to(source_resolved) or source_resolved.is_relative_to(
        target_resolved
    ):
        raise ValueError(
            "out_dir must be outside the source skill directory and must not contain it"
        )
    if target.exists():
        raise ValueError("out_dir already exists; choose a new directory")
    report = claude_skill_compatibility(source_root)
    if report.level == "looplet-bundle":
        temp_target = _copytree_temp_directory(source_root, target)
        try:
            temp_target.rename(target)
        except Exception:
            if temp_target.exists():
                shutil.rmtree(temp_target)
            raise
        return target
    skill = Skill.from_markdown(
        skill_file.read_text(encoding="utf-8"),
        source_path=skill_file,
        default_name=source_root.name,
    )
    temp_target = _copytree_temp_directory(source_root, target)
    try:
        (temp_target / "SKILL.md").write_text(
            _render_skill_markdown(
                name=skill.name,
                description=skill.description,
                tags=skill.tags,
                entrypoint="looplet.py",
                metadata={
                    "source_format": "claude-skill",
                    "compatibility": report.level,
                },
                instructions=skill.instructions,
            ),
            encoding="utf-8",
        )
        (temp_target / "looplet.py").write_text(
            '''"""Generated looplet wrapper for an instruction-style Claude Skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from looplet import Skill, minimal_preset

_SKILL_FILE = Path(__file__).with_name("SKILL.md")


def build(runtime: Any):
    """Build a minimal runnable agent using the skill instructions."""
    skill = Skill.from_markdown(
        _SKILL_FILE.read_text(encoding="utf-8"),
        source_path=_SKILL_FILE,
        default_name=_SKILL_FILE.parent.name,
    )
    prompt = "\\n\\n".join(
        part
        for part in (
            f"You are using the {skill.name} skill.",
            skill.instructions,
        )
        if part
    )
    return minimal_preset(max_steps=runtime.max_steps, system_prompt=prompt)
''',
            encoding="utf-8",
        )
        temp_target.rename(target)
    except Exception:
        if temp_target.exists():
            shutil.rmtree(temp_target)
        raise
    return target


def _iter_tool_specs(preset: AgentPreset) -> list[ToolSpec]:
    return [preset.tools._tools[name] for name in preset.tools.tool_names]


def _tool_blueprint(spec: ToolSpec) -> ToolBlueprint:
    return ToolBlueprint(
        name=spec.name,
        description=spec.description,
        parameters=_jsonable_mapping(spec.to_json_schema()),
        handler=_callable_ref(spec.execute),
        concurrent_safe=spec.concurrent_safe,
        free=spec.free,
        timeout_s=spec.timeout_s,
    )


def _component_blueprint(component: Any) -> ComponentBlueprint:
    return ComponentBlueprint(kind=_type_ref(component), config=_object_config(component))


def _config_blueprint(preset: AgentPreset) -> dict[str, Any]:
    config = preset.config
    data: dict[str, Any] = {}
    for name in (
        "max_steps",
        "max_tokens",
        "system_prompt",
        "temperature",
        "recovery_temperature",
        "done_tool",
        "max_turn_continuations",
        "use_native_tools",
        "concurrent_dispatch",
        "reactive_recovery",
        "acceptance_criteria",
        "max_briefing_tokens",
        "checkpoint_dir",
        "context_window",
        "tool_metadata",
        "generate_kwargs",
    ):
        data[name] = _jsonable(getattr(config, name))
    for name in (
        "build_briefing",
        "extract_entities",
        "build_trace",
        "build_prompt",
        "extract_step_metadata",
        "approval_handler",
        "render_messages_override",
    ):
        value = getattr(config, name)
        if value is not None:
            data[name] = _callable_ref(value)
    for name in (
        "domain",
        "router",
        "tracer",
        "recovery_registry",
        "compact_service",
        "output_schema",
        "initial_checkpoint",
        "cache_policy",
        "cancel_token",
    ):
        value = getattr(config, name)
        if value is not None:
            data[name] = _component_blueprint(value).kind
    return data


def _comparison_payload(blueprint: AgentBlueprint, *, include_metadata: bool) -> dict[str, Any]:
    data = blueprint.to_dict()
    if not include_metadata:
        for key in ("name", "description", "tags", "source", "instructions", "metadata"):
            data.pop(key, None)
    return data


def _object_config(component: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name, value in vars(component).items() if hasattr(component, "__dict__") else []:
        jsonable = _jsonable(value)
        if jsonable is not None:
            config[name] = jsonable
    return config


def _jsonable_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in data.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return None


def _callable_ref(fn: Any) -> str | None:
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module and qualname and "<locals>" not in qualname:
        return f"{module}:{qualname}"
    return None


def _resolve_factory(factory_ref: str) -> Any:
    module_name, attr_name = factory_ref.split(":", 1)
    try:
        factory = getattr(import_module(module_name), attr_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not import factory {factory_ref!r}: {exc}") from exc
    if not callable(factory):
        raise ValueError(f"factory {factory_ref!r} is not callable")
    return factory


def _type_ref(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def _skill_file_for(root: str | Path) -> Path:
    path = Path(root)
    skill_file = path if path.is_file() and path.name == "SKILL.md" else path / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"Claude Skill folder has no SKILL.md: {path}")
    return skill_file


def _entrypoint_path(root: Path, entrypoint: str) -> Path:
    target = (root / entrypoint).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("entrypoint must stay inside bundle directory")
    return target


def _copytree_temp_directory(source: Path, target: Path) -> Path:
    if target.parent.exists() and not target.parent.is_dir():
        raise ValueError("out_dir parent must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temp_target)
    except Exception:
        if temp_target.exists():
            shutil.rmtree(temp_target)
        raise
    return temp_target


def _looks_like_script(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] == "scripts":
        return True
    return path.suffix in {".py", ".sh", ".js", ".ts", ".rb", ".pl"}


def _render_skill_markdown(
    *,
    name: str,
    description: str,
    tags: list[str] | None = None,
    entrypoint: str,
    metadata: Mapping[str, Any] | None = None,
    instructions: str = "",
) -> str:
    lines = ["---", f"name: {name}", f"description: {description}", f"entrypoint: {entrypoint}"]
    if tags:
        lines.append("tags: [" + ", ".join(tags) + "]")
    for key, value in (metadata or {}).items():
        lines.append(f"{key}: {_frontmatter_value(value)}")
    lines.append("---")
    body = instructions.strip()
    return "\n".join(lines) + "\n\n" + (body + "\n" if body else "")


def _frontmatter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)
