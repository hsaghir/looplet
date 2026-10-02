"""Declaration-only cartridge inspection; no imports, builders or subprocesses."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from looplet.cartridge._layout import CartridgeSerializationError
from looplet.cartridge._load import _shallow_merge_config
from looplet.cartridge._manifest import _read_manifest_data, _read_schema_version
from looplet.cartridge._render import _RUNTIME_PLACEHOLDER, _apply_runtime_substitutions
from looplet.cartridge._yaml import _load_yaml
from looplet.cartridge.spec_slots import compile_model_block


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = _load_yaml(path.read_text(encoding="utf-8"), source_path=path)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CartridgeSerializationError(f"{path} must contain a mapping")
    return value


def _layers(root: Path, seen: frozenset[Path] = frozenset()) -> list[Path]:
    resolved = root.resolve()
    if resolved in seen:
        raise CartridgeSerializationError(f"circular extends detected at {root}")
    if _read_schema_version(root) != 2:
        raise CartridgeSerializationError(f"{root} must declare schema_version=2")
    parent = _mapping(root / "config.yaml").get("extends")
    if parent is None:
        return [root]
    if not isinstance(parent, str) or not parent:
        raise CartridgeSerializationError(f"extends at {root} must be a non-empty path")
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    return [*_layers(parent_path.resolve(), seen | {resolved}), root]


def inspect_cartridge(
    cartridge_dir: str | Path,
    *,
    runtime: dict[str, Any] | None = None,
    include_sensitive: bool = False,
) -> dict[str, Any]:
    """Read declared structure without materializing an executable preset.

    Dynamic factories, references and protocol tool discovery remain explicitly
    unresolved. This is inspection, not executable validation or admission.
    """
    root = Path(cartridge_dir).resolve()
    layers = _layers(root)
    config: dict[str, Any] = {}
    runtime_config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    files: dict[str, Path] = {}
    unresolved: set[str] = set()
    values = {"cartridge_root": str(root), "python_executable": sys.executable, **(runtime or {})}

    def substitute(text: str) -> str:
        def replace(match: Any) -> str:
            expression = match.group(0)
            try:
                replaced = _apply_runtime_substitutions(expression, values)
            except CartridgeSerializationError:
                unresolved.add(expression)
                return expression
            if replaced == expression:
                unresolved.add(expression)
            return replaced

        return _RUNTIME_PLACEHOLDER.sub(replace, text)

    def origins(mapping: dict[str, Any], path: Path, prefix: str = "") -> None:
        for name, value in mapping.items():
            key = f"{prefix}{name}"
            sources[key] = str(path)
            if isinstance(value, dict):
                origins(value, path, key + ".")

    for layer in layers:
        for filename, target in (("config.yaml", config), ("runtime.yaml", runtime_config)):
            path = layer / filename
            if not path.is_file():
                continue
            loaded = (
                _load_yaml(substitute(path.read_text(encoding="utf-8")), source_path=path) or {}
            )
            if not isinstance(loaded, dict):
                raise CartridgeSerializationError(f"{path} must contain a mapping")
            loaded.pop("extends", None)
            merged = _shallow_merge_config(target, loaded)
            target.clear()
            target.update(merged)
            origins(loaded, path)
        for pattern in (
            "prompts/system.md",
            "tools/*/tool.yaml",
            "tools/*.py",
            "hooks/*/config.yaml",
            "hooks/*/hook.py",
            "resources/*.py",
            "memory/*.py",
        ):
            for path in sorted(layer.glob(pattern)):
                if path.is_file():
                    files[path.relative_to(layer).as_posix()] = path

    effective = _shallow_merge_config(config, runtime_config)
    if isinstance(effective.get("model"), dict):
        model_overrides = compile_model_block(effective["model"], existing_cfg=effective)
        effective.update(model_overrides)
        for name in model_overrides:
            sources[name] = sources.get(f"model.{name}", sources.get("model", "model declaration"))
    prompt = files.get("prompts/system.md")
    if prompt is not None:
        effective["system_prompt"] = prompt.read_text(encoding="utf-8")
        sources["system_prompt"] = str(prompt)
    tools: list[dict[str, Any]] = []
    hooks: list[dict[str, Any]] = []
    resources: list[str] = []
    for relative, path in sorted(files.items()):
        parts = Path(relative).parts
        if parts[0] == "tools" and path.name == "tool.yaml":
            definition = _mapping(path)
            tools.append(
                {
                    **definition,
                    "name": definition.get("name", parts[1]),
                    "source": str(path),
                    "kind": "declared",
                }
            )
        elif parts[0] == "tools" and len(parts) == 2:
            tools.append({"name": path.stem, "source": str(path), "kind": "python-factory"})
            unresolved.add(relative)
        elif parts[0] == "hooks" and path.name == "config.yaml":
            definition = _mapping(path)
            hooks.append(
                {
                    "name": parts[1],
                    "source": str(path),
                    "kind": definition.get("kind", "python"),
                    "class_name": definition.get("class_name"),
                }
            )
            unresolved.add(f"hook:{parts[1]}")
        elif parts[0] == "hooks" and path.name == "hook.py":
            unresolved.add(relative)
        elif parts[0] == "resources":
            resources.append(path.stem)
            unresolved.add(relative)
        elif parts[0] == "memory":
            unresolved.add(relative)

    def declared_names(field_name: str) -> list[str]:
        entries = effective.get(field_name, []) or []
        if not isinstance(entries, list):
            raise CartridgeSerializationError(f"{field_name} must be a list")
        names = []
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict) and len(entry) == 1 and isinstance(next(iter(entry)), str):
                names.append(next(iter(entry)))
            else:
                raise CartridgeSerializationError(
                    f"{field_name} entries must be names or single-key mappings"
                )
        return names

    for name in declared_names("builtin_tools"):
        tools.append({"name": name, "kind": "builtin"})
    for field_name in ("mcp_servers", "state_services"):
        providers = effective.get(field_name, {}) or {}
        if not isinstance(providers, dict) or not all(isinstance(name, str) for name in providers):
            raise CartridgeSerializationError(f"{field_name} must be a name-keyed mapping")
        for name in providers:
            if field_name == "mcp_servers":
                tools.append({"name": f"mcp:{name}", "kind": "protocol-discovery"})
                unresolved.add(f"mcp:{name}")
            else:
                resources.append(name)
                unresolved.add(f"state-service:{name}")
    for name in declared_names("builtin_hooks"):
        hooks.append({"name": name, "kind": "builtin"})

    def references(value: Any) -> None:
        if isinstance(value, str) and value.startswith(("${ref:", "${py:", "${runtime:", "@")):
            unresolved.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)

    if not include_sensitive:
        for name in (
            "system_prompt",
            "tool_metadata",
            "generate_kwargs",
            "memory",
            "memory_sources",
            "mcp_servers",
            "lep_servers",
            "state_services",
            "services",
        ):
            if name in effective:
                effective[name] = "[redacted]"
        model = effective.get("model")
        if isinstance(model, dict) and "extra" in model:
            model["extra"] = "[redacted]"
    references(effective)
    return {
        "mode": "declaration-only",
        "runtime_validated": False,
        "root": str(root),
        "manifest": _read_manifest_data(root),
        "config": effective,
        "configuration_sources": sources,
        "tools": tools,
        "hooks": hooks,
        "resources": sorted(set(resources)),
        "runtime_required": sorted(unresolved),
    }
