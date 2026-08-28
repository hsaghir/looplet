"""Compatibility descriptor for saved provenance and eval-run directories."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

ARTIFACT_DESCRIPTOR = "artifact.json"
ARTIFACT_PENDING = ".artifact.json.pending"
ARTIFACT_SCHEMA = "looplet.saved-artifact"
ARTIFACT_VERSION = 1

_KINDS = {"eval_run", "provenance"}
_COMPONENT_FILES = {
    "artifacts": "artifacts.json",
    "eval_results": "evals.json",
    "model_calls": "manifest.jsonl",
    "trajectory": "trajectory.json",
}
_REQUIRED_COMPONENTS = {
    "eval_run": {"artifacts", "eval_results", "trajectory"},
    "provenance": set(),
}


def begin_artifact_write(directory: str | Path) -> None:
    """Mark an artifact incomplete before removing its previous descriptor."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(root / ARTIFACT_PENDING, "incomplete\n")
    (root / ARTIFACT_DESCRIPTOR).unlink(missing_ok=True)


def write_artifact_descriptor(
    directory: str | Path,
    *,
    kind: str,
    components: Iterable[str],
) -> Path:
    """Atomically publish the descriptor after an artifact payload is complete."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    normalized = _validate_kind_and_components(kind, components)
    _validate_component_files(root, normalized)

    # Import lazily so package initialization cannot cycle through this module.
    from looplet import __version__  # noqa: PLC0415

    payload = {
        "schema": ARTIFACT_SCHEMA,
        "version": ARTIFACT_VERSION,
        "kind": kind,
        "producer": {"name": "looplet", "version": __version__},
        "components": normalized,
    }
    descriptor = root / ARTIFACT_DESCRIPTOR
    _atomic_write(descriptor, json.dumps(payload, indent=2) + "\n")
    (root / ARTIFACT_PENDING).unlink(missing_ok=True)
    return descriptor


def _atomic_write(path: Path, text: str) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_artifact_descriptor(
    directory: str | Path,
    *,
    expected_kinds: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Read and validate a v1 descriptor; return ``None`` for legacy v0."""
    root = Path(directory)
    if (root / ARTIFACT_PENDING).exists():
        raise ValueError(f"Incomplete saved artifact in {root}: write is still pending")
    descriptor = root / ARTIFACT_DESCRIPTOR
    if not descriptor.exists():
        return None
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: expected an object")

    schema = payload.get("schema")
    if schema != ARTIFACT_SCHEMA:
        raise ValueError(f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: unsupported schema {schema!r}")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: version must be an integer")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported {ARTIFACT_SCHEMA} version {version} in {root}; "
            f"this Looplet supports version {ARTIFACT_VERSION}"
        )

    kind = payload.get("kind")
    raw_components = payload.get("components")
    if not isinstance(kind, str):
        raise ValueError(f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: kind must be a string")
    if not isinstance(raw_components, list) or any(
        not isinstance(component, str) for component in raw_components
    ):
        raise ValueError(
            f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: components must be an array of strings"
        )
    components = _validate_kind_and_components(kind, raw_components, root=root)

    producer = payload.get("producer")
    if not isinstance(producer, dict):
        raise ValueError(f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: producer must be an object")
    if producer.get("name") != "looplet" or not isinstance(producer.get("version"), str):
        raise ValueError(
            f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: producer requires looplet name and version"
        )

    if expected_kinds is not None:
        allowed = set(expected_kinds)
        if kind not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ValueError(
                f"Invalid {ARTIFACT_DESCRIPTOR} in {root}: kind {kind!r} is not {expected}"
            )
    _validate_component_files(root, components)
    return payload


def _validate_kind_and_components(
    kind: str,
    components: Iterable[str],
    *,
    root: Path | None = None,
) -> list[str]:
    where = f" in {root}" if root is not None else ""
    if kind not in _KINDS:
        raise ValueError(f"Invalid artifact kind {kind!r}{where}")
    normalized = list(components)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Invalid artifact components{where}: duplicates are not allowed")
    unknown = sorted(set(normalized) - set(_COMPONENT_FILES))
    if unknown:
        raise ValueError(f"Invalid artifact components{where}: unknown {', '.join(unknown)}")
    required = _REQUIRED_COMPONENTS[kind]
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"Invalid {kind} artifact components{where}: missing {', '.join(missing)}")
    if kind == "provenance" and not normalized:
        raise ValueError(f"Invalid provenance artifact components{where}: at least one is required")
    return sorted(normalized)


def _validate_component_files(root: Path, components: Iterable[str]) -> None:
    for component in components:
        path = root / _COMPONENT_FILES[component]
        if not path.is_file():
            raise ValueError(
                f"Incomplete saved artifact in {root}: component {component!r} "
                f"requires {_COMPONENT_FILES[component]}"
            )
        if component == "model_calls":
            _validate_model_call_files(root, path)


def _validate_model_call_files(root: Path, manifest: Path) -> None:
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Invalid model_calls component in {root}: {exc}") from exc
    expected_index = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid model_calls component in {root}: manifest.jsonl line {line_number}: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid model_calls component in {root}: "
                f"manifest.jsonl line {line_number} must be an object"
            )
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
            raise ValueError(
                f"Invalid model_calls component in {root}: manifest.jsonl line "
                f"{line_number} expected index {expected_index}, got {index!r}"
            )
        if entry.get("method", "generate") not in {"generate", "generate_with_tools"}:
            raise ValueError(
                f"Invalid model_calls component in {root}: manifest.jsonl line "
                f"{line_number} has unsupported method {entry.get('method')!r}"
            )
        for suffix in ("prompt", "response"):
            call_path = root / f"call_{index:02d}_{suffix}.txt"
            if not call_path.is_file():
                raise ValueError(
                    f"Incomplete saved artifact in {root}: model_calls index {index} "
                    f"requires {call_path.name}"
                )
        expected_index += 1
