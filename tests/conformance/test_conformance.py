"""Conformance fixtures for cartridge-spec v1.0.

Each fixture under :file:`tests/conformance/fixtures/` is a minimal
cartridge paired with an ``expected.json`` describing the loader
output any v1.0 conformant runtime must produce. The list grows over
time; v2 will mandate this as a release criterion. For now it is the
seed.

A fixture's ``expected.json`` is a small JSON document describing the
parts of the loader output the spec actually pins down. Implementation
details (live Python objects, hook ordering beyond what the cartridge
declares, etc.) are out of scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from looplet.cartridge import CartridgeSerializationError, cartridge_to_preset
from looplet.permissions import PermissionDecision, PermissionHook

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _list_fixtures() -> list[Path]:
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())


def _summarise_preset(preset: Any) -> dict[str, Any]:
    """Reduce a loaded preset to the spec-pinned subset.

    Adding a field to this summary widens the conformance contract.
    Don't do it without updating SPEC.md.
    """
    cfg = preset.config
    tools_summary = sorted(
        (
            {
                "name": name,
                "requires": list(getattr(preset.tools._tools[name], "requires", []) or []),  # type: ignore[attr-defined]
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


@pytest.mark.parametrize("fixture_dir", _list_fixtures(), ids=lambda p: p.name)
def test_conformance_fixture(fixture_dir: Path) -> None:
    """Load the fixture and compare the spec-pinned subset of the preset."""
    expected_path = fixture_dir / "expected.json"
    expected_error_path = fixture_dir / "expected_error.json"
    if not expected_path.is_file() and not expected_error_path.is_file():
        pytest.skip(f"fixture {fixture_dir.name} has no expected.json")
    cartridge = fixture_dir / "cartridge"
    assert cartridge.is_dir(), f"fixture {fixture_dir.name} missing cartridge/"
    runtime_kwargs: dict[str, Any] = {
        "ask_handler": lambda _call, _rule: PermissionDecision.DENY,
    }

    # Rejection fixtures: assert the loader raises the named error class
    # with a message that names the offending file. The spec pins the
    # error class + message substring; everything else is implementation
    # detail.
    if expected_error_path.is_file():
        spec = json.loads(expected_error_path.read_text())
        error_class_name = spec["error_class"]
        message_substr = spec["message_contains"]
        # Currently the only spec-pinned rejection class is
        # CartridgeSerializationError; keep the lookup simple.
        assert error_class_name == "CartridgeSerializationError", (
            f"fixture {fixture_dir.name}: unsupported error_class {error_class_name!r}"
        )
        with pytest.raises(CartridgeSerializationError) as excinfo:
            cartridge_to_preset(str(cartridge), strict=True, runtime=runtime_kwargs)
        assert message_substr in str(excinfo.value), (
            f"fixture {fixture_dir.name}: error message did not contain "
            f"{message_substr!r}.\n  actual: {excinfo.value}"
        )
        return

    expected = json.loads(expected_path.read_text())
    # Cartridge spec v2: any fixture that declares ``ask:`` rules must
    # supply an ``ask_handler`` to the loader, otherwise loading
    # fail-louds. Fixtures don't carry executable Python, so wire in a
    # deterministic stub here that always denies - conformance is
    # about loader behaviour, not runtime decisions.
    preset = cartridge_to_preset(str(cartridge), strict=True, runtime=runtime_kwargs)
    summary = _summarise_preset(preset)
    assert summary == expected, (
        f"conformance fixture {fixture_dir.name!r} mismatch.\n"
        f"  expected: {json.dumps(expected, indent=2)}\n"
        f"  actual:   {json.dumps(summary, indent=2)}"
    )
