"""Test-only compat shim: aggregates the symbols the legacy bundle
tests expected on ``examples.coder.agent``.

Imports go through the bundle loader (``examples.coder.tools`` /
``examples.coder.wiring``) which the bundle's ``looplet.py`` registers
in ``sys.modules`` on first load. Tests that need this module must
load the bundle once first (e.g. via ``load_skill_bundle(CODER_BUNDLE)``)
so the sibling modules are populated.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_bundle_loaded() -> None:
    if "examples.coder.tools" in sys.modules and "examples.coder.wiring" in sys.modules:
        return
    # Force-load the bundle so its sibling registrations run.
    import importlib.util  # noqa: PLC0415

    looplet_py = Path(__file__).resolve().parent / "looplet.py"
    spec = importlib.util.spec_from_file_location(
        "tests.fixtures.coder_skill_bundle.looplet", looplet_py
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {looplet_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]


_ensure_bundle_loaded()

from examples.coder.hooks import (  # noqa: E402
    FileCacheHook,
    LinterHook,
    StaleFileHook,
    TestGuardHook,
)
from examples.coder.tools import FileCache, make_tools  # noqa: E402
from examples.coder.wiring import SYSTEM_PROMPT, scripted_responses  # noqa: E402

__all__ = [
    "FileCache",
    "FileCacheHook",
    "LinterHook",
    "StaleFileHook",
    "SYSTEM_PROMPT",
    "TestGuardHook",
    "make_tools",
    "scripted_responses",
]
