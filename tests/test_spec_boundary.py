"""Enforce the cartridge-spec / runtime boundary inside looplet.

The spec modules listed in :mod:`looplet._spec_modules` constitute
the surface that ``SPEC.md`` describes and that a future spec-only
package would carve out. They MUST NOT take top-level imports on
the looplet runtime (the loop, backends, presets, examples, CLI,
etc.). Function-local imports and ``TYPE_CHECKING`` imports are
fine; the check inspects the module AST.

If you need to use a runtime module from a spec module, import it
inside the function or guard it with ``if TYPE_CHECKING:``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from looplet._spec_modules import SPEC_FORBIDDEN_TOP_LEVEL_IMPORTS, SPEC_MODULES

PKG_ROOT = Path(__file__).resolve().parent.parent / "src" / "looplet"


def _iter_module_files() -> list[tuple[str, Path]]:
    """Return ``(short_name, path)`` for each spec module on disk.

    A spec module is either ``<name>.py`` directly under
    ``src/looplet/`` or a package directory ``<name>/`` with an
    ``__init__.py``.
    """
    found: list[tuple[str, Path]] = []
    for name in sorted(SPEC_MODULES):
        py = PKG_ROOT / f"{name}.py"
        pkg_init = PKG_ROOT / name / "__init__.py"
        if py.is_file():
            found.append((name, py))
        elif pkg_init.is_file():
            found.append((name, pkg_init))
        else:
            pytest.fail(
                f"_spec_modules lists {name!r} but no module is on disk at {py} or {pkg_init}"
            )
    return found


def _top_level_looplet_imports(source: str) -> set[str]:
    """Return the short module names that ``source`` imports at module top level.

    Function-local imports and ``TYPE_CHECKING`` imports are excluded.
    The check is conservative: it walks ``ast.Module.body`` directly,
    so anything wrapped in ``if TYPE_CHECKING:``, ``def``, ``class``,
    or ``try/except`` falls outside its scope.
    """
    names: set[str] = set()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "looplet" or module.startswith("looplet."):
                short = module.split(".", 2)[1] if module.startswith("looplet.") else ""
                if short:
                    names.add(short)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "looplet" or alias.name.startswith("looplet."):
                    short = alias.name.split(".", 2)[1] if "." in alias.name else ""
                    if short:
                        names.add(short)
    return names


@pytest.mark.parametrize("module_name,module_path", _iter_module_files())
def test_spec_module_has_no_runtime_top_level_imports(module_name: str, module_path: Path) -> None:
    source = module_path.read_text(encoding="utf-8")
    imported = _top_level_looplet_imports(source)
    forbidden = imported & SPEC_FORBIDDEN_TOP_LEVEL_IMPORTS
    assert not forbidden, (
        f"spec module {module_name!r} top-level-imports runtime modules {sorted(forbidden)}; "
        f"move these into a function body or a ``TYPE_CHECKING`` block. "
        f"See looplet._spec_modules for the boundary definition."
    )


def test_spec_module_files_exist() -> None:
    """Every spec module name resolves to an actual file."""
    found = _iter_module_files()
    assert {name for name, _ in found} == set(SPEC_MODULES)


def test_no_runtime_module_appears_in_spec_modules() -> None:
    """Spec and runtime sets must be disjoint."""
    assert not (SPEC_MODULES & SPEC_FORBIDDEN_TOP_LEVEL_IMPORTS), (
        "SPEC_MODULES and SPEC_FORBIDDEN_TOP_LEVEL_IMPORTS overlap; "
        "this means a module is being declared as both spec and runtime, "
        "which is contradictory."
    )
