"""Lock the loader-independence contract for ``looplet.cartridge``.

The cartridge package is intended to be extractable into a standalone
``looplet-cartridge`` distribution. To preserve that property, only a
narrow allowlist of looplet modules may be imported at the top level
of cartridge/**.py; everything else must be lazy (function-local).

If you intentionally need a new top-level import, update the allowlist
in BOTH this test AND the docstring at the top of
``src/looplet/cartridge/__init__.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

CARTRIDGE_DIR = Path(__file__).resolve().parents[1] / "src" / "looplet" / "cartridge"

# Top-level looplet.* modules cartridge may import at file scope.
# Everything outside this set must be lazy (imported inside function
# bodies / class methods / TYPE_CHECKING guards).
ALLOWED_TOPLEVEL_LOOPLET_IMPORTS = frozenset(
    {
        "looplet.refs",
        "looplet.hook_decision",
        "looplet.permissions",
        "looplet.validation",
    }
)


def _collect_toplevel_looplet_imports(path: Path) -> set[str]:
    """Return the looplet.* modules imported at module scope by ``path``.

    Imports inside function bodies, class bodies, or ``TYPE_CHECKING``
    blocks are deliberately excluded - they don't count against
    extractability since they don't fire at module load time.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: set[str] = set()
    for node in tree.body:  # ← only top-level statements
        if isinstance(node, ast.If):
            # ``if TYPE_CHECKING:`` blocks are type-only; skip.
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            if (
                isinstance(test, ast.Attribute)
                and isinstance(test.value, ast.Name)
                and test.attr == "TYPE_CHECKING"
            ):
                continue
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("looplet."):
                hits.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("looplet."):
                    hits.add(alias.name)
    return hits


def test_cartridge_toplevel_imports_match_allowlist() -> None:
    """No cartridge/**.py imports a non-cartridge looplet module at file scope
    outside the allowlist."""
    violations: dict[str, set[str]] = {}
    for path in sorted(CARTRIDGE_DIR.rglob("*.py")):
        rel = path.relative_to(CARTRIDGE_DIR.parent.parent)  # → looplet/cartridge/...
        toplevel = _collect_toplevel_looplet_imports(path)
        # Imports of sibling cartridge modules are fine (intra-package).
        external = {m for m in toplevel if not m.startswith("looplet.cartridge")}
        bad = external - ALLOWED_TOPLEVEL_LOOPLET_IMPORTS
        if bad:
            violations[str(rel)] = bad
    assert not violations, (
        "looplet.cartridge top-level import allowlist violated. "
        "Either move the import inside a function body / TYPE_CHECKING "
        "guard, or update ALLOWED_TOPLEVEL_LOOPLET_IMPORTS in this test "
        "AND the docstring at the top of "
        "src/looplet/cartridge/__init__.py.\n\n"
        f"Violations: {violations}"
    )
