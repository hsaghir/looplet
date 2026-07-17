"""Hook + tool + resource source loader.

Single function: :func:`_import_module_from_path`. Compiles+execs a
Python file into a fresh ``types.ModuleType``, registered in
``sys.modules`` under a synthetic name, with no ``__pycache__/``
pollution. Used by every cartridge.* loader that materialises Python
sources from disk (tools, hooks, resources, single-file tools,
scaffolded modules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from looplet.cartridge._layout import CartridgeSerializationError


def _import_module_from_path(path: Path, module_name: str) -> Any:
    """Load a workspace-shipped Python module from ``path``.

    Reads the source verbatim and executes it into a fresh
    :class:`types.ModuleType`. We deliberately avoid Python's normal
    :class:`importlib.machinery.SourceFileLoader` for two reasons:

    1. *Hot reload correctness.* ``SourceFileLoader`` writes a
       ``__pycache__/<name>.cpython-XYZ.pyc`` next to the source, and
       on subsequent imports it reuses that bytecode whenever the
       source mtime matches the value stored in the ``.pyc`` header.
       That mtime is recorded with **second** resolution, so two
       writes within the same wall-clock second silently re-use the
       old bytecode - which means a cartridge reloaded after a fast
       edit returns the previous tool body. Reading the source
       directly each time avoids the cache entirely.

    2. *Cartridge cleanliness.* ``SourceFileLoader`` litters the
       cartridge directory with ``__pycache__/`` folders, which makes
       \"the cartridge is just files\" demonstrably untrue - a
       reviewer doing ``ls`` sees engine artefacts mixed with the
       agent's contract.

    The module is registered in :data:`sys.modules` under
    ``module_name`` so :func:`inspect.getmodule` /
    :func:`inspect.getsource` can recover the on-disk source for
    workspace-defined classes (hooks / resources). Linecache uses
    ``__file__`` to fetch source on demand for tracebacks.
    """
    import sys as _sys  # noqa: PLC0415
    import types as _types  # noqa: PLC0415

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CartridgeSerializationError(f"cannot read module {path}: {exc}") from exc

    code = compile(source, str(path), "exec")
    module = _types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__loader__ = None  # type: ignore[assignment]
    _sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        _sys.modules.pop(module_name, None)
        raise
    return module
