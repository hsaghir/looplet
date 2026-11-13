"""Load cartridge Python source without bytecode caches.

Direct tool, hook, and resource loading uses :func:`_import_module_from_path`.
Ordinary helper imports use source-only loading for Python's standard loader.
Custom host loaders retain their behavior.
"""

from __future__ import annotations

import builtins
import sys
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from importlib.util import module_from_spec
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any, Iterator
from uuid import uuid4

from looplet.cartridge._layout import CartridgeSerializationError


class _CartridgeSourceLoader(SourceFileLoader):
    def get_code(self, fullname: str) -> CodeType:
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _CartridgeSourceFinder:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None or type(spec.loader) is not SourceFileLoader:
            return None
        if not Path(spec.origin).resolve().is_relative_to(self.root):
            return None
        spec.loader = _CartridgeSourceLoader(fullname, spec.origin)
        return spec


class _CartridgeNamespace:
    """Per-cartridge isolated module namespace without global sys.path/sys.modules pollution."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.uid = uuid4().hex[:12]
        res_dir = self.root / "resources"
        self.search_paths: list[str] = [str(self.root)] + (
            [str(res_dir)] if res_dir.is_dir() else []
        )
        self.modules: dict[str, ModuleType] = {}
        self._lock = threading.RLock()
        self.builtins: dict[str, Any] = dict(builtins.__dict__)
        self.builtins["__import__"] = self._import_hook

    def _host_meta_path_claims(self, top_name: str) -> bool:
        for finder in sys.meta_path:
            if finder is PathFinder:
                break
            find_spec = getattr(finder, "find_spec", None)
            if callable(find_spec):
                try:
                    if find_spec(top_name, None) is not None:
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def _find_local_spec(self, fullname: str, search_locations: Sequence[str]) -> ModuleSpec | None:
        spec = PathFinder.find_spec(fullname, search_locations)
        if spec is None or spec.origin is None:
            return None
        try:
            if not Path(spec.origin).resolve().is_relative_to(self.root):
                return None
        except OSError:
            return None
        if type(spec.loader) is SourceFileLoader:
            spec.loader = _CartridgeSourceLoader(fullname, spec.origin)
        return spec

    def _load_one(self, fullname: str, search_locations: Sequence[str]) -> ModuleType | None:
        with self._lock:
            existing = self.modules.get(fullname)
            if existing is not None:
                return existing
            spec = self._find_local_spec(fullname, search_locations)
            if spec is None or spec.loader is None:
                return None
            mod = module_from_spec(spec)
            mod.__dict__["__builtins__"] = self.builtins
            self.modules[fullname] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                self.modules.pop(fullname, None)
                raise
            origin_str = str(spec.origin) if spec.origin is not None else None
            if origin_str is not None:
                setattr(mod, "__looplet_source_path__", origin_str)
            for val in mod.__dict__.values():
                if getattr(val, "__module__", None) != fullname:
                    continue
                try:
                    val.__looplet_helper_module__ = fullname
                    if origin_str is not None:
                        val.__looplet_source_path__ = origin_str
                except (AttributeError, TypeError):
                    pass
            return mod

    def _load_chain(self, dotted_name: str) -> ModuleType | None:
        parts = [p for p in dotted_name.split(".") if p]
        if not parts:
            return None
        current_search: Sequence[str] = self.search_paths
        prefix = ""
        last_mod: ModuleType | None = None
        for idx, part in enumerate(parts):
            prefix = f"{prefix}.{part}" if prefix else part
            mod = self._load_one(prefix, current_search)
            if mod is None:
                if idx == 0:
                    return None
                raise ModuleNotFoundError(f"No module named {prefix!r}", name=prefix)
            if last_mod is not None:
                setattr(last_mod, part, mod)
            last_mod = mod
            sub_locs = getattr(mod, "__path__", None)
            if sub_locs is None and idx + 1 < len(parts):
                raise ModuleNotFoundError(
                    f"No module named {dotted_name!r}; {prefix!r} is not a package",
                    name=dotted_name,
                )
            current_search = list(sub_locs) if sub_locs is not None else ()
        return last_mod

    def _import_hook(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        target_name = name
        if level > 0 and globals is not None:
            pkg = globals.get("__package__") or ""
            if pkg:
                pkg_parts = pkg.split(".")
                if level - 1 <= len(pkg_parts):
                    base_pkg = ".".join(pkg_parts[: len(pkg_parts) - (level - 1)])
                    target_name = (
                        f"{base_pkg}.{name}" if (base_pkg and name) else (base_pkg or name)
                    )
        top_name = target_name.partition(".")[0]
        if top_name and (level > 0 or not self._host_meta_path_claims(top_name)):
            target_mod = self._load_chain(target_name)
            if target_mod is not None:
                top_mod = self.modules[top_name]
                if not fromlist:
                    return top_mod
                pkg_paths = getattr(target_mod, "__path__", None)
                if pkg_paths is not None:
                    for item in fromlist:
                        if item != "*" and not hasattr(target_mod, item):
                            sub = self._load_one(f"{target_name}.{item}", list(pkg_paths))
                            if sub is not None:
                                setattr(target_mod, item, sub)
                return target_mod
        return builtins.__import__(name, globals, locals, fromlist, level)


_ACTIVE_CARTRIDGE_NS: ContextVar[_CartridgeNamespace | None] = ContextVar(
    "looplet_active_cartridge_ns", default=None
)


@contextmanager
def _cartridge_import_scope(root: Path) -> Iterator[_CartridgeNamespace]:
    ns = _CartridgeNamespace(root)
    token = _ACTIVE_CARTRIDGE_NS.set(ns)
    try:
        yield ns
    finally:
        _ACTIVE_CARTRIDGE_NS.reset(token)


def _import_module_from_path(path: Path, module_name: str) -> Any:
    """Load a workspace-shipped Python module from ``path``."""
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
    active_ns = _ACTIVE_CARTRIDGE_NS.get()
    if active_ns is not None:
        module.__dict__["__builtins__"] = active_ns.builtins
    previous = _sys.modules.get(module_name)
    _sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        if previous is not None:
            _sys.modules[module_name] = previous
        else:
            _sys.modules.pop(module_name, None)
        raise
    if active_ns is not None:
        active_ns.modules[module_name] = module
    setattr(module, "__looplet_source_path__", str(path))
    for value in module.__dict__.values():
        if getattr(value, "__module__", None) != module_name:
            continue
        try:
            value.__looplet_source_path__ = str(path)
        except (AttributeError, TypeError):
            pass
    return module
