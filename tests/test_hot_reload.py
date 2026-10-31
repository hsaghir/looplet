"""Tests for looplet.hot_reload - mtime-poll workspace reloading."""

from __future__ import annotations

import os
import py_compile
import sys
import time
from pathlib import Path

import pytest

from looplet import ToolCall
from looplet.cartridge.hot_reload import WorkspaceWatcher, fingerprint_workspace
from looplet.cartridge.scaffold import scaffold_cartridge


def _scaffold(root: Path, name: str) -> Path:
    return scaffold_cartridge(root, name=name, tools=["greet"])


def test_fingerprint_empty_when_root_missing(tmp_path: Path) -> None:
    assert fingerprint_workspace(tmp_path / "nope") == {}


def test_fingerprint_picks_up_python_and_yaml(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    fp = fingerprint_workspace(ws)
    assert any(k.endswith(".py") for k in fp)
    assert any(k.endswith(".yaml") for k in fp)


def test_initial_load_marks_unchanged(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    w = WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)})
    _ = w.preset()
    assert w.changed() is False
    assert w.reload_if_changed() is None


def test_detects_edit_to_tool_execute(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    w = WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)})
    _ = w.preset()

    # mtime resolution can be coarse on some FSes; bump explicitly.
    target = ws / "tools" / "greet" / "execute.py"
    contents = target.read_text()
    time.sleep(0.01)
    target.write_text(contents + "\n# touched\n")
    # Make sure the mtime really moved
    import os

    new_mtime_ns = time.time_ns()
    os.utime(target, ns=(new_mtime_ns, new_mtime_ns))

    assert w.changed() is True
    new_preset = w.reload_if_changed()
    assert new_preset is not None
    # After reload, fingerprint settles again
    assert w.changed() is False


@pytest.mark.parametrize("location", ["root", "package", "resources"])
@pytest.mark.parametrize("readonly_cache", [False, True])
def test_reload_reads_current_helper_source_with_stale_bytecode(
    tmp_path: Path, location: str, readonly_cache: bool
) -> None:
    workspace = _scaffold(tmp_path / "helper.workspace", name="helper")
    helper_dir = workspace
    if location == "package":
        helper_dir = workspace / "reload_policy"
        helper_dir.mkdir()
        (helper_dir / "__init__.py").write_text("from .policy import policy\n")
    elif location == "resources":
        helper_dir = workspace / "resources"
        helper_dir.mkdir(exist_ok=True)
    helper = helper_dir / ("policy.py" if location == "package" else "reload_policy.py")
    source = 'def policy():\n    return "allow"\n'
    if location == "resources":
        source += "def build(runtime=None):\n    return policy\n"
    helper.write_text(source)
    cache = Path(
        py_compile.compile(
            str(helper),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
    )
    cached_bytes = cache.read_bytes()
    if readonly_cache:
        cache.chmod(0o444)
        cache.parent.chmod(0o555)
    (workspace / "tools/greet/tool.yaml").write_text("name: greet\nparameters: {}\n")
    (workspace / "tools/greet/execute.py").write_text(
        'from reload_policy import policy\ndef execute():\n    return {"policy": policy()}\n'
    )
    original_meta_path = list(sys.meta_path)
    original_sys_path = list(sys.path)
    try:
        with WorkspaceWatcher(
            workspace, runtime={"workspace": str(tmp_path)}, strict=True
        ) as watcher:
            original = watcher.preset()
            initial_result = original.tools.dispatch(ToolCall(tool="greet", args={}))
            assert initial_result.error is None
            assert initial_result.data == {"policy": "allow"}
            original_stat = helper.stat()
            helper.write_text(source.replace("allow", "deny_"))
            os.utime(helper, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            assert watcher.changed()

            replacement = watcher.reload_if_changed()

            assert replacement is not None
            result = replacement.tools.dispatch(ToolCall(tool="greet", args={}))
            assert result.error is None
            assert result.data == {"policy": "deny_"}
            assert not watcher.changed()
            assert cache.read_bytes() == cached_bytes
    finally:
        if readonly_cache:
            cache.parent.chmod(0o755)
            cache.chmod(0o644)
    assert sys.meta_path == original_meta_path
    assert sys.path == original_sys_path


def test_pycache_and_dotfiles_ignored(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    (ws / "__pycache__").mkdir(exist_ok=True)
    (ws / "__pycache__" / "junk.cpython-313.pyc").write_text("x")
    (ws / ".cache").mkdir(exist_ok=True)
    (ws / ".cache" / "noise.py").write_text("x")

    fp = fingerprint_workspace(ws)
    assert not any("__pycache__" in k for k in fp)
    assert not any(".cache" in k for k in fp)


def test_failed_helper_reload_restores_import_state_and_keeps_previous_preset(
    tmp_path: Path,
) -> None:
    workspace = _scaffold(tmp_path / "reload.workspace", name="reload")
    helper = workspace / "reload_failure_policy.py"
    helper.write_text("def policy():\n    return True\n")
    (workspace / "tools/greet/tool.yaml").write_text("name: greet\nparameters: {}\n")
    (workspace / "tools/greet/execute.py").write_text(
        'from reload_failure_policy import policy\ndef execute():\n    return {"valid": policy()}\n'
    )
    original_meta_path = list(sys.meta_path)
    original_sys_path = list(sys.path)
    with WorkspaceWatcher(workspace, runtime={"workspace": str(tmp_path)}, strict=True) as watcher:
        original = watcher.preset()
        helper.write_text('raise RuntimeError("synthetic helper reload failure")\n')

        with pytest.raises(RuntimeError, match="synthetic helper reload failure"):
            watcher.reload_if_changed()

        assert watcher.preset() is original
        assert watcher.changed()
        result = original.tools.dispatch(ToolCall(tool="greet", args={}))
        assert result.error is None
        assert result.data == {"valid": True}
        assert sys.meta_path == original_meta_path
        assert sys.path == original_sys_path


def test_cartridge_source_finder_does_not_replace_host_module_loader(tmp_path: Path) -> None:
    from looplet.cartridge._imports import _CartridgeSourceFinder

    cartridge_root = tmp_path / "cartridge"
    cartridge_root.mkdir()
    host_root = tmp_path / "cartridge-host"
    host_root.mkdir()
    helper = host_root / "host_policy.py"
    helper.write_text("value = True\n")
    cache = Path(py_compile.compile(str(helper), doraise=True))
    original_cache = cache.read_bytes()

    assert _CartridgeSourceFinder(cartridge_root).find_spec("host_policy", [str(host_root)]) is None
    assert cache.read_bytes() == original_cache


def test_cartridge_helper_import_preserves_host_finder_precedence(tmp_path: Path) -> None:
    from importlib.util import spec_from_file_location

    workspace = _scaffold(tmp_path / "host.workspace", name="host")
    module_name = "host_override_policy"
    (workspace / f"{module_name}.py").write_text('def policy():\n    return "local"\n')
    host_helper = tmp_path / "host_policy.py"
    host_helper.write_text('def policy():\n    return "host"\n')
    (workspace / "tools/greet/tool.yaml").write_text("name: greet\nparameters: {}\n")
    (workspace / "tools/greet/execute.py").write_text(
        f'from {module_name} import policy\ndef execute():\n    return {{"policy": policy()}}\n'
    )

    class HostFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == module_name:
                return spec_from_file_location(fullname, host_helper)

    host_finder = HostFinder()
    sys.meta_path.insert(0, host_finder)
    original_meta_path = list(sys.meta_path)
    try:
        with WorkspaceWatcher(
            workspace, runtime={"workspace": str(tmp_path)}, strict=True
        ) as watcher:
            result = watcher.preset().tools.dispatch(ToolCall(tool="greet", args={}))
            assert result.error is None
            assert result.data == {"policy": "host"}
            assert sys.meta_path == original_meta_path
    finally:
        sys.meta_path.remove(host_finder)
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("custom_loader", [False, True])
@pytest.mark.parametrize("package", [False, True])
def test_helper_reload_preserves_host_path_loader(
    tmp_path: Path, monkeypatch, custom_loader: bool, package: bool
) -> None:
    from importlib.machinery import SOURCE_SUFFIXES, FileFinder, SourceFileLoader

    workspace = _scaffold(tmp_path / "path-loader.workspace", name="path-loader")
    module_name = "host_path_loader_policy"
    if package:
        helper_dir = workspace / module_name
        helper_dir.mkdir()
        helper = helper_dir / "__init__.py"
    else:
        helper = workspace / f"{module_name}.py"
    helper.write_text("VALUE = 11\n")
    (workspace / "tools/greet/tool.yaml").write_text("name: greet\nparameters: {}\n")
    (workspace / "tools/greet/execute.py").write_text(
        f"from {module_name} import VALUE\ndef execute():\n    return {{'value': VALUE}}\n"
    )
    loader_calls = []

    class HostLoader(SourceFileLoader):
        def get_code(self, fullname):
            loader_calls.append(fullname)
            source = self.get_data(self.get_filename(fullname)).decode("utf-8")
            return compile(source + "\nVALUE = 29\n", self.path, "exec")

    if custom_loader:
        monkeypatch.setitem(
            sys.path_importer_cache,
            str(workspace),
            FileFinder(str(workspace), (HostLoader, SOURCE_SUFFIXES)),
        )
    original_meta_path = list(sys.meta_path)
    try:
        with WorkspaceWatcher(workspace, strict=True) as watcher:
            initial = watcher.preset().tools.dispatch(ToolCall(tool="greet", args={}))
            assert initial.error is None
            assert initial.data == {"value": 29 if custom_loader else 11}
            helper.write_text("VALUE = 17\n")
            replacement = watcher.force_reload()
            updated = replacement.tools.dispatch(ToolCall(tool="greet", args={}))
            assert updated.error is None
            assert updated.data == {"value": 29 if custom_loader else 17}
        assert loader_calls == ([module_name, module_name] if custom_loader else [])
        assert sys.meta_path == original_meta_path
    finally:
        sys.modules.pop(module_name, None)


def test_force_reload_works_without_changes(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    w = WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)})
    p1 = w.preset()
    p2 = w.force_reload()
    # Different objects; same workspace
    assert p1 is not p2


def test_reload_closes_previous_preset(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    watcher = WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)})
    previous = watcher.preset()
    closed: list[bool] = []
    previous.close = lambda: closed.append(True)  # type: ignore[method-assign]

    watcher.force_reload()

    assert closed == [True]


def test_watcher_close_and_context_manager_close_preset(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    watcher = WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)})
    preset = watcher.preset()
    closed: list[bool] = []
    preset.close = lambda: closed.append(True)  # type: ignore[method-assign]

    watcher.close()
    watcher.close()

    assert closed == [True]
    assert watcher.changed() is True

    with WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)}) as scoped:
        scoped.preset()
