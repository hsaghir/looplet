"""Tests for looplet.hot_reload — mtime-poll workspace reloading."""

from __future__ import annotations

import time
from pathlib import Path

from looplet.hot_reload import WorkspaceWatcher, fingerprint_workspace
from looplet.scaffold import scaffold_cartridge


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


def test_pycache_and_dotfiles_ignored(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    (ws / "__pycache__").mkdir(exist_ok=True)
    (ws / "__pycache__" / "junk.cpython-313.pyc").write_text("x")
    (ws / ".cache").mkdir(exist_ok=True)
    (ws / ".cache" / "noise.py").write_text("x")

    fp = fingerprint_workspace(ws)
    assert not any("__pycache__" in k for k in fp)
    assert not any(".cache" in k for k in fp)


def test_force_reload_works_without_changes(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path / "w.workspace", name="w")
    w = WorkspaceWatcher(ws, runtime={"workspace": str(tmp_path)})
    p1 = w.preset()
    p2 = w.force_reload()
    # Different objects; same workspace
    assert p1 is not p2
