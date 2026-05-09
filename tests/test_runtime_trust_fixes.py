"""Regression tests for two frictions surfaced by the runtime-trust dogfood.

1. **Hot-reload bytecode-cache staleness.** Python's ``SourceFileLoader``
   writes ``__pycache__/<name>.cpython-XYZ.pyc`` next to a workspace
   tool's ``execute.py`` and reuses that bytecode whenever the source
   mtime matches the value stored in the ``.pyc`` header. The mtime is
   recorded with **second** resolution, so two writes within the same
   wall-clock second silently re-use the old bytecode — a cartridge
   reloaded after a fast edit returned the previous tool body.
   ``_import_module_from_path`` now reads the source verbatim and
   ``compile`` + ``exec`` it directly, so the cache is bypassed and
   no ``__pycache__`` is written into the user's cartridge.

2. **scaffold-then-edit silently rejects calls.** ``scaffold_workspace``
   writes ``parameters: {}`` and ``def execute(ctx, **kwargs)`` together.
   Users replace ``**kwargs`` with explicit keyword params (``*, name: str``)
   but forget to update ``parameters:``. The dispatcher then rejects
   every call with VALIDATION because the schema advertises zero
   parameters. The loader now warns at strict-load time when an
   ``execute.py`` declares explicit kwargs but ``tool.yaml`` is empty.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from pathlib import Path

import pytest

from looplet import workspace_to_preset
from looplet.types import ToolCall


def _write_basic_workspace(root: Path) -> Path:
    ws = root / "rt.workspace"
    ws.mkdir()
    (ws / "workspace.json").write_text(json.dumps({"name": "rt", "schema_version": 1}) + "\n")
    (ws / "config.yaml").write_text("max_steps: 4\ndone_tool: done\n")
    (ws / "prompts").mkdir()
    (ws / "prompts" / "system.md").write_text("test agent\n")
    (ws / "tools" / "done").mkdir(parents=True)
    (ws / "tools" / "done" / "tool.yaml").write_text(
        "name: done\ndescription: Done.\nparameters: {}\n"
    )
    (ws / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'done': True}\n"
    )
    return ws


# ── fix 1: hot-reload + no __pycache__ pollution ──────────────────


def test_workspace_reload_picks_up_edited_tool_body(tmp_path: Path) -> None:
    """Two loads in the same wall-clock second see the new tool body."""
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "stamp").mkdir(parents=True)
    (ws / "tools" / "stamp" / "tool.yaml").write_text(
        "name: stamp\ndescription: Return version.\nparameters: {}\n"
    )
    (ws / "tools" / "stamp" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'version': 1}\n"
    )

    preset1 = workspace_to_preset(str(ws), strict=True)
    out1 = preset1.tools.dispatch(ToolCall(tool="stamp", args={}, reasoning="x", call_id="1")).data
    assert (out1 or {}).get("version") == 1

    # Edit immediately — must NOT depend on mtime ticking past one second.
    (ws / "tools" / "stamp" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'version': 2}\n"
    )

    preset2 = workspace_to_preset(str(ws), strict=True)
    out2 = preset2.tools.dispatch(ToolCall(tool="stamp", args={}, reasoning="x", call_id="2")).data
    assert (out2 or {}).get("version") == 2, (
        f"second load returned stale body: {out2!r} "
        f"(if you see version=1, the bytecode cache is back)"
    )


def test_workspace_load_does_not_pollute_cartridge_with_pycache(tmp_path: Path) -> None:
    """Loading a workspace must not create __pycache__/ inside the cartridge.

    The cartridge boundary promises 'just files' — leaving compiled
    bytecode in tools/<name>/__pycache__/ violates that promise (and
    surfaces in `git status`, breaks `find . -type f` listings, etc.).
    """
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "x").mkdir(parents=True)
    (ws / "tools" / "x" / "tool.yaml").write_text("name: x\ndescription: x.\nparameters: {}\n")
    (ws / "tools" / "x" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'ok': True}\n"
    )

    workspace_to_preset(str(ws), strict=True)

    pycache_dirs = list(ws.rglob("__pycache__"))
    assert pycache_dirs == [], (
        f"workspace_to_preset created __pycache__ inside the cartridge: {pycache_dirs}"
    )


def test_workspace_reload_after_seconds_still_works(tmp_path: Path) -> None:
    """The slow-edit path (mtime ticked past one second) still works.

    Sanity: if someone reverts the cache fix to use SourceFileLoader
    naively, this test would *also* catch the case where mtime DOES
    tick but bytecode invalidation has another bug.
    """
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "stamp").mkdir(parents=True)
    (ws / "tools" / "stamp" / "tool.yaml").write_text(
        "name: stamp\ndescription: x.\nparameters: {}\n"
    )
    (ws / "tools" / "stamp" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'version': 1}\n"
    )
    workspace_to_preset(str(ws), strict=True)
    time.sleep(1.1)
    (ws / "tools" / "stamp" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'version': 2}\n"
    )
    preset = workspace_to_preset(str(ws), strict=True)
    out = preset.tools.dispatch(ToolCall(tool="stamp", args={}, reasoning="x", call_id="1")).data
    assert (out or {}).get("version") == 2


# ── fix 2: parameters-vs-signature mismatch warning ───────────────


def test_loader_warns_on_empty_parameters_with_explicit_signature(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Mismatched ``parameters: {}`` and explicit kwargs trigger a warning."""
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "greet").mkdir(parents=True)
    (ws / "tools" / "greet" / "tool.yaml").write_text(
        "name: greet\ndescription: Greet.\nparameters: {}\n"
    )
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name: str) -> dict:\n    return {'hi': name}\n"
    )

    with caplog.at_level(logging.WARNING, logger="looplet.workspace"):
        workspace_to_preset(str(ws))

    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("greet" in m and "parameters" in m and "VALIDATION" in m for m in msgs), (
        f"expected parameters-mismatch warning, got: {msgs}"
    )


def test_loader_strict_rejects_empty_parameters_with_explicit_signature(
    tmp_path: Path,
) -> None:
    """Strict mode upgrades the warning to a structured error."""
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "greet").mkdir(parents=True)
    (ws / "tools" / "greet" / "tool.yaml").write_text(
        "name: greet\ndescription: Greet.\nparameters: {}\n"
    )
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name: str) -> dict:\n    return {'hi': name}\n"
    )
    with pytest.raises(Exception, match="parameters"):
        workspace_to_preset(str(ws), strict=True)


def test_loader_does_not_warn_when_kwargs_signature(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``def execute(ctx, **kwargs)`` is internally consistent with empty parameters."""
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "greet").mkdir(parents=True)
    (ws / "tools" / "greet" / "tool.yaml").write_text(
        "name: greet\ndescription: Greet.\nparameters: {}\n"
    )
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, **kwargs) -> dict:\n    return {**kwargs}\n"
    )
    with caplog.at_level(logging.WARNING, logger="looplet.workspace"):
        workspace_to_preset(str(ws), strict=True)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert not any("greet" in m and "VALIDATION" in m for m in msgs), (
        f"unexpected mismatch warning for **kwargs: {msgs}"
    )


def test_loader_does_not_warn_when_parameters_filled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Filled ``parameters:`` and explicit signature is the happy path."""
    ws = _write_basic_workspace(tmp_path)
    (ws / "tools" / "greet").mkdir(parents=True)
    (ws / "tools" / "greet" / "tool.yaml").write_text(
        textwrap.dedent("""\
            name: greet
            description: Greet.
            parameters:
              name: { type: string }
        """)
    )
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name: str) -> dict:\n    return {'hi': name}\n"
    )
    with caplog.at_level(logging.WARNING, logger="looplet.workspace"):
        workspace_to_preset(str(ws), strict=True)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert not any("greet" in m and "VALIDATION" in m for m in msgs), (
        f"unexpected mismatch warning when parameters are filled: {msgs}"
    )


# ── fix 3: watcher fingerprint must detect content edits even when
# mtime resolution is coarse (tmpfs / network mounts) ─────────────────


def test_fingerprint_detects_content_edit_with_same_mtime(tmp_path: Path) -> None:
    """fingerprint changes when content changes even if mtime_ns collides."""
    from looplet.hot_reload import fingerprint_workspace  # noqa: PLC0415

    ws = _write_basic_workspace(tmp_path)
    src = ws / "tools" / "done" / "execute.py"
    src.write_text("def execute(ctx) -> dict:\n    return {'ver': 1}\n")
    fp1 = fingerprint_workspace(ws)

    # Two writes back-to-back can collide on mtime even with ns precision
    # on tmpfs. We cannot reliably *force* the collision in a unit test,
    # but we *can* verify that fingerprint changes whenever the content
    # changes — which is the property we actually care about.
    src.write_text("def execute(ctx) -> dict:\n    return {'ver': 2}\n")
    fp2 = fingerprint_workspace(ws)

    assert fp1 != fp2, "content edit did not change fingerprint; watcher would miss the edit"


def test_watcher_detects_change_after_edit(tmp_path: Path) -> None:
    """WorkspaceWatcher.changed() returns True after a content edit."""
    from looplet.hot_reload import WorkspaceWatcher  # noqa: PLC0415

    ws = _write_basic_workspace(tmp_path)
    src = ws / "tools" / "done" / "execute.py"
    src.write_text("def execute(ctx) -> dict:\n    return {'ver': 1}\n")

    watcher = WorkspaceWatcher(ws)
    # The watcher is lazy: it seeds its fingerprint on the first
    # preset() call. Pre-load so the subsequent changed() probe is
    # honest.
    watcher.preset()
    assert not watcher.changed(), "watcher reported change with no edit"

    src.write_text("def execute(ctx) -> dict:\n    return {'ver': 2}\n")
    assert watcher.changed(), "watcher missed a content edit"


# ── fix 4: rejection reason mirrored into ToolResult.error ─────────────


def test_output_schema_rejection_populates_tool_result_error(tmp_path: Path) -> None:
    """output_schema rejection puts the reason on .error, not just .data.

    Builders who grep tool_result.error to find failures otherwise miss
    schema-level rejections entirely.
    """
    import json as _json  # noqa: PLC0415

    from looplet import (  # noqa: PLC0415
        DefaultState,
        MockLLMBackend,
        composable_loop,
        workspace_to_preset,
    )

    ws = tmp_path / "se.workspace"
    ws.mkdir()
    (ws / "workspace.json").write_text(_json.dumps({"name": "se", "schema_version": 1}) + "\n")
    (ws / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    (ws / "prompts").mkdir()
    (ws / "prompts" / "system.md").write_text("go\n")
    (ws / "tools" / "done").mkdir(parents=True)
    (ws / "tools" / "done" / "tool.yaml").write_text(
        textwrap.dedent("""\
            name: done
            description: x
            parameters:
              summary: { type: string }
            output_schema:
              type: object
              required: [summary]
              properties:
                summary: { type: string }
        """)
    )
    (ws / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx, **k) -> dict:\n    return {**k}\n"
    )

    preset = workspace_to_preset(str(ws), strict=True)
    backend = MockLLMBackend(
        responses=[
            _json.dumps({"tool": "done", "args": {}, "reasoning": "r", "call_id": "1"}),
            _json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "r", "call_id": "2"}
            ),
        ]
    )
    state = DefaultState(max_steps=3)
    steps = list(
        composable_loop(
            llm=backend,
            tools=preset.tools,
            state=state,
            config=preset.config,
            hooks=preset.hooks,
            task={"description": "go"},
        )
    )
    # First step: malformed -> error must be set, data.rejected too.
    first = steps[0]
    assert first.tool_result.error is not None, (
        "output_schema rejection did not set tool_result.error"
    )
    assert "schema" in first.tool_result.error.lower()
    assert (first.tool_result.data or {}).get("rejected") is True
    # Second step: well-formed, accepted.
    second = steps[1]
    assert second.tool_result.error is None
    assert (second.tool_result.data or {}).get("summary") == "ok"
