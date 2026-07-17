"""End-to-end dogfood of the fully-portable ``coder_portable`` cartridge.

This is the cross-runtime twin of ``coder`` - the most complex example
cartridge (16 tools, a shared ``FileCache`` resource, and six hooks).
Every portable component lives out of process behind a protocol:

* 16 tools .............. MCP stdio server   (``_mcp/tools_server.py``)
* shared FileCache ...... State Service       (``_state/file_cache.py``)
* TestGuard/FileCache/
  StaleFile/Linter/
  ShellSafetyGate hooks .. ``kind: lep`` servers
* web_fetch / subagent .. reach the host LLM over the Model Gateway
* compaction ........... a RUNTIME builtin (``default_compact_service``)

The tests prove the twin composes across process boundaries:

1. Static profile is ``portable`` with zero blockers.
2. The loader spawns exactly one MCP server + one State Service, and the
   16 tools are reachable across the boundary.
3. The deterministic file tools (write/read/edit/grep) run out of
   process AND their reads land in the SAME FileCache State Service that
   a DIFFERENT process (the cache-backed LEP hooks) would observe -
   reproducing the original in-process ``@ref`` sharing with zero shared
   Python objects.
4. The ``subagent`` tool reaches the host LLM over the Model Gateway: it
   degrades cleanly with no backend bound, and runs a real sub-loop when
   a scripted backend is bound to the twin's gateway.
"""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = Path(__file__).resolve().parent.parent / "examples" / "coder_portable.cartridge"

_TOOLS = {
    "bash",
    "list_dir",
    "read_file",
    "write_file",
    "edit_file",
    "multi_edit",
    "notebook_edit",
    "glob",
    "grep",
    "git_inspect",
    "worktree",
    "web_fetch",
    "subagent",
    "todo",
    "think",
    "done",
}


class _ScriptedLLM:
    """Deterministic backend: returns canned replies in order."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else "DONE"


def test_coder_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()


@pytest.mark.timeout(90)
def test_coder_portable_loads_from_path_with_spaces(monkeypatch, tmp_path) -> None:
    project = tmp_path / "project with spaces"
    project.mkdir()
    cartridge = tmp_path / "portable coder.cartridge"
    shutil.copytree(_CARTRIDGE, cartridge)
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(project))

    preset = cartridge_to_preset(cartridge, strict=True)
    try:
        assert len(preset.state_service_handles) == 1
        assert len(preset.mcp_adapters) == 1
        assert set(preset.tools.tool_names) == _TOOLS
    finally:
        preset.close()


@pytest.mark.timeout(90)
def test_coder_portable_cross_process_tools(monkeypatch, tmp_path) -> None:
    # The MCP server + State Service subprocesses bind their project root
    # from the environment at spawn time, so set it BEFORE loading.
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(tmp_path))
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        # Exactly one MCP server + one State Service were spawned.
        assert len(preset.state_service_handles) == 1
        assert len(preset.mcp_adapters) == 1

        reg = preset.tools
        assert set(reg.tool_names) == _TOOLS

        cache = preset.resources["file_cache"]

        # (1) write_file (separate process) creates a real file.
        target = "pkg/mod.py"
        w = reg.dispatch(
            ToolCall(
                tool="write_file",
                args={"file_path": target, "content": "def add(a, b):\n    return a + b\n"},
            )
        )
        assert w.error is None
        # NB: the framework strips whitespace on string args, so the
        # trailing newline is trimmed - assert on the body, not bytes.
        assert (tmp_path / target).read_text().rstrip() == "def add(a, b):\n    return a + b"

        # (2) read_file (separate process) reads it back AND records the
        #     read into the shared FileCache State Service.
        r = reg.dispatch(ToolCall(tool="read_file", args={"file_path": target}))
        assert r.error is None
        assert "def add(a, b):" in r.data["content"]
        assert r.data["total_lines"] == 2

        # (3) The read is visible across the process boundary: a DIFFERENT
        #     process (the State Service) recorded it, exactly as the
        #     cache-backed LEP hooks would observe.
        assert cache.was_read(target) is True

        # (4) edit_file (separate process) mutates the file in place.
        e = reg.dispatch(
            ToolCall(
                tool="edit_file",
                args={
                    "file_path": target,
                    "old_string": "a + b",
                    "new_string": "a + b  # sum",
                },
            )
        )
        assert e.error is None
        assert "# sum" in (tmp_path / target).read_text()

        # (5) grep (separate process) finds the new content.
        g = reg.dispatch(ToolCall(tool="grep", args={"pattern": "# sum", "path": "."}))
        assert g.error is None
        assert g.data["count"] >= 1

        # (6) think + done (no-ctx tools) dispatch across the boundary.
        t = reg.dispatch(ToolCall(tool="think", args={"thought": "looks good"}))
        assert t.error is None
        d = reg.dispatch(ToolCall(tool="done", args={"summary": "added add()"}))
        assert d.error is None
    finally:
        preset.close()
        assert preset.state_service_handles == []
        assert preset.mcp_adapters == []


def test_coder_portable_structured_param_schema_and_coercion(monkeypatch, tmp_path) -> None:
    """Regression: structured tool params (``multi_edit.edits: list``) must
    advertise as JSON-Schema ``array`` - not ``string`` - and a JSON-string
    value for such a param must be coerced to a list across the MCP boundary.

    The vendored tool modules use ``from __future__ import annotations``
    (PEP 563), so ``inspect.signature`` reports the annotation as the string
    ``"list"``. A type-keyed lookup misses it and falls back to ``"string"``,
    which (a) tells the model the param is a string so it double-encodes the
    value and (b) trips the tool's list validation. This pins both halves of
    the fix: correct schema type + host-side str→list coercion.
    """
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(tmp_path))
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        reg = preset.tools
        multi_edit = reg._tools["multi_edit"]
        assert (multi_edit.parameters or {}).get("edits") == "array"
        # subagent.max_steps is an int param → integer, not string.
        subagent = reg._tools["subagent"]
        assert (subagent.parameters or {}).get("max_steps", "").endswith("integer")

        target = "frob.py"
        (tmp_path / target).write_text("VALUE = 1\n")
        reg.dispatch(ToolCall(tool="read_file", args={"file_path": target}))

        # Pass ``edits`` as a JSON STRING (what a model does when the schema
        # mislabels the param). The host adapter must parse it to a list.
        import json as _json

        edits_str = _json.dumps([{"old_string": "VALUE = 1", "new_string": "VALUE = 2"}])
        m = reg.dispatch(
            ToolCall(tool="multi_edit", args={"file_path": target, "edits": edits_str})
        )
        assert m.error is None
        assert "error" not in (m.data or {}), m.data
        assert (tmp_path / target).read_text().strip() == "VALUE = 2"
    finally:
        preset.close()
        assert preset.state_service_handles == []
        assert preset.mcp_adapters == []


@pytest.mark.timeout(60)
def test_coder_portable_subagent_degrades_without_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(tmp_path))
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        # No backend bound (load-time default) → the documented
        # ctx.llm-None degradation branch, identical to the original.
        s = preset.tools.dispatch(
            ToolCall(tool="subagent", args={"prompt": "investigate the repo"})
        )
        assert s.error is None
        assert "requires an active ctx.llm" in s.data["error"]
        assert "recovery" in s.data
    finally:
        preset.close()


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="model gateway requires AF_UNIX sockets",
)
@pytest.mark.timeout(90)
def test_coder_portable_subagent_runs_over_model_gateway(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(tmp_path))
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        # The loader auto-started a Model Gateway (the cartridge has
        # mcp_servers); bind a scripted backend as AgentPreset.run would.
        assert preset.model_gateway is not None, "gateway should auto-start"
        backend_used = _ScriptedLLM(['{"tool": "done", "args": {"summary": "investigated"}}'])
        preset.model_gateway.set_backend(backend_used)

        # The subagent runs an isolated sub-loop whose LLM calls cross the
        # process boundary (MCP tool subprocess → host gateway → backend).
        s = preset.tools.dispatch(
            ToolCall(
                tool="subagent",
                args={"prompt": "summarize the repo", "max_steps": 2},
            )
        )
        assert s.error is None
        assert "error" not in s.data
        # The sub-loop completed via the host LLM: at least one
        # generation crossed the boundary (MCP subprocess → gateway).
        assert s.data["llm_calls"] >= 1
        assert "summary" in s.data
        # The scripted backend's reply was consumed by the sub-loop.
        assert backend_used.prompts, "gateway backend was never called"
    finally:
        preset.close()
