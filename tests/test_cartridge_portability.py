"""Tests for the whole-cartridge portability report.

``looplet.cartridge.portability.analyse_cartridge`` is a *static*
analyser: it reads a cartridge directory (no Python bodies imported) and
classifies every component into one of three portability tiers —
``protocol`` (config/prompts/MCP tools/LEP hooks), ``stdlib``
(``builtin_tools``/``builtin_hooks`` declarative references), or
``inprocess`` (Python tool bodies, class hooks, ``resources/*.py``).

These tests author tiny synthetic cartridges so the assertions are
hermetic, plus a parametrised smoke test over the shipped
``examples/*.cartridge`` so a regression in any real cartridge's
component shape is caught.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge
from looplet.cartridge.portability import INPROCESS, PROTOCOL, RUNTIME, STDLIB

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


# ── synthetic cartridges ──────────────────────────────────────


def test_fully_portable_cartridge_has_portable_profile(tmp_path: Path) -> None:
    """config + prompts + MCP tool + LEP hook → no Python pins."""
    root = tmp_path / "portable.cartridge"
    _write(
        root / "config.yaml",
        """
        max_steps: 5
        mcp_servers:
          calc:
            command: "python3 ${runtime.cartridge_root}/_server/calc.py"
        """,
    )
    _write(root / "prompts" / "system.md", "You are a calculator.\n")
    _write(
        root / "hooks" / "00_guard" / "config.yaml",
        """
        kind: lep
        server: server.py
        """,
    )

    report = analyse_cartridge(root)

    assert report.profile == "portable"
    assert report.blockers == ()
    tiers = {(c.kind, c.name): c.tier for c in report.components}
    assert tiers[("tool", "mcp:calc")] == PROTOCOL
    assert tiers[("hook", "00_guard")] == PROTOCOL
    assert tiers[("config", "config.yaml")] == PROTOCOL
    assert tiers[("prompts", "prompts/")] == PROTOCOL


def test_python_tool_body_pins_to_python_host(tmp_path: Path) -> None:
    root = tmp_path / "pinned.cartridge"
    _write(root / "config.yaml", "max_steps: 5\n")
    _write(root / "tools" / "greet" / "tool.yaml", "name: greet\n")
    _write(root / "tools" / "greet" / "execute.py", "def execute():\n    return {}\n")

    report = analyse_cartridge(root)

    assert report.profile == "python-host"
    blockers = {c.name for c in report.blockers}
    assert "greet" in blockers
    greet = next(c for c in report.components if c.name == "greet")
    assert greet.tier == INPROCESS
    assert greet.detail == "python-execute"


def test_single_file_tool_is_inprocess(tmp_path: Path) -> None:
    root = tmp_path / "single.cartridge"
    _write(root / "config.yaml", "max_steps: 5\n")
    _write(root / "tools" / "ping.py", "def execute():\n    return {}\n")

    report = analyse_cartridge(root)

    ping = next(c for c in report.components if c.name == "ping")
    assert ping.tier == INPROCESS
    assert ping.detail == "python-single-file"
    assert report.profile == "python-host"


def test_class_hook_is_inprocess_lep_hook_is_protocol(tmp_path: Path) -> None:
    root = tmp_path / "hooks.cartridge"
    _write(root / "config.yaml", "max_steps: 5\n")
    _write(
        root / "hooks" / "00_policy" / "config.yaml",
        "kind: class\nclass_name: Policy\n",
    )
    _write(root / "hooks" / "00_policy" / "hook.py", "class Policy:\n    pass\n")
    _write(
        root / "hooks" / "01_lep" / "config.yaml",
        "kind: lep\nserver: server.py\n",
    )

    report = analyse_cartridge(root)

    by_name = {c.name: c for c in report.components}
    assert by_name["00_policy"].tier == INPROCESS
    assert by_name["01_lep"].tier == PROTOCOL
    assert report.profile == "python-host"


def test_builtin_hooks_and_tools_are_stdlib_tier(tmp_path: Path) -> None:
    root = tmp_path / "stdlib.cartridge"
    _write(
        root / "config.yaml",
        """
        max_steps: 5
        builtin_tools:
          - done
        builtin_hooks:
          - stagnation
          -
            per_tool_limit:
              default_limit: 5
        """,
    )

    report = analyse_cartridge(root)

    by_name = {(c.kind, c.name): c for c in report.components}
    assert by_name[("tool", "done")].tier == STDLIB
    assert by_name[("hook", "stagnation")].tier == STDLIB
    assert by_name[("hook", "per_tool_limit")].tier == STDLIB
    # stdlib-only (no Python bodies) → still portable profile.
    assert report.profile == "portable"


def test_resource_is_inprocess_shared_ref(tmp_path: Path) -> None:
    root = tmp_path / "res.cartridge"
    _write(root / "config.yaml", "max_steps: 5\n")
    _write(root / "resources" / "store.py", "def build():\n    return {}\n")

    report = analyse_cartridge(root)

    store = next(c for c in report.components if c.name == "store")
    assert store.tier == INPROCESS
    assert store.detail == "shared-ref"
    assert report.profile == "python-host"


def test_state_service_is_protocol_tier(tmp_path: Path) -> None:
    """A ``state_services:`` entry is out-of-process shared state → protocol."""
    root = tmp_path / "ssp.cartridge"
    _write(
        root / "config.yaml",
        """
        max_steps: 5
        state_services:
          log:
            command: "python3 ${runtime.cartridge_root}/_state/log.py"
        """,
    )

    report = analyse_cartridge(root)

    svc = next(c for c in report.components if c.name == "state:log")
    assert svc.kind == "resource"
    assert svc.tier == PROTOCOL
    assert svc.detail == "ssp"
    assert report.profile == "portable"


def test_resource_backed_by_state_service_is_protocol(tmp_path: Path) -> None:
    """A ``resources/<n>.py`` whose name matches a declared state service is
    served out-of-process (the loader injects a client) → protocol, not a
    Python pin."""
    root = tmp_path / "ssp_res.cartridge"
    _write(
        root / "config.yaml",
        """
        max_steps: 5
        state_services:
          greeting_log:
            command: "python3 ${runtime.cartridge_root}/_state/greeting_log.py"
        """,
    )
    # A same-named resource file is overridden by the injected client.
    _write(
        root / "resources" / "greeting_log.py",
        "def build():\n    return {}\n",
    )

    report = analyse_cartridge(root)

    res = next(c for c in report.components if c.kind == "resource" and c.name == "greeting_log")
    assert res.tier == PROTOCOL
    assert res.detail == "state-service"
    # The declared service still emits its own protocol component too.
    assert any(c.name == "state:greeting_log" for c in report.components)
    assert report.profile == "portable"


def test_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    import json

    root = tmp_path / "x.cartridge"
    _write(root / "config.yaml", "max_steps: 5\n")
    report = analyse_cartridge(root)
    blob = json.dumps(report.to_dict())
    assert '"profile"' in blob
    assert report.counts()[PROTOCOL] >= 1


def test_non_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(NotADirectoryError):
        analyse_cartridge(missing)


# ── shipped examples smoke test ───────────────────────────────


@pytest.mark.parametrize(
    "cartridge_dir",
    sorted(p for p in _EXAMPLES.glob("*.cartridge") if p.is_dir()),
    ids=lambda p: p.name,
)
def test_example_cartridges_analyse_without_error(cartridge_dir: Path) -> None:
    report = analyse_cartridge(cartridge_dir)
    # Every cartridge has at least the config component.
    assert any(c.kind == "config" for c in report.components)
    # Profile is always one of the two known verdicts.
    assert report.profile in ("portable", "python-host")
    # Render never raises and mentions the cartridge name.
    rendered = report.render()
    assert cartridge_dir.name in rendered


def test_mcp_demo_calc_tool_is_protocol_portable() -> None:
    """The mcp_demo cartridge's `calc` tool comes from an MCP server."""
    report = analyse_cartridge(_EXAMPLES / "mcp_demo.cartridge")
    mcp_tools = [c for c in report.components if c.detail == "mcp"]
    assert mcp_tools, "expected at least one MCP-provided tool"
    assert all(c.tier == PROTOCOL for c in mcp_tools)


def test_hello_portable_example_is_fully_portable() -> None:
    """The ``hello_portable`` cartridge has zero Python-pinned components.

    It is the cross-runtime twin of ``hello`` (which is python-host): the
    greet/done tools are an MCP server, the shared greeting log is a
    state service, and both hooks are ``kind: lep``.
    """
    portable = _EXAMPLES / "hello_portable.cartridge"
    if not portable.is_dir():
        pytest.skip("hello_portable example cartridge not present")
    report = analyse_cartridge(portable)
    assert report.profile == "portable"
    assert report.blockers == ()
    details = {c.detail for c in report.components}
    assert "mcp" in details  # greet/done tools
    assert "lep" in details  # PolitenessGate + NameGuard
    assert "ssp" in details  # greeting_log state service


def test_hello_example_is_python_host() -> None:
    """The original ``hello`` cartridge stays python-host (in-process tools,
    class hook, and @ref resource) — the portable twin lives alongside it."""
    report = analyse_cartridge(_EXAMPLES / "hello.cartridge")
    assert report.profile == "python-host"


def test_builtin_runtime_service_resource_is_runtime_tier(tmp_path: Path) -> None:
    """A resource that only wraps a looplet builtin host service (compaction,
    skill manager, …) is RUNTIME tier — a host responsibility, not an
    author-owned shared-state blocker."""
    root = tmp_path / "svc.cartridge"
    (root / "resources").mkdir(parents=True)
    (root / "config.yaml").write_text("max_steps: 3\n", encoding="utf-8")
    (root / "resources" / "compact_service.py").write_text(
        textwrap.dedent(
            """\
            from __future__ import annotations

            from looplet.compact import default_compact_service

            def build(runtime=None):
                return default_compact_service(keep_recent=3)
            """
        ),
        encoding="utf-8",
    )
    report = analyse_cartridge(root)
    by_name = {c.name: c for c in report.components}
    assert by_name["compact_service"].tier == RUNTIME
    # RUNTIME components are not blockers.
    assert by_name["compact_service"] not in report.blockers


def test_author_state_resource_stays_inprocess(tmp_path: Path) -> None:
    """A resource with author-owned mutable state (no builtin-service factory)
    is still an INPROCESS blocker."""
    root = tmp_path / "state.cartridge"
    (root / "resources").mkdir(parents=True)
    (root / "config.yaml").write_text("max_steps: 3\n", encoding="utf-8")
    (root / "resources" / "file_cache.py").write_text(
        textwrap.dedent(
            """\
            from __future__ import annotations

            def build(runtime=None):
                return {}
            """
        ),
        encoding="utf-8",
    )
    report = analyse_cartridge(root)
    by_name = {c.name: c for c in report.components}
    assert by_name["file_cache"].tier == INPROCESS
    assert by_name["file_cache"] in report.blockers


def test_compact_service_examples_are_runtime_not_blockers() -> None:
    """The real example cartridges' ``compact_service`` resource is a builtin
    host service, so it must not count as a python-host blocker."""
    for name in ("dep_doctor", "threat_intel", "git_detective", "coder"):
        report = analyse_cartridge(_EXAMPLES / f"{name}.cartridge")
        compact = next(c for c in report.components if c.name == "compact_service")
        assert compact.tier == RUNTIME, name
        assert compact not in report.blockers, name


def test_extends_inheritance_folds_in_parent_blockers(tmp_path: Path) -> None:
    """A child that ``extends:`` a python-host parent inherits its blockers.

    Without resolving ``extends:`` the analyzer would silently under-count
    (e.g. a one-tool child extending a many-blocker parent would look almost
    portable). Inherited components are tagged ``(inherited)``.
    """
    parent = tmp_path / "parent.cartridge"
    _write(parent / "config.yaml", "max_steps: 5\n")
    _write(parent / "tools" / "heavy" / "execute.py", "def execute():\n    return {}\n")
    _write(parent / "tools" / "heavy" / "tool.yaml", "name: heavy\n")

    child = tmp_path / "child.cartridge"
    _write(child / "config.yaml", "extends: ../parent.cartridge\nmax_steps: 5\n")
    _write(child / "tools" / "own" / "execute.py", "def execute():\n    return {}\n")
    _write(child / "tools" / "own" / "tool.yaml", "name: own\n")

    report = analyse_cartridge(child)
    names = {c.name for c in report.components}
    assert "own" in names  # child's own tool
    assert "heavy (inherited)" in names  # parent's tool, folded in
    # Both are python-host blockers.
    assert report.profile == "python-host"
    assert any(b.name == "heavy (inherited)" for b in report.blockers)


def test_agent_factory_inherits_coder_blockers() -> None:
    """``agent_factory`` extends ``coder``; the report must reflect coder's
    inherited Python-pinned components, not just its own one tool."""
    report = analyse_cartridge(_EXAMPLES / "agent_factory.cartridge")
    assert report.profile == "python-host"
    inherited = [b for b in report.blockers if b.name.endswith("(inherited)")]
    assert len(inherited) >= 10  # coder contributes many blockers
