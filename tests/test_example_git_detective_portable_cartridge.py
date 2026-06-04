"""End-to-end dogfood of the fully-portable ``git_detective_portable``.

Cross-runtime twin of ``git_detective``. All 10 tools are served by a
bundled MCP stdio server, the CouplingGuard threshold policy is a
``kind: lep`` hook, and compaction is the host-provided RUNTIME-tier
``compact_service`` resource. The original's INPROCESS ``repo_config``
resource is replaced by ``$LOOPLET_PROJECT_ROOT`` resolution in the
server, so nothing pins it to Python.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = (
    Path(__file__).resolve().parent.parent / "examples" / "git_detective_portable.cartridge"
)


def test_git_detective_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()
    details = {c.detail for c in report.components}
    assert "mcp" in details  # all 10 tools
    assert "lep" in details  # CouplingGuard
    assert "host-service" in details  # compact_service (RUNTIME tier)
    # The original's INPROCESS repo_config resource must be gone.
    assert "repo_config" not in {c.name for c in report.components}


def _make_repo(root: Path) -> None:
    def run(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    run("init", "-q")
    run("config", "user.email", "detective@example.com")
    run("config", "user.name", "Sherlock")
    (root / "a.py").write_text("print('a')\n")
    (root / "b.py").write_text("print('b')\n")
    run("add", "-A")
    run("commit", "-q", "-m", "feat: initial files")
    (root / "a.py").write_text("print('a2')\n")
    (root / "b.py").write_text("print('b2')\n")
    run("add", "-A")
    run("commit", "-q", "-m", "fix: tweak both files together")


@pytest.mark.timeout(90)
def test_git_detective_portable_cross_process_composition(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    # The MCP server resolves the target repo from $LOOPLET_PROJECT_ROOT,
    # which the spawned subprocess inherits from this process's environment.
    prev = os.environ.get("LOOPLET_PROJECT_ROOT")
    os.environ["LOOPLET_PROJECT_ROOT"] = str(repo)
    try:
        preset = cartridge_to_preset(_CARTRIDGE)
        try:
            assert len(preset.mcp_adapters) == 1
            assert preset.state_service_handles == []
            names = set(preset.tools.tool_names)
            assert {
                "repo_overview",
                "contributor_stats",
                "recent_activity",
                "file_hotspots",
                "coupled_files",
                "commit_patterns",
                "directory_structure",
                "file_age_analysis",
                "think",
                "done",
            } <= names

            guard = next(
                h for h in preset.hooks if "CouplingGuard" in getattr(h, "_cartridge_id", "")
            )
            for h in preset.hooks:
                if hasattr(h, "pre_loop"):
                    h.pre_loop(None, None, None)

            try:
                # (1) LEP CouplingGuard refuses a non-positive threshold,
                #     allows a sane one.
                assert (
                    guard.check_permission(
                        ToolCall(tool="coupled_files", args={"min_coupling": "0"}),
                        None,
                    )
                    is False
                )
                assert (
                    guard.check_permission(
                        ToolCall(tool="coupled_files", args={"min_coupling": "2"}),
                        None,
                    )
                    is True
                )

                # (2) MCP repo_overview over the real temp git repo.
                ro = preset.tools.dispatch(ToolCall(tool="repo_overview", args={}))
                assert ro.error is None
                assert ro.data["repo_name"] == "repo"
                assert ro.data["total_commits"] == 2

                # (3) MCP contributor_stats.
                cs = preset.tools.dispatch(ToolCall(tool="contributor_stats", args={}))
                assert cs.error is None
                assert cs.data["contributor_count"] == 1
                assert cs.data["total_commits"] == 2

                # (4) MCP file_hotspots — a.py and b.py both changed twice.
                fh = preset.tools.dispatch(ToolCall(tool="file_hotspots", args={"top_n": "5"}))
                assert fh.error is None
                hotspot_files = {h["file"] for h in fh.data["hotspots"]}
                assert {"a.py", "b.py"} <= hotspot_files

                # (5) MCP coupled_files — a.py and b.py co-change.
                cf = preset.tools.dispatch(
                    ToolCall(tool="coupled_files", args={"min_coupling": "2"})
                )
                assert cf.error is None
                assert cf.data["coupling_threshold"] == 2

                # (6) MCP commit_patterns (deterministic; no LLM assessment).
                cp = preset.tools.dispatch(ToolCall(tool="commit_patterns", args={}))
                assert cp.error is None
                assert cp.data["total_analyzed"] == 2
                assert cp.data["conventional_commits_pct"] == 100.0
                assert "commit_quality_assessment" not in cp.data  # degraded

                # (7) think + done round-trip.
                th = preset.tools.dispatch(
                    ToolCall(tool="think", args={"thought": "tight coupling"})
                )
                assert th.error is None
                assert th.data["noted"] is True

                done = preset.tools.dispatch(
                    ToolCall(tool="done", args={"summary": "health report"})
                )
                assert done.error is None
                assert done.data["status"] == "completed"
            finally:
                for h in preset.hooks:
                    if hasattr(h, "close"):
                        h.close()
        finally:
            preset.close()
    finally:
        if prev is None:
            os.environ.pop("LOOPLET_PROJECT_ROOT", None)
        else:
            os.environ["LOOPLET_PROJECT_ROOT"] = prev
