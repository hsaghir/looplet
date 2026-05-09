"""Tests for AgentsMdMemorySource — Pi/Claude-Code-style AGENTS.md walking."""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet import AgentsMdMemorySource


def test_walks_parents_outermost_first(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("OUTER")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    (sub / "AGENTS.md").write_text("INNER")

    src = AgentsMdMemorySource(start=sub, stop=tmp_path)
    text = src.load(state=None) or ""
    assert "OUTER" in text and "INNER" in text
    assert text.index("OUTER") < text.index("INNER")


def test_claude_md_fallback(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("CLAUDE-NOTES")
    src = AgentsMdMemorySource(start=tmp_path, stop=tmp_path)
    assert "CLAUDE-NOTES" in (src.load(state=None) or "")


def test_agents_md_wins_over_claude_md_in_same_dir(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("FROM-AGENTS")
    (tmp_path / "CLAUDE.md").write_text("FROM-CLAUDE")
    src = AgentsMdMemorySource(start=tmp_path, stop=tmp_path)
    text = src.load(state=None) or ""
    assert "FROM-AGENTS" in text
    assert "FROM-CLAUDE" not in text


def test_returns_none_when_nothing_found(tmp_path: Path) -> None:
    src = AgentsMdMemorySource(start=tmp_path, stop=tmp_path)
    assert src.load(state=None) is None


def test_max_chars_truncates(tmp_path: Path) -> None:
    big = "X" * 5000
    (tmp_path / "AGENTS.md").write_text(big)
    src = AgentsMdMemorySource(start=tmp_path, stop=tmp_path, max_chars=200)
    text = src.load(state=None) or ""
    assert "[…truncated…]" in text
    assert len(text) < 600  # header + budget + marker, well under raw 5000


def test_loadable_into_render_memory(tmp_path: Path) -> None:
    from looplet.memory import render_memory

    (tmp_path / "AGENTS.md").write_text("HELLO")
    src = AgentsMdMemorySource(start=tmp_path, stop=tmp_path)
    out = render_memory([src], state=None)
    assert "HELLO" in out


@pytest.mark.smoke
def test_smoke_default_cwd_does_not_raise() -> None:
    # Constructing with all defaults must not raise even if the cwd
    # has no AGENTS.md anywhere up the tree.
    src = AgentsMdMemorySource()
    # load must always return a string-or-None, not raise
    out = src.load(state=None)
    assert out is None or isinstance(out, str)
