"""Tests for looplet.session_tree."""

from __future__ import annotations

from pathlib import Path

from looplet.events import EventPayload, LifecycleEvent
from looplet.session_tree import (
    SessionTree,
    SessionTreeRecorder,
    TreeNode,
    load_tree,
    save_tree,
)
from looplet.types import ToolCall, ToolResult


def _mk_payload(step: int, tool: str, args: str = "{}", error: str | None = None) -> EventPayload:
    return EventPayload(
        event=LifecycleEvent.POST_TOOL_USE,
        step_num=step,
        tool_call=ToolCall(tool=tool, args={}, reasoning="", call_id=f"c{step}"),
        tool_result=ToolResult(tool=tool, args_summary=args, data=None, error=error),
    )


def test_root_only_has_no_leaves_other_than_root() -> None:
    t = SessionTree()
    assert len(t.leaves()) == 1
    assert t.leaves()[0].id == t.root_id


def test_linear_appends_form_single_branch() -> None:
    t = SessionTree()
    t.append(step_num=1, tool="read", args_summary="a")
    t.append(step_num=2, tool="write", args_summary="b")
    t.append(step_num=3, tool="done", args_summary="c")

    branches = t.branches()
    assert len(branches) == 1
    branch = branches[0]
    assert [n.tool for n in branch] == ["<root>", "read", "write", "done"]


def test_fork_creates_sibling_branch() -> None:
    t = SessionTree()
    t.append(step_num=1, tool="read", args_summary="a")
    fork_target = t.append(step_num=2, tool="write", args_summary="b")
    t.append(step_num=3, tool="done", args_summary="c")

    # Fork from the "write" node and continue down a new path
    t.fork(fork_target.id)
    t.append(step_num=3, tool="bash", args_summary="b'")
    t.append(step_num=4, tool="done", args_summary="d'")

    leaves = t.leaves()
    assert len(leaves) == 2
    tools_in_branches = {tuple(n.tool for n in br) for br in t.branches()}
    assert ("<root>", "read", "write", "done") in tools_in_branches
    assert ("<root>", "read", "write", "bash", "done") in tools_in_branches


def test_path_to_step() -> None:
    t = SessionTree()
    t.append(step_num=1, tool="a", args_summary="")
    t.append(step_num=2, tool="b", args_summary="")
    t.append(step_num=3, tool="c", args_summary="")
    p = t.path_to_step(2)
    assert [n.tool for n in p] == ["<root>", "a", "b"]


def test_recorder_listens_to_post_tool_use_only() -> None:
    t = SessionTree()
    rec = SessionTreeRecorder(t)

    rec.on_event(_mk_payload(1, "read"))
    rec.on_event(EventPayload(event=LifecycleEvent.PRE_LLM_CALL))  # ignored
    rec.on_event(_mk_payload(2, "write"))

    assert [n.tool for n in t.branches()[0]] == ["<root>", "read", "write"]


def test_recorder_handles_error_steps() -> None:
    t = SessionTree()
    rec = SessionTreeRecorder(t)
    rec.on_event(_mk_payload(1, "bash", error="boom"))
    leaf = t.leaves()[0]
    assert leaf.error == "boom"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    t = SessionTree()
    t.append(step_num=1, tool="a", args_summary="")
    fork = t.append(step_num=2, tool="b", args_summary="")
    t.append(step_num=3, tool="c", args_summary="")
    t.fork(fork.id)
    t.append(step_num=3, tool="d", args_summary="")

    p = save_tree(t, tmp_path / "tree.jsonl")
    loaded = load_tree(p)
    assert loaded.root_id == t.root_id
    assert len(loaded.nodes) == len(t.nodes)
    assert {n.tool for n in loaded.nodes.values()} == {n.tool for n in t.nodes.values()}
    assert len(loaded.leaves()) == 2


def test_unknown_node_raises() -> None:
    import pytest

    t = SessionTree()
    with pytest.raises(KeyError):
        t.fork("does-not-exist")
    with pytest.raises(KeyError):
        t.path_to("nope")


def test_treenode_dict_roundtrip() -> None:
    n = TreeNode(
        id="abc",
        parent_id="par",
        step_num=2,
        tool="bash",
        args_summary="cmd=ls",
        error=None,
        label="checkpoint",
    )
    n2 = TreeNode.from_dict(n.to_dict())
    assert n2 == n


def test_render_summary_smoke() -> None:
    from looplet.session_tree import render_summary

    t = SessionTree()
    t.append(step_num=1, tool="read_file", args_summary="file_path=src/a.py")
    t.append(step_num=2, tool="write_file", args_summary="file_path=src/a.py")
    t.append(step_num=3, tool="bash", args_summary="cmd=pytest", error="fail")
    out = render_summary(t)
    assert "Tool frequency" in out
    assert "read_file" in out and "write_file" in out and "bash" in out
    assert "src/a.py`: 2" in out
    assert "Errors" in out and "fail" in out
    assert "Step timeline" in out


def test_render_summary_empty_tree() -> None:
    from looplet.session_tree import render_summary

    t = SessionTree()
    out = render_summary(t)
    assert "0 steps" in out
    assert "(empty)" in out
