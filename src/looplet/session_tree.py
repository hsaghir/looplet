"""Session-as-tree — branching/forking over a recorded agent trajectory.

Pi treats sessions as a tree: each entry has ``id`` and ``parentId`` so
operators can fork, clone, and replay from any prior point. looplet's
``Step`` stream is naturally linear, but recording each step as a node
under a parent gives the same capability with no change to the loop.

This module provides:

* :class:`TreeNode` — one recorded step (linear by default; gets
  siblings only when a fork is created).
* :class:`SessionTree` — rooted tree with ``fork(node_id)``,
  ``leaves()``, ``path_to(node_id)``.
* :class:`SessionTreeRecorder` — hook that builds the tree from the
  ``POST_TOOL_USE`` lifecycle events emitted by the loop.
* :func:`save_tree` / :func:`load_tree` — JSONL round-trip on disk.

The tree is **descriptive, not prescriptive**: it does not change the
loop. Replaying a path is done by reconstructing prompts/results from
the stored nodes (a cheap form of provenance for "what would have
happened if I'd stopped here and asked a different question").

Usage::

    from looplet.session_tree import SessionTree, SessionTreeRecorder, save_tree

    tree = SessionTree()
    recorder = SessionTreeRecorder(tree)
    for step in composable_loop(llm=llm, hooks=[recorder], ...):
        ...

    save_tree(tree, "trees/run_1.jsonl")

    # Later — fork from step 3 and continue the conversation differently:
    fork_id = tree.fork(tree.path_to_step(3)[-1].id)
    # ...feed that fork_id as parent into a new sub-loop run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from looplet.events import EventPayload, LifecycleEvent

__all__ = [
    "TreeNode",
    "SessionTree",
    "SessionTreeRecorder",
    "save_tree",
    "load_tree",
    "render_summary",
]


def _new_id() -> str:
    return uuid4().hex[:12]


@dataclass
class TreeNode:
    """One step in a session tree.

    ``parent_id`` is None only for the synthetic root. Two nodes
    sharing a parent represent a fork.
    """

    id: str
    parent_id: str | None
    step_num: int
    tool: str
    args_summary: str
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "step_num": self.step_num,
            "tool": self.tool,
            "args_summary": self.args_summary,
            "error": self.error,
            "timestamp": self.timestamp,
            "label": self.label,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TreeNode":
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            step_num=int(d.get("step_num", 0)),
            tool=str(d.get("tool", "")),
            args_summary=str(d.get("args_summary", "")),
            error=d.get("error"),
            timestamp=float(d.get("timestamp", time.time())),
            label=str(d.get("label", "")),
            extra=dict(d.get("extra") or {}),
        )


class SessionTree:
    """Rooted tree of :class:`TreeNode`.

    Maintains a synthetic root node so every recorded step has a
    parent. The first appended step becomes a child of root; each
    subsequent step is a child of the most recently appended node on
    the *active branch*.
    """

    def __init__(self, root_label: str = "root") -> None:
        self.nodes: dict[str, TreeNode] = {}
        self.children: dict[str, list[str]] = {}
        root = TreeNode(
            id=_new_id(),
            parent_id=None,
            step_num=0,
            tool="<root>",
            args_summary="",
            label=root_label,
        )
        self.root_id = root.id
        self.nodes[root.id] = root
        self.children[root.id] = []
        self._active_id: str = root.id

    @property
    def active_id(self) -> str:
        """ID of the most recently appended node on the active branch."""
        return self._active_id

    def append(
        self,
        *,
        step_num: int,
        tool: str,
        args_summary: str,
        error: str | None = None,
        parent_id: str | None = None,
        label: str = "",
        extra: dict[str, Any] | None = None,
    ) -> TreeNode:
        """Add a node as a child of ``parent_id`` (default: active branch tail)."""
        parent = parent_id or self._active_id
        if parent not in self.nodes:
            raise KeyError(f"unknown parent_id: {parent}")
        node = TreeNode(
            id=_new_id(),
            parent_id=parent,
            step_num=step_num,
            tool=tool,
            args_summary=args_summary,
            error=error,
            label=label,
            extra=dict(extra or {}),
        )
        self.nodes[node.id] = node
        self.children.setdefault(parent, []).append(node.id)
        self.children.setdefault(node.id, [])
        self._active_id = node.id
        return node

    def fork(self, node_id: str) -> str:
        """Mark ``node_id`` as the active tip so subsequent appends branch from it.

        Returns the node id (unchanged) for chaining. Use this to
        rewind: ``tree.fork(some_earlier_id)`` then run another loop.
        """
        if node_id not in self.nodes:
            raise KeyError(node_id)
        self._active_id = node_id
        return node_id

    def leaves(self) -> list[TreeNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [self.nodes[nid] for nid, kids in self.children.items() if not kids]

    def path_to(self, node_id: str) -> list[TreeNode]:
        """Return the path from root → node_id (inclusive of both)."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        chain: list[TreeNode] = []
        cur: str | None = node_id
        while cur is not None:
            chain.append(self.nodes[cur])
            cur = self.nodes[cur].parent_id
        chain.reverse()
        return chain

    def path_to_step(self, step_num: int) -> list[TreeNode]:
        """Return the path along the active branch up to ``step_num``."""
        active_path = self.path_to(self._active_id)
        return [n for n in active_path if n.step_num <= step_num]

    def branches(self) -> list[list[TreeNode]]:
        """Return one root→leaf path per leaf (= one path per branch)."""
        return [self.path_to(leaf.id) for leaf in self.leaves()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "active_id": self._active_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionTree":
        tree = cls.__new__(cls)
        tree.nodes = {}
        tree.children = {}
        for nd in d.get("nodes", []):
            node = TreeNode.from_dict(nd)
            tree.nodes[node.id] = node
            tree.children.setdefault(node.id, [])
            if node.parent_id is not None:
                tree.children.setdefault(node.parent_id, []).append(node.id)
        tree.root_id = str(d["root_id"])
        tree._active_id = str(d.get("active_id", tree.root_id))
        return tree


@dataclass
class SessionTreeRecorder:
    """Hook that records POST_TOOL_USE events into a :class:`SessionTree`."""

    tree: SessionTree

    def on_event(self, payload: EventPayload) -> None:
        if payload.event != LifecycleEvent.POST_TOOL_USE:
            return
        tc = payload.tool_call
        tr = payload.tool_result
        if tc is None or tr is None:
            return
        self.tree.append(
            step_num=int(payload.step_num or 0),
            tool=str(getattr(tc, "tool", "") or ""),
            args_summary=str(getattr(tr, "args_summary", "") or ""),
            error=getattr(tr, "error", None),
        )


def save_tree(tree: SessionTree, path: str | Path) -> Path:
    """Write the tree as JSONL: one node per line, header line first."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"_header": True, "root_id": tree.root_id, "active_id": tree.active_id})
            + "\n"
        )
        for node in tree.nodes.values():
            f.write(json.dumps(node.to_dict()) + "\n")
    return p


def load_tree(path: str | Path) -> SessionTree:
    """Inverse of :func:`save_tree`."""
    p = Path(path)
    nodes: list[dict[str, Any]] = []
    header: dict[str, Any] | None = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                nodes.append(obj)
    if header is None:
        raise ValueError(f"missing header line in {p}")
    return SessionTree.from_dict({**header, "nodes": nodes})


def render_summary(tree: SessionTree, *, t_start: float | None = None) -> str:
    """Render a Markdown summary of a recorded :class:`SessionTree`.

    The closest looplet equivalent to Pi's ``/tree`` view: a printable
    audit of which tools the agent used, how many times each file was
    touched, error rate, and a per-step timeline. Designed for stdout
    or for dropping next to a saved trajectory as a human-readable
    companion file.

    Args:
        tree: The tree to summarise.
        t_start: Optional epoch-seconds anchor for the timeline.
            Defaults to the timestamp of the first non-root node so
            timestamps print as ``t+N.Ns`` from the start of the run.

    Returns:
        A Markdown string. Always non-empty (a tree with no recorded
        steps still produces a header + zero-row sections).

    Example::

        from looplet.session_tree import SessionTree, save_tree, render_summary
        # ...run a loop with SessionTreeRecorder(tree)...
        save_tree(tree, "trees/run.jsonl")
        Path("trees/run.md").write_text(render_summary(tree))
    """
    from collections import Counter, defaultdict

    nodes = [n for n in tree.nodes.values() if n.tool != "<root>"]
    nodes.sort(key=lambda n: n.timestamp)
    if t_start is None and nodes:
        t_start = nodes[0].timestamp
    if t_start is None:
        t_start = 0.0

    by_tool: Counter[str] = Counter(n.tool for n in nodes)
    edits_per_file: dict[str, int] = defaultdict(int)
    error_count = sum(1 for n in nodes if n.error)

    file_tools = {"write_file", "edit_file", "multi_edit", "read_file", "write", "edit", "read"}
    for n in nodes:
        if n.tool in file_tools:
            for chunk in (n.args_summary or "").split(","):
                k, _, v = chunk.partition("=")
                if k.strip() in {"file_path", "path"}:
                    edits_per_file[v.strip()] += 1
                    break

    lines: list[str] = []
    lines.append(
        f"# Session tree summary — {len(nodes)} steps, "
        f"{len(tree.branches())} branch(es), {error_count} error(s)"
    )
    lines.append("")
    lines.append("## Tool frequency")
    if by_tool:
        for tool, n in by_tool.most_common():
            lines.append(f"- `{tool}`: {n}")
    else:
        lines.append("_(no steps recorded)_")
    lines.append("")
    lines.append("## File touches")
    if edits_per_file:
        for path, n in sorted(edits_per_file.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{path}`: {n}")
    else:
        lines.append("_(no file-tool calls recorded)_")
    lines.append("")
    if error_count:
        lines.append("## Errors")
        for n in nodes:
            if n.error:
                lines.append(f"- step {n.step_num} `{n.tool}`: {n.error[:120]}")
        lines.append("")
    lines.append("## Step timeline")
    if not nodes:
        lines.append("_(empty)_")
    for n in nodes[:200]:
        rel = n.timestamp - t_start
        marker = "✗" if n.error else "✓"
        args = (n.args_summary or "")[:60]
        lines.append(f"- t+{rel:6.1f}s  {marker} #{n.step_num:02d} {n.tool}({args})")
    if len(nodes) > 200:
        lines.append(f"- … {len(nodes) - 200} more steps elided")
    return "\n".join(lines)
