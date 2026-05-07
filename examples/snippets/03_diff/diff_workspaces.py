"""Semantic diff between two workspaces.

Walks both directories, classifies each file by its top-level
category (config, prompt, tool, hook, resource, manifest, other),
and prints a one-line summary per changed file. The point is to
show that a workspace diff carries categorical information *in the
path*, not in the diff content.

Run::

    uv run python examples/snippets/03_diff/diff_workspaces.py \
        examples/coder.workspace \
        examples/snippets/01_inheritance/refactorer.workspace
"""

from __future__ import annotations

import sys
from pathlib import Path

CATEGORIES = [
    ("manifest", lambda p: p.name == "workspace.json" and len(p.parts) == 1),
    ("prompt", lambda p: p.parts[0] == "prompts"),
    ("tool", lambda p: p.parts[0] == "tools"),
    ("hook", lambda p: p.parts[0] == "hooks"),
    ("resource", lambda p: p.parts[0] == "resources"),
    ("config", lambda p: p.name == "config.yaml" and len(p.parts) == 1),
]


def categorise(rel: Path) -> str:
    for name, pred in CATEGORIES:
        try:
            if pred(rel):
                return name
        except IndexError:
            continue
    return "other"


def list_files(root: Path) -> dict[Path, bytes]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        files[path.relative_to(root)] = path.read_bytes()
    return files


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: diff_workspaces.py <ws_a> <ws_b>")
    a = Path(sys.argv[1]).resolve()
    b = Path(sys.argv[2]).resolve()
    files_a = list_files(a)
    files_b = list_files(b)
    all_paths = sorted(set(files_a) | set(files_b))

    rows = []
    for rel in all_paths:
        cat = categorise(rel)
        if rel not in files_a:
            rows.append((cat, "added", rel))
        elif rel not in files_b:
            rows.append((cat, "removed", rel))
        elif files_a[rel] != files_b[rel]:
            la = len(files_a[rel].splitlines())
            lb = len(files_b[rel].splitlines())
            rows.append((cat, f"changed ({la}->{lb} lines)", rel))

    if not rows:
        print("workspaces are identical")
        return

    by_cat: dict[str, list[tuple[str, str, Path]]] = {}
    for cat, action, rel in rows:
        by_cat.setdefault(cat, []).append((cat, action, rel))

    for cat, items in sorted(by_cat.items()):
        print(f"# {cat}")
        for _, action, rel in items:
            print(f"  {action:<24} {rel}")
        print()


if __name__ == "__main__":
    main()
