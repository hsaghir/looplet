"""Toy cartridge registry: list cartridges in a directory, or pull one
from a git URL.

Run::

    # List local cartridges under <root>:
    python registry.py list /path/to/repo

    # Clone a cartridge subdirectory from a git URL:
    python registry.py pull <git_url> <subpath_in_repo> <dest>
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def cmd_list(root: Path) -> None:
    rows = []
    for ws_json in root.rglob("cartridge.json"):
        if "__pycache__" in ws_json.parts:
            continue
        ws_dir = ws_json.parent
        try:
            meta = json.loads(ws_json.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(
            (meta.get("name", "?"), meta.get("description", "")[:60], ws_dir.relative_to(root))
        )
    rows.sort()
    for name, desc, path in rows:
        print(f"{name:<24} {str(path):<40} {desc}")


def cmd_pull(git_url: str, subpath: str, dest: Path) -> None:
    if dest.exists():
        raise SystemExit(f"destination already exists: {dest}")
    with tempfile.TemporaryDirectory(prefix="ws_pull_") as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, str(tmp_path / "repo")],
            check=True,
            capture_output=True,
        )
        src = tmp_path / "repo" / subpath
        if not (src / "cartridge.json").is_file():
            raise SystemExit(f"no cartridge.json in {subpath}")
        shutil.copytree(src, dest)
        for cache in dest.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
    print(f"pulled cartridge into {dest}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: registry.py [list <root> | pull <git_url> <subpath> <dest>]")
    cmd = sys.argv[1]
    if cmd == "list" and len(sys.argv) == 3:
        cmd_list(Path(sys.argv[2]))
    elif cmd == "pull" and len(sys.argv) == 5:
        cmd_pull(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        raise SystemExit("usage: registry.py [list <root> | pull <git_url> <subpath> <dest>]")


if __name__ == "__main__":
    main()
