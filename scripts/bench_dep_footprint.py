"""Dependency-footprint benchmark - count transitive deps per framework.

Creates a fresh venv per framework, installs the package, and counts
how many third-party packages ended up installed. A proxy for "how
much am I pulling into my environment?"

Usage::

    python scripts/bench_dep_footprint.py               # default 5 targets
    python scripts/bench_dep_footprint.py --markdown    # emit markdown
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

TARGETS = (
    ("looplet", "looplet"),
    ("looplet[all]", "looplet[all]"),
    ("claude-agent-sdk", "claude-agent-sdk"),
    ("pydantic-ai", "pydantic-ai"),
    ("langgraph", "langgraph"),
    ("strands-agents", "strands-agents"),
)


def count_deps(package: str, extras: str) -> int:
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_{package.replace('/', '_')}_"))
    try:
        # Create venv with uv (fast).
        subprocess.run(
            ["uv", "venv", str(tmp / "venv"), "--python", "3.11", "-q"],
            check=True,
            capture_output=True,
        )
        py = tmp / "venv" / "bin" / "python"
        subprocess.run(
            ["uv", "pip", "install", "--python", str(py), "-q", extras],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["uv", "pip", "list", "--python", str(py), "--format=freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Count lines; exclude pip/setuptools/wheel.
        skip = {"pip", "setuptools", "wheel"}
        count = 0
        for line in result.stdout.splitlines():
            name = line.split("==", 1)[0].strip().lower()
            if name and name not in skip:
                count += 1
        return count
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows: list[tuple[str, int]] = []
    for display, install_spec in TARGETS:
        print(f"Measuring {display}...", flush=True)
        n = count_deps(display, install_spec)
        rows.append((display, n))
        print(f"  {display}: {n} packages")

    rows.sort(key=lambda r: r[1])
    if args.markdown:
        print()
        print("| Install | Packages installed |")
        print("| --- | ---: |")
        for name, n in rows:
            print(f"| `pip install {name}` | **{n}** |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
