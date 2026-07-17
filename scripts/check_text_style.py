#!/usr/bin/env python3
"""Enforce repository-wide text rules that formatters do not cover."""

from __future__ import annotations

import subprocess
from pathlib import Path

FORBIDDEN_EM_DASH = chr(0x2014)


def repository_files() -> list[Path]:
    """Return tracked and untracked, non-ignored repository paths."""
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    )
    return [Path(value) for value in raw.decode().split("\0") if value]


def main() -> int:
    violations: list[str] = []
    for path in repository_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN_EM_DASH in line:
                violations.append(f"{path}:{line_number}: em dash is not allowed")

    if violations:
        print("\n".join(violations))
        return 1
    print("Text style check passed: no em dashes in tracked UTF-8 files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
