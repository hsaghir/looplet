"""Toy admission policy for workspaces.

Three rules:
  1. If a `bash` tool is present, a hook with `Permission` in its
     class_name must also be present.
  2. The system prompt must not contain any forbidden phrase.
  3. `max_steps` must be at most ``MAX_ALLOWED_STEPS``.

Exits 0 on pass, 1 on policy violation. Designed to fit in a CI
pipeline alongside lint and tests.

Run::

    python admit.py <workspace_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN_PHRASES = ["ignore previous instructions", "do anything you want"]
MAX_ALLOWED_STEPS = 50


def violations(ws: Path) -> list[str]:
    out: list[str] = []

    has_bash = (ws / "tools" / "bash").is_dir()
    has_perm_hook = False
    hooks_dir = ws / "hooks"
    if hooks_dir.is_dir():
        for hd in hooks_dir.iterdir():
            cfg = hd / "config.yaml"
            if cfg.is_file() and "Permission" in cfg.read_text():
                has_perm_hook = True
                break
    if has_bash and not has_perm_hook:
        out.append("bash tool present without a hook with 'Permission' in class_name")

    prompt_path = ws / "prompts" / "system.md"
    if prompt_path.is_file():
        text = prompt_path.read_text().lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                out.append(f"forbidden phrase in system prompt: {phrase!r}")

    cfg = ws / "config.yaml"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("max_steps:"):
                _, _, val = line.partition(":")
                try:
                    n = int(val.strip())
                except ValueError:
                    continue
                if n > MAX_ALLOWED_STEPS:
                    out.append(f"max_steps={n} exceeds policy max ({MAX_ALLOWED_STEPS})")

    return out


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: admit.py <workspace_dir>")
    ws = Path(sys.argv[1]).resolve()
    if not (ws / "workspace.json").is_file():
        raise SystemExit(f"not a workspace: {ws}")
    issues = violations(ws)
    if not issues:
        print(f"OK: {ws.name} passes admission policy")
        return
    print(f"DENIED: {ws.name} violates admission policy:")
    for v in issues:
        print(f"  - {v}")
    sys.exit(1)


if __name__ == "__main__":
    main()
