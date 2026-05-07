"""Static workspace ablation driver.

Mutates a copy of examples/coder.workspace 5 ways and prints a table
of static properties. No LLM calls; runs in <1 second. The point is
to show the *mechanism* of an ablation matrix, not to measure
performance.

Run::

    uv run python examples/snippets/02_ablation/ablate.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from looplet import workspace_to_preset

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "examples" / "coder.workspace"


def _baseline(_ws: Path) -> None:
    pass


def _remove_glob(ws: Path) -> None:
    shutil.rmtree(ws / "tools" / "glob")


def _remove_bash(ws: Path) -> None:
    shutil.rmtree(ws / "tools" / "bash")


def _low_budget(ws: Path) -> None:
    cfg = ws / "config.yaml"
    cfg.write_text(cfg.read_text().replace("max_steps: 20", "max_steps: 4"))


def _terse_prompt(ws: Path) -> None:
    (ws / "prompts" / "system.md").write_text(
        "You are a coding agent. Run tests until they pass. Use done() only after.\n"
    )


ABLATIONS = [
    ("baseline", _baseline),
    ("no_glob", _remove_glob),
    ("no_bash", _remove_bash),
    ("budget=4", _low_budget),
    ("terse_prompt", _terse_prompt),
]


def _measure(ws_path: Path) -> dict:
    preset = workspace_to_preset(str(ws_path), runtime={"workspace": "."})
    tools = sorted(t["name"] for t in preset.tools.introspect()["tools"])
    return {
        "n_tools": len(tools),
        "max_steps": preset.config.max_steps,
        "prompt_len": len(preset.config.system_prompt),
    }


def main() -> None:
    rows = []
    for name, mut in ABLATIONS:
        with tempfile.TemporaryDirectory(prefix=f"ablate_{name}_") as tmp:
            ws = Path(tmp) / "ws"
            shutil.copytree(BASELINE, ws)
            for cache in ws.rglob("__pycache__"):
                shutil.rmtree(cache, ignore_errors=True)
            mut(ws)
            rows.append((name, _measure(ws)))

    print(f"| {'ablation':<14} | {'n_tools':>7} | {'max_steps':>9} | {'prompt_len':>10} |")
    print(f"|{'-' * 16}|{'-' * 9}|{'-' * 11}|{'-' * 12}|")
    for name, m in rows:
        print(f"| {name:<14} | {m['n_tools']:>7} | {m['max_steps']:>9} | {m['prompt_len']:>10} |")


if __name__ == "__main__":
    main()
