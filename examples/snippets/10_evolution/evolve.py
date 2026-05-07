"""Toy autonomous evolution loop over a workspace.

For N iterations:
  1. Snapshot the current workspace.
  2. Pick a candidate mutation (deterministic round-robin from a
     small pool — in real systems this would be LLM-proposed).
  3. Apply the mutation in a copy.
  4. Score the candidate with a static eval function.
  5. Keep the mutation if score improved.

This is *deliberately* small: no LLM, no real benchmark, runs in
seconds. The point is to expose the loop's shape -- read trajectory,
propose mutation, apply, evaluate, accept/reject -- so the same loop
can later be wired to an LLM proposer and a real benchmark (see
AHE 2026 for the production-grade version).

Run::

    python evolve.py <seed_workspace> [iterations]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def score_workspace(ws: Path) -> int:
    """Toy quality score: rewards balanced tool counts and a
    medium-length system prompt. Designed so the round-robin
    mutations below produce visible up/down movement."""
    score = 0
    tools_dir = ws / "tools"
    n_tools = sum(1 for d in tools_dir.iterdir() if d.is_dir()) if tools_dir.is_dir() else 0
    if 2 <= n_tools <= 8:
        score += 10
    elif n_tools > 0:
        score += 4
    prompt = ws / "prompts" / "system.md"
    if prompt.is_file():
        n_words = len(prompt.read_text().split())
        if 30 <= n_words <= 200:
            score += 10
        elif n_words > 0:
            score += 3
    cfg = ws / "config.yaml"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("max_steps:"):
                _, _, val = line.partition(":")
                try:
                    n = int(val.strip())
                    if 8 <= n <= 30:
                        score += 6
                    elif n > 0:
                        score += 2
                except ValueError:
                    pass
                break
    return score


# Mutations: each takes a workspace dir and modifies in place.


def mut_pad_prompt(ws: Path) -> str:
    p = ws / "prompts" / "system.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text() if p.is_file() else ""
    p.write_text(text + "\nAlways verify your reasoning by re-reading tool outputs.\n")
    return "padded prompt with verification instruction"


def mut_set_max_steps(ws: Path) -> str:
    cfg = ws / "config.yaml"
    if not cfg.is_file():
        cfg.write_text("max_steps: 12\n")
        return "wrote new config with max_steps=12"
    lines = cfg.read_text().splitlines()
    new = []
    found = False
    for line in lines:
        if line.strip().startswith("max_steps:"):
            new.append("max_steps: 16")
            found = True
        else:
            new.append(line)
    if not found:
        new.append("max_steps: 16")
    cfg.write_text("\n".join(new) + "\n")
    return "set max_steps=16"


def mut_add_think_tool(ws: Path) -> str:
    tdir = ws / "tools" / "think"
    if tdir.exists():
        return "think tool already present (no-op)"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "tool.yaml").write_text(
        "name: think\n"
        "description: |-\n"
        "  Pause and write down your current reasoning. Returns the text.\n"
        "parameters:\n"
        "  text:\n"
        "    type: string\n"
        "    description: The reasoning to record.\n"
    )
    (tdir / "execute.py").write_text(
        'def execute(ctx, *, text: str) -> dict:\n    return {"thought": text}\n'
    )
    return "added a think tool"


MUTATIONS = [mut_pad_prompt, mut_set_max_steps, mut_add_think_tool]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: evolve.py <seed_workspace> [iterations]")
    seed = Path(sys.argv[1]).resolve()
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    workdir = Path(tempfile.mkdtemp(prefix="evolve_"))
    current = workdir / "current"
    shutil.copytree(seed, current)
    for cache in current.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    base_score = score_workspace(current)
    print(f"iter 0  score={base_score}  (baseline)")

    history = [("baseline", base_score, "kept")]
    for i in range(1, iterations + 1):
        candidate = workdir / f"cand_{i}"
        if candidate.exists():
            shutil.rmtree(candidate)
        shutil.copytree(current, candidate)
        mutation = MUTATIONS[(i - 1) % len(MUTATIONS)]
        action_msg = mutation(candidate)
        new_score = score_workspace(candidate)
        verdict = "kept" if new_score >= base_score else "rejected"
        if new_score >= base_score:
            shutil.rmtree(current)
            shutil.move(str(candidate), str(current))
            base_score = new_score
        else:
            shutil.rmtree(candidate)
        history.append((mutation.__name__, new_score, verdict))
        print(
            f"iter {i}  mutation={mutation.__name__:<22} score={new_score:>3}  {verdict}  ({action_msg})"
        )

    print()
    print(f"final score: {base_score}")
    print(f"final workspace: {current}")
    print(
        f"({iterations} iterations, {sum(1 for _, _, v in history if v == 'kept') - 1} accepted mutations)"
    )


if __name__ == "__main__":
    main()
