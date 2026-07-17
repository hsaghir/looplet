"""Cold-import benchmark - measure startup cost of four frameworks.

Runs each import in a fresh Python subprocess so we measure *cold*
import time (no warm cache from previous runs). Reports median of N
runs to tame noise.

Usage::

    python scripts/bench_cold_import.py              # default 9 runs
    python scripts/bench_cold_import.py --runs 21    # smoother
    python scripts/bench_cold_import.py --markdown   # emit markdown

This is not an apples-to-apples performance benchmark - each framework
does different amounts of work on import. It *is* apples-to-apples for
"how long until I can write the first line of my agent".
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    import_stmt: str
    version_stmt: str


TARGETS: tuple[Target, ...] = (
    Target(
        name="looplet",
        import_stmt="import looplet",
        version_stmt="import importlib.metadata as m; print(m.version('looplet'))",
    ),
    Target(
        name="claude-agent-sdk",
        import_stmt="import claude_agent_sdk",
        version_stmt="import importlib.metadata as m; print(m.version('claude-agent-sdk'))",
    ),
    Target(
        name="pydantic-ai",
        import_stmt="import pydantic_ai",
        version_stmt="import importlib.metadata as m; print(m.version('pydantic-ai'))",
    ),
    Target(
        name="langgraph",
        import_stmt="import langgraph, langgraph.graph",
        version_stmt="import importlib.metadata as m; print(m.version('langgraph'))",
    ),
    Target(
        name="strands-agents",
        import_stmt="import strands",
        version_stmt="import importlib.metadata as m; print(m.version('strands-agents'))",
    ),
)


def time_one_import(py: str, stmt: str) -> float:
    """Launch a fresh Python subprocess, time the import, return seconds."""
    script = f"import time; t0 = time.perf_counter(); {stmt}; print(time.perf_counter() - t0)"
    result = subprocess.run(
        [py, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    # Timing is self-reported by the subprocess so wall-clock subprocess
    # startup overhead is not included.
    return float(result.stdout.strip())


def get_version(py: str, stmt: str) -> str:
    result = subprocess.run([py, "-c", stmt], capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def bench(py: str, target: Target, runs: int) -> list[float]:
    samples: list[float] = []
    for _ in range(runs):
        try:
            samples.append(time_one_import(py, target.import_stmt))
        except subprocess.CalledProcessError as exc:
            print(
                f"  ! {target.name} import failed: {exc.stderr[:200]}",
                file=sys.stderr,
            )
            return []
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=9)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows: list[tuple[str, str, float, float]] = []
    for target in TARGETS:
        version = get_version(args.python, target.version_stmt)
        samples = bench(args.python, target, args.runs)
        if not samples:
            continue
        median = statistics.median(samples) * 1000.0
        stdev = statistics.stdev(samples) * 1000.0 if len(samples) > 1 else 0.0
        rows.append((target.name, version, median, stdev))
        print(
            f"{target.name:<20} {version:<10}  "
            f"median={median:6.1f} ms   stdev={stdev:5.1f} ms   "
            f"n={len(samples)}",
            flush=True,
        )

    rows.sort(key=lambda r: r[2])
    if args.markdown:
        print()
        print(
            f"> Python {sys.version_info.major}.{sys.version_info.minor}, "
            f"median of {args.runs} cold-start subprocess runs"
        )
        print()
        print("| Framework | Version | Median cold import | vs looplet |")
        print("| --- | --- | ---: | ---: |")
        looplet_med = next((r[2] for r in rows if r[0] == "looplet"), None)
        for name, version, median, _ in rows:
            ratio = f"{median / looplet_med:.1f}×" if looplet_med and name != "looplet" else " - "
            print(f"| `{name}` | {version} | **{median:.0f} ms** | {ratio} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
