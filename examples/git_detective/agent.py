#!/usr/bin/env python3
"""Git History Detective — codebase health report from commit history.

Point it at any git repo. The agent analyzes commit patterns, identifies
the bus factor, finds coupled files, spots hotspots, and produces a
structured codebase health report.

Not a coding agent — it reads history, the LLM does the analysis.

Usage:
    # Analyze the current repo:
    python examples/git_detective/agent.py

    # Analyze a specific repo:
    python examples/git_detective/agent.py /path/to/repo

    # With Ollama:
    OPENAI_BASE_URL=http://localhost:11434/v1 python examples/git_detective/agent.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    OpenAIBackend,
    StaticMemorySource,
    StreamingHook,
    ToolSpec,
    TrajectoryRecorder,
    composable_loop,
    register_done_tool,
)
from looplet.compact import PruneToolResults, TruncateCompact, compact_chain
from looplet.limits import PerToolLimitHook
from looplet.provenance import RecordingLLMBackend
from looplet.resilient import ResilientBackend
from looplet.session import SessionLog
from looplet.stagnation import StagnationHook, tool_call_fingerprint
from looplet.streaming import CallbackEmitter
from looplet.tools import register_think_tool
from looplet.types import ToolContext


def _git(repo: str, *args: str, max_lines: int = 200) -> dict:
    """Run a git command and return structured output."""
    try:
        result = subprocess.run(
            ["git", "-C", repo, "--no-pager"] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        truncated = len(lines) > max_lines
        return {
            "output": "\n".join(lines[:max_lines]),
            "line_count": len(lines),
            "truncated": truncated,
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }
    except Exception as e:
        return {"output": "", "line_count": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════


def make_tools(repo_path: str) -> BaseToolRegistry:
    """Build tool registry with git analysis tools bound to repo_path."""

    def repo_overview() -> dict:
        """Get basic repo info: name, branch, total commits, age."""
        name = _git(repo_path, "rev-parse", "--show-toplevel")
        branch = _git(repo_path, "branch", "--show-current")
        count = _git(repo_path, "rev-list", "--count", "HEAD")
        first = _git(repo_path, "log", "--reverse", "--format=%ai", "-1")
        last = _git(repo_path, "log", "--format=%ai", "-1")
        remotes = _git(repo_path, "remote", "-v")

        return {
            "repo_name": Path(name["output"].strip()).name if name["output"] else "unknown",
            "branch": branch["output"].strip(),
            "total_commits": int(count["output"].strip())
            if count["output"].strip().isdigit()
            else 0,
            "first_commit": first["output"].strip(),
            "last_commit": last["output"].strip(),
            "remotes": remotes["output"][:200],
        }

    def contributor_stats() -> dict:
        """Get contributor stats: who committed how much."""
        result = _git(repo_path, "shortlog", "-sne", "HEAD", max_lines=50)
        lines = result["output"].strip().split("\n") if result["output"] else []
        contributors = []
        total = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                count = int(parts[0].strip())
                name = parts[1].strip()
                contributors.append({"name": name, "commits": count})
                total += count

        # Bus factor: how many people account for 80% of commits
        bus_factor = 0
        cumulative = 0
        for c in contributors:
            cumulative += c["commits"]
            bus_factor += 1
            if cumulative >= total * 0.8:
                break

        return {
            "contributor_count": len(contributors),
            "top_contributors": contributors[:10],
            "total_commits": total,
            "bus_factor": bus_factor,
            "bus_factor_names": [c["name"] for c in contributors[:bus_factor]],
        }

    def recent_activity(*, days: str = "30") -> dict:
        """Get commit activity for the last N days."""
        d = int(days) if days.isdigit() else 30
        result = _git(
            repo_path, "log", f"--since={d} days ago", "--format=%H|%an|%ai|%s", max_lines=100
        )
        lines = result["output"].strip().split("\n") if result["output"] else []
        commits = []
        authors = Counter()
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append(
                    {
                        "hash": parts[0][:8],
                        "author": parts[1],
                        "date": parts[2][:10],
                        "message": parts[3][:80],
                    }
                )
                authors[parts[1]] += 1

        return {
            "period_days": d,
            "commit_count": len(commits),
            "active_authors": dict(authors.most_common(10)),
            "recent_commits": commits[:15],
        }

    def file_hotspots(*, top_n: str = "15") -> dict:
        """Find most frequently changed files (churn hotspots)."""
        n = int(top_n) if top_n.isdigit() else 15
        result = _git(
            repo_path, "log", "--format=", "--name-only", "--diff-filter=AMRD", max_lines=5000
        )
        lines = result["output"].strip().split("\n") if result["output"] else []
        counter = Counter(l.strip() for l in lines if l.strip())
        hotspots = [{"file": f, "changes": c} for f, c in counter.most_common(n)]
        return {
            "total_unique_files": len(counter),
            "hotspots": hotspots,
        }

    def coupled_files(*, min_coupling: str = "5") -> dict:
        """Find files that always change together (co-change coupling)."""
        threshold = int(min_coupling) if min_coupling.isdigit() else 5
        result = _git(repo_path, "log", "--format=COMMIT_SEP", "--name-only", max_lines=3000)
        commits = result["output"].split("COMMIT_SEP")
        pair_counts: Counter = Counter()
        for commit in commits:
            files = [f.strip() for f in commit.strip().split("\n") if f.strip()]
            files = files[:20]  # cap per commit
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pair = tuple(sorted([files[i], files[j]]))
                    pair_counts[pair] += 1

        coupled = [
            {"file_a": p[0], "file_b": p[1], "co_changes": c}
            for p, c in pair_counts.most_common(15)
            if c >= threshold
        ]
        return {
            "coupling_threshold": threshold,
            "coupled_pairs": coupled[:10],
            "total_pairs_found": len(coupled),
        }

    def commit_patterns(*, ctx: ToolContext) -> dict:
        """Analyze commit message patterns and conventions."""
        result = _git(repo_path, "log", "--format=%s", "-200", max_lines=200)
        messages = [l.strip() for l in result["output"].split("\n") if l.strip()]

        # Pattern detection
        conventional = sum(
            1
            for m in messages
            if any(
                m.startswith(p) for p in ["feat", "fix", "docs", "refactor", "test", "chore", "ci"]
            )
        )
        merge_commits = sum(1 for m in messages if m.lower().startswith("merge"))
        wip_commits = sum(
            1 for m in messages if "wip" in m.lower() or "work in progress" in m.lower()
        )
        fixup = sum(1 for m in messages if m.startswith("fixup!") or m.startswith("squash!"))

        # Average message length
        avg_len = sum(len(m) for m in messages) / max(len(messages), 1)

        stats = {
            "total_analyzed": len(messages),
            "conventional_commits_pct": round(conventional / max(len(messages), 1) * 100, 1),
            "merge_commits": merge_commits,
            "wip_commits": wip_commits,
            "fixup_commits": fixup,
            "avg_message_length": round(avg_len, 1),
            "sample_messages": messages[:8],
        }

        # Use ctx.llm to assess commit quality
        if ctx.llm is not None:
            try:
                assessment = ctx.llm.generate(
                    "Analyze these git commit messages and rate the commit discipline "
                    "on a scale of 1-10. Consider: conventional commits usage, "
                    "descriptiveness, consistency, and whether they tell a coherent story.\n\n"
                    "Messages:\n"
                    + "\n".join(f"- {m}" for m in messages[:15])
                    + f"\n\nStats: {conventional}% conventional, avg length {avg_len:.0f} chars, "
                    f"{wip_commits} WIP, {fixup} fixup\n\n"
                    f"Respond with: SCORE: N/10 followed by 2-3 sentence assessment.",
                    max_tokens=150,
                )
                stats["commit_quality_assessment"] = assessment.strip()
                ctx.warn("Used LLM to assess commit quality")
            except Exception:
                pass

        return stats

    def directory_structure() -> dict:
        """Get the top-level directory structure with file counts."""
        result = _git(repo_path, "ls-tree", "--name-only", "HEAD", max_lines=100)
        items = [l.strip() for l in result["output"].split("\n") if l.strip()]

        dirs = []
        files = []
        for item in items:
            full = os.path.join(repo_path, item)
            if os.path.isdir(full):
                # Count files in directory
                try:
                    count = sum(1 for _ in Path(full).rglob("*") if _.is_file())
                except Exception:
                    count = 0
                dirs.append({"name": item + "/", "file_count": count})
            else:
                files.append(item)

        return {
            "directories": dirs,
            "root_files": files[:20],
            "total_dirs": len(dirs),
            "total_root_files": len(files),
        }

    def file_age_analysis() -> dict:
        """Find oldest unchanged files and newest additions."""
        # Oldest files (by last modification)
        result = _git(
            repo_path,
            "log",
            "--format=%ai",
            "--diff-filter=M",
            "--name-only",
            "-500",
            max_lines=2000,
        )
        lines = result["output"].split("\n")

        file_last_modified: dict[str, str] = {}
        current_date = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and len(line) > 10:
                current_date = line[:10]
            elif current_date and not line.startswith("commit"):
                if line not in file_last_modified:
                    file_last_modified[line] = current_date

        # Sort by date
        sorted_files = sorted(file_last_modified.items(), key=lambda x: x[1])
        stale = sorted_files[:10]  # oldest
        fresh = sorted_files[-10:]  # newest

        return {
            "total_tracked_files": len(file_last_modified),
            "stalest_files": [{"file": f, "last_modified": d} for f, d in stale],
            "freshest_files": [{"file": f, "last_modified": d} for f, d in reversed(fresh)],
        }

    # ── Register all tools ──────────────────────────────────────

    tools = BaseToolRegistry()
    register_done_tool(
        tools,
        parameters={
            "report": "The complete codebase health report in structured markdown",
        },
    )
    register_think_tool(tools)

    for name, desc, params, fn in [
        (
            "repo_overview",
            "Get basic repo info: name, branch, total commits, age, remotes",
            {},
            repo_overview,
        ),
        (
            "contributor_stats",
            "Get contributor stats and bus factor (how many people account for 80% of commits)",
            {},
            contributor_stats,
        ),
        (
            "recent_activity",
            "Get commit activity for the last N days",
            {"days": "Number of days to look back (default: 30)"},
            recent_activity,
        ),
        (
            "file_hotspots",
            "Find most frequently changed files (churn hotspots)",
            {"top_n": "How many hotspots to return (default: 15)"},
            file_hotspots,
        ),
        (
            "coupled_files",
            "Find files that always change together (co-change coupling)",
            {"min_coupling": "Minimum co-changes to report (default: 5)"},
            coupled_files,
        ),
        (
            "commit_patterns",
            "Analyze commit message patterns, conventions, and quality",
            {},
            commit_patterns,
        ),
        (
            "directory_structure",
            "Get the top-level directory structure with file counts",
            {},
            directory_structure,
        ),
        (
            "file_age_analysis",
            "Find oldest unchanged files (stale) and newest additions (fresh)",
            {},
            file_age_analysis,
        ),
    ]:
        tools.register(
            ToolSpec(
                name=name,
                description=desc,
                parameters=params,
                execute=fn,
                concurrent_safe=True,
            )
        )

    return tools


# ═══════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════


def main():
    # Resolve repo path
    if len(sys.argv) > 1:
        repo_path = os.path.abspath(sys.argv[1])
    else:
        repo_path = os.getcwd()

    # Verify it's a git repo
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"Error: {repo_path} is not a git repository", file=sys.stderr)
        sys.exit(1)

    repo_name = Path(repo_path).name

    # LLM
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "local")
    model = os.environ.get("OPENAI_MODEL", "Qwen3.6-27B")

    base_llm = OpenAIBackend(
        base_url=base_url, api_key=api_key, model=model, tool_choice="required"
    )
    llm = ResilientBackend(base_llm, retries=2, timeout_s=120)
    recording = RecordingLLMBackend(llm)

    # Tools
    tools = make_tools(repo_path)

    # Hooks
    stag_hook = StagnationHook(fingerprint=tool_call_fingerprint, threshold=3)
    limit_hook = PerToolLimitHook(default_limit=3)
    events: list = []
    stream_hook = StreamingHook(CallbackEmitter(events.append))

    config = LoopConfig(
        max_steps=12,
        max_tokens=1200,
        temperature=0.3,
        use_native_tools=True,
        system_prompt=(
            f"You are a senior software engineer analyzing the git history of '{repo_name}'.\n\n"
            f"Your task: produce a comprehensive codebase health report.\n\n"
            f"Work systematically:\n"
            f"1. Get repo overview (repo_overview)\n"
            f"2. Analyze contributors and bus factor (contributor_stats)\n"
            f"3. Check recent activity (recent_activity)\n"
            f"4. Find file hotspots (file_hotspots)\n"
            f"5. Detect coupled files (coupled_files)\n"
            f"6. Analyze commit patterns (commit_patterns) — this uses LLM to score quality\n"
            f"7. Call done() with a structured health report including:\n"
            f"   - Health Score (A-F grade)\n"
            f"   - Key Metrics (bus factor, commit frequency, hotspot count)\n"
            f"   - Findings (what's good, what's concerning)\n"
            f"   - Recommendations\n\n"
            f"Use each tool once. Be thorough but efficient."
        ),
        compact_service=compact_chain(
            PruneToolResults(keep_recent=6),
            TruncateCompact(keep_recent=3),
        ),
        memory_sources=[
            StaticMemorySource(
                "## Report Standards\n"
                "- Bus factor < 2 is a risk\n"
                "- Files changed >50 times are hotspots\n"
                "- >20% WIP commits suggests poor discipline\n"
                "- Coupled files suggest tight coupling that may need refactoring\n"
                "- Stale files (>1 year unchanged) may be abandoned or stable\n"
            ),
        ],
    )

    state = DefaultState(max_steps=12)
    session_log = SessionLog()

    # ── Run ──────────────────────────────────────────────────────

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            GIT HISTORY DETECTIVE                            ║")
    print("║            Powered by looplet • 100% local                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Analyzing: {repo_path}")
    print(f"  Model: {model}")
    print()

    with tempfile.TemporaryDirectory() as traj_dir:
        recorder = TrajectoryRecorder(recording_llm=recording, output_dir=traj_dir)

        steps = []
        report = None

        for step in composable_loop(
            llm=recording,
            task={"description": f"Analyze git history of {repo_name}"},
            tools=tools,
            state=state,
            config=config,
            hooks=[stag_hook, limit_hook, stream_hook, recorder],
            session_log=session_log,
        ):
            steps.append(step)
            tool = step.tool_call.tool
            err = step.tool_result.error
            warns = step.tool_result.warnings
            data = step.tool_result.data or {}

            if tool == "done":
                report = data
                print(f"  ✓ Step {step.number}: Report complete!")
            elif tool == "think":
                print(f"  💭 Step {step.number}: thinking...")
            elif err:
                print(f"  ✗ Step {step.number}: {tool} — {str(err)[:60]}")
            else:
                # Show relevant metric from each tool
                preview = ""
                if isinstance(data, dict):
                    if "bus_factor" in data:
                        preview = f"bus_factor={data['bus_factor']}, contributors={data['contributor_count']}"
                    elif "total_commits" in data:
                        preview = (
                            f"{data['total_commits']} commits, branch={data.get('branch', '?')}"
                        )
                    elif "commit_count" in data:
                        preview = (
                            f"{data['commit_count']} commits in {data.get('period_days', '?')} days"
                        )
                    elif "hotspots" in data:
                        top = data["hotspots"][0]["file"] if data["hotspots"] else "none"
                        preview = f"{len(data['hotspots'])} hotspots, top={top}"
                    elif "coupled_pairs" in data:
                        preview = f"{data['total_pairs_found']} coupled pairs found"
                    elif "conventional_commits_pct" in data:
                        preview = f"{data['conventional_commits_pct']}% conventional, avg {data['avg_message_length']:.0f} chars"
                    elif "directories" in data:
                        preview = (
                            f"{data['total_dirs']} dirs, {data['total_root_files']} root files"
                        )
                    elif "stalest_files" in data:
                        preview = f"{data['total_tracked_files']} tracked files"
                    else:
                        preview = str(data)[:60]
                print(f"  → Step {step.number}: {tool} — {preview}")
                if warns:
                    for w in warns:
                        print(f"    ⚠ {w}")

        # ── Output ──────────────────────────────────────────────
        print()
        print("═" * 64)

        if report and isinstance(report, dict):
            report_text = report.get("report", report.get("summary", str(report)))
            print()
            print(report_text)
        else:
            print("\n⚠ Agent did not produce a structured report.")
            if steps:
                last = steps[-1]
                print(f"Last step: {last.tool_call.tool}")
                if last.tool_result.data:
                    print(json.dumps(last.tool_result.data, indent=2, default=str)[:2000])

        print()
        print("═" * 64)

        # ── Stats ───────────────────────────────────────────────
        scoped = [c for c in recording.calls if c.scope]
        print("\n📊 Agent Statistics:")
        print(f"  Steps: {len(steps)}")
        print(f"  LLM calls: {len(recording.calls)} ({len(scoped)} tool-internal)")
        print(f"  Streaming events: {len(events)}")

        traj_path = Path(traj_dir) / "trajectory.json"
        if traj_path.exists():
            print("  Trajectory: saved")
        print()


if __name__ == "__main__":
    main()
