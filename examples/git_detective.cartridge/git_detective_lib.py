"""Inlined v1 ``git_detective`` agent code - the ``_git`` helper and
``make_tools(repo_path)`` factory previously hosted at
``examples/git_detective/agent.py``.

The 8 underlying tool implementations are closures inside
``make_tools``; each ``tools/<name>/execute.py`` shim instantiates
the registry on first call (per-process, lazy) and dispatches to the
matching tool.
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path

from looplet.tools import tool, tools_from
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


def make_tools(repo_path: str):
    """Build tool registry with git analysis tools bound to repo_path."""

    @tool(
        description="Get basic repo info: name, branch, total commits, age, and remotes.",
        concurrent_safe=True,
    )
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

    @tool(
        description="Get contributor stats and bus factor.",
        concurrent_safe=True,
    )
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

    @tool(description="Get commit activity for the last N days.", concurrent_safe=True)
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

    @tool(description="Find most frequently changed files.", concurrent_safe=True)
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

    @tool(description="Find files that often change together.", concurrent_safe=True)
    def coupled_files(*, min_coupling: str = "3") -> dict:
        """Find files that always change together (co-change coupling)."""
        threshold = int(min_coupling) if min_coupling.isdigit() else 3
        # Adaptive: for small repos (<100 commits), lower threshold to 2
        commit_count = _git(repo_path, "rev-list", "--count", "HEAD")
        total = (
            int(commit_count["output"].strip()) if commit_count["output"].strip().isdigit() else 100
        )
        if total < 100 and threshold > 2:
            threshold = 2

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

    @tool(description="Analyze commit message patterns, conventions, and quality.")
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

    @tool(
        description="Get the top-level directory structure with file counts.", concurrent_safe=True
    )
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

    @tool(description="Find oldest unchanged files and newest additions.", concurrent_safe=True)
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

    return tools_from(
        [
            repo_overview,
            contributor_stats,
            recent_activity,
            file_hotspots,
            coupled_files,
            commit_patterns,
            directory_structure,
            file_age_analysis,
        ],
        include_think=True,
        include_done=True,
        done_parameters={
            "report": "The complete codebase health report in structured markdown",
        },
    )
