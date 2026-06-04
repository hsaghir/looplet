"""Stdio MCP server for the git_detective_portable cartridge.

Serves all 10 tools that were in-process ``tools/*/execute.py`` closures
in the original ``git_detective`` cartridge — ``repo_overview``,
``contributor_stats``, ``recent_activity``, ``file_hotspots``,
``coupled_files``, ``commit_patterns``, ``directory_structure``,
``file_age_analysis``, ``think``, ``done`` — over the MCP stdio
transport. Moving them out of process is what makes the twin fully
portable: no Python tool body is required by the host.

The target repository is resolved from ``$LOOPLET_PROJECT_ROOT`` (set by
the host), falling back to the server's working directory — the portable
equivalent of the original's INPROCESS ``repo_config`` resource.

``commit_patterns`` reaches the host model through the Model Gateway
(MGP): the loader exports ``LOOPLET_LLM_SOCKET`` and this server connects
to it lazily, so the tool adds its LLM commit-quality assessment — full
parity with the in-process original. When no gateway is present (or no
backend is bound yet) it omits that field, exactly the branch the
original takes when ``ctx.llm is None``. The deterministic git statistics
are preserved in full.

Standard library + the ``git`` executable only.
Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path


class _HostLLM:
    """Minimal stdlib-only client to the host Model Gateway (MGP).

    Connects to ``$LOOPLET_LLM_SOCKET`` (set by the loader) and forwards
    ``generate`` to the host's live LLM backend. ``generate`` raises if no
    gateway/backend is reachable, so callers degrade exactly like the
    in-process original's ``ctx.llm is None`` branch.
    """

    def __init__(self):
        self._sock = None
        self._buf = b""
        self._id = 0
        path = os.environ.get("LOOPLET_LLM_SOCKET")
        if not path or not hasattr(socket, "AF_UNIX"):
            return
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30.0)
            sock.connect(path)
            self._sock = sock
            self._rpc("llm/initialize", {})
        except OSError:
            self._sock = None

    def _readline(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                line, self._buf = self._buf, b""
                return line
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _rpc(self, method, params):
        self._id += 1
        rid = self._id
        self._sock.sendall(
            (json.dumps({"id": rid, "method": method, "params": params}) + "\n").encode("utf-8")
        )
        line = self._readline()
        if not line:
            raise OSError("model gateway closed the connection")
        msg = json.loads(line.decode("utf-8"))
        if msg.get("error"):
            raise RuntimeError(msg["error"].get("message", "model gateway error"))
        return msg.get("result") or {}

    def available(self):
        """True iff the gateway has a live LLM backend bound *right now*.

        Re-checks per call because the host binds the backend lazily at
        run time (``AgentPreset.run(llm)``), which may happen after this
        client connected. Maps to the original's ``ctx.llm is not None``
        guard: when False, callers take their no-LLM degradation branch
        instead of treating absence as an error.
        """
        if self._sock is None:
            return False
        try:
            return bool(self._rpc("llm/initialize", {}).get("ready"))
        except (OSError, RuntimeError):
            return False

    def generate(self, prompt, **kwargs):
        if self._sock is None:
            raise RuntimeError("no host LLM gateway is reachable")
        return str(self._rpc("llm/generate", {"prompt": prompt, "kwargs": kwargs}).get("text", ""))


_HOST_LLM = None
_HOST_LLM_TRIED = False


def _host_llm():
    """Return the host Model Gateway client only when a backend is bound.

    Returns ``None`` when there is no reachable gateway *or* no backend is
    currently bound — i.e. exactly the cases where the in-process original
    sees ``ctx.llm is None`` and degrades.
    """
    global _HOST_LLM, _HOST_LLM_TRIED
    if not _HOST_LLM_TRIED:
        _HOST_LLM_TRIED = True
        client = _HostLLM()
        if client._sock is not None:
            _HOST_LLM = client
    if _HOST_LLM is not None and _HOST_LLM.available():
        return _HOST_LLM
    return None


def _repo_path() -> str:
    return os.environ.get("LOOPLET_PROJECT_ROOT") or os.getcwd()


def _git(repo, *args, max_lines=200):
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
    except Exception as e:  # noqa: BLE001 - faithful to the original helper
        return {"output": "", "line_count": 0, "error": str(e)}


def repo_overview():
    repo = _repo_path()
    name = _git(repo, "rev-parse", "--show-toplevel")
    branch = _git(repo, "branch", "--show-current")
    count = _git(repo, "rev-list", "--count", "HEAD")
    first = _git(repo, "log", "--reverse", "--format=%ai", "-1")
    last = _git(repo, "log", "--format=%ai", "-1")
    remotes = _git(repo, "remote", "-v")
    return {
        "repo_name": Path(name["output"].strip()).name if name["output"] else "unknown",
        "branch": branch["output"].strip(),
        "total_commits": int(count["output"].strip()) if count["output"].strip().isdigit() else 0,
        "first_commit": first["output"].strip(),
        "last_commit": last["output"].strip(),
        "remotes": remotes["output"][:200],
    }


def contributor_stats():
    repo = _repo_path()
    result = _git(repo, "shortlog", "-sne", "HEAD", max_lines=50)
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


def recent_activity(days="30"):
    repo = _repo_path()
    d = int(days) if str(days).isdigit() else 30
    result = _git(repo, "log", f"--since={d} days ago", "--format=%H|%an|%ai|%s", max_lines=100)
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


def file_hotspots(top_n="15"):
    repo = _repo_path()
    n = int(top_n) if str(top_n).isdigit() else 15
    result = _git(repo, "log", "--format=", "--name-only", "--diff-filter=AMRD", max_lines=5000)
    lines = result["output"].strip().split("\n") if result["output"] else []
    counter = Counter(line.strip() for line in lines if line.strip())
    hotspots = [{"file": f, "changes": c} for f, c in counter.most_common(n)]
    return {"total_unique_files": len(counter), "hotspots": hotspots}


def coupled_files(min_coupling="3"):
    repo = _repo_path()
    threshold = int(min_coupling) if str(min_coupling).isdigit() else 3
    commit_count = _git(repo, "rev-list", "--count", "HEAD")
    total = int(commit_count["output"].strip()) if commit_count["output"].strip().isdigit() else 100
    if total < 100 and threshold > 2:
        threshold = 2
    result = _git(repo, "log", "--format=COMMIT_SEP", "--name-only", max_lines=3000)
    commits = result["output"].split("COMMIT_SEP")
    pair_counts = Counter()
    for commit in commits:
        files = [f.strip() for f in commit.strip().split("\n") if f.strip()]
        files = files[:20]
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


def commit_patterns():
    repo = _repo_path()
    result = _git(repo, "log", "--format=%s", "-200", max_lines=200)
    messages = [line.strip() for line in result["output"].split("\n") if line.strip()]
    conventional = sum(
        1
        for m in messages
        if any(m.startswith(p) for p in ["feat", "fix", "docs", "refactor", "test", "chore", "ci"])
    )
    merge_commits = sum(1 for m in messages if m.lower().startswith("merge"))
    wip_commits = sum(1 for m in messages if "wip" in m.lower() or "work in progress" in m.lower())
    fixup = sum(1 for m in messages if m.startswith("fixup!") or m.startswith("squash!"))
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

    # Use the host LLM (via the Model Gateway) to assess commit quality if
    # reachable — same branch the in-process original takes with ctx.llm.
    llm = _host_llm()
    if llm is not None:
        try:
            assessment = llm.generate(
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
        except Exception:
            pass

    return stats


def directory_structure():
    repo = _repo_path()
    result = _git(repo, "ls-tree", "--name-only", "HEAD", max_lines=100)
    items = [line.strip() for line in result["output"].split("\n") if line.strip()]
    dirs = []
    files = []
    for item in items:
        full = os.path.join(repo, item)
        if os.path.isdir(full):
            try:
                count = sum(1 for _ in Path(full).rglob("*") if _.is_file())
            except Exception:  # noqa: BLE001
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


def file_age_analysis():
    repo = _repo_path()
    result = _git(
        repo,
        "log",
        "--format=%ai",
        "--diff-filter=M",
        "--name-only",
        "-500",
        max_lines=2000,
    )
    lines = result["output"].split("\n")
    file_last_modified = {}
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
    sorted_files = sorted(file_last_modified.items(), key=lambda x: x[1])
    stale = sorted_files[:10]
    fresh = sorted_files[-10:]
    return {
        "total_tracked_files": len(file_last_modified),
        "stalest_files": [{"file": f, "last_modified": d} for f, d in stale],
        "freshest_files": [{"file": f, "last_modified": d} for f, d in reversed(fresh)],
    }


TOOLS = [
    {
        "name": "repo_overview",
        "description": "Get basic repo info: name, branch, total commits, age, and remotes.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "contributor_stats",
        "description": "Get contributor stats and bus factor.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "recent_activity",
        "description": "Get commit activity for the last N days.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "string",
                    "description": 'Number of days back (e.g. "30").',
                }
            },
        },
    },
    {
        "name": "file_hotspots",
        "description": "Find most frequently changed files (churn hotspots).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "string",
                    "description": 'How many top files (e.g. "15").',
                }
            },
        },
    },
    {
        "name": "coupled_files",
        "description": "Find files that often change together (co-change coupling).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_coupling": {
                    "type": "string",
                    "description": 'Minimum coupling count (e.g. "3").',
                }
            },
        },
    },
    {
        "name": "commit_patterns",
        "description": "Analyze commit message patterns, conventions, and quality, with an "
        "LLM commit-quality assessment via the host Model Gateway (degrades to the "
        "deterministic stats when no host LLM is reachable).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "directory_structure",
        "description": "Get the top-level directory structure with file counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "file_age_analysis",
        "description": "Find oldest unchanged files and newest additions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "think",
        "description": "Analytical reasoning step. No side effects.",
        "inputSchema": {
            "type": "object",
            "properties": {"thought": {"type": "string", "description": "Brief note."}},
            "required": ["thought"],
        },
    },
    {
        "name": "done",
        "description": "Signal completion with the final report.",
        "inputSchema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Codebase health report."}},
            "required": ["summary"],
        },
    },
]


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _content(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


def _dispatch(name, args):
    if name == "repo_overview":
        return repo_overview()
    if name == "contributor_stats":
        return contributor_stats()
    if name == "recent_activity":
        return recent_activity(args.get("days", "30"))
    if name == "file_hotspots":
        return file_hotspots(args.get("top_n", "15"))
    if name == "coupled_files":
        return coupled_files(args.get("min_coupling", "3"))
    if name == "commit_patterns":
        return commit_patterns()
    if name == "directory_structure":
        return directory_structure()
    if name == "file_age_analysis":
        return file_age_analysis()
    if name == "think":
        return {"thought": args.get("thought"), "noted": True}
    if name == "done":
        return {"status": "completed", "summary": args.get("summary")}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        msg_id = req.get("id")
        if method == "initialize":
            respond(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "git-detective-tools", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            result = _dispatch(name, args)
            if result is None:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
            else:
                respond(msg_id, _content(result))
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
