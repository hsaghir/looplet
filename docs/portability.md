# Cross-runtime portability

A looplet cartridge is just a directory of declarative files plus
(optionally) some Python. The **portability analyser** answers one
question about any cartridge:

> Can this agent package run on a non-Python loader (Rust / Go /
> TypeScript), and if not, *exactly which components* pin it to a Python
> host?

```python
from looplet.cartridge import analyse_cartridge

report = analyse_cartridge("examples/dep_doctor_portable.cartridge")
print(report.render())
print(report.profile)   # "portable" | "python-host"
print(report.blockers)  # the INPROCESS components, if any
```

The analyser is **static**: it reads the cartridge directory and its
`config.yaml` / `runtime.yaml` rather than importing any Python bodies,
so it can grade cartridges whose code can't even be imported in the
current environment.

## The four tiers

Every component is classified into one of four tiers (best → worst for
portability). A cartridge is in the **portable profile** when it has
*zero* `INPROCESS` components; otherwise it is **python-host** and the
`INPROCESS` components are the exact blockers.

| Tier | Symbol | Meaning | Blocker? |
|------|--------|---------|----------|
| `PROTOCOL` | ● | Pure data or out-of-process protocol - `config.yaml`, `prompts/`, `mcp_servers:` tools, `kind: lep` hooks. Runs on any conforming loader with no shared code. | no |
| `STDLIB` | ◐ | Declarative reference to a looplet-shipped archetype (`builtin_tools:` / `builtin_hooks:`). No Python body in the cartridge. | no |
| `RUNTIME` | ◈ | A `resources/*.py` whose only job is to wrap a host-provided *service* (compaction, the skill manager) via a looplet factory like `default_compact_service`. The service is a host responsibility every loader ships its own equivalent of. | no |
| `INPROCESS` | ○ | A Python body or author-owned shared object that pins the cartridge to a Python host - `tools/<n>/execute.py`, single-file `tools/<n>.py`, `hooks/<n>/` class hooks, author-owned `resources/*.py` (`@ref` shared mutable state). | **yes** |

`extends:` is resolved transitively: a child cartridge inherits its
parent's components (and therefore its parent's blockers), tagged
`(inherited)` in the report.

## The three protocol surfaces

A fully-portable cartridge moves every author-owned capability onto one
of looplet's three out-of-process protocol surfaces:

| Surface | Carries | Transport | Cardinality |
|---------|---------|-----------|-------------|
| **MCP** | tools | stdio JSON-RPC | 1 server : N tools |
| **LEP** (Loop Effect Protocol) | hooks (permission / done policies) | stdio JSON-RPC | 1 server : 1 hook |
| **SSP** (State Service Protocol) | shared mutable state | `AF_UNIX` SOCK_STREAM | 1 service : N clients |

The host ships only a declared **view** to a hook; the hook returns a
decision. The host spawns an MCP server and calls its tools. Neither
requires the host to execute any author Python.

## Portable example cartridges

Each non-trivial example ships a `*_portable` twin that is byte-for-byte
behaviour-faithful to its in-process original, but classifies as
`portable`. Every twin has an end-to-end *cross-process* dogfood test in
`tests/test_example_*_portable_cartridge.py`.

| Portable twin | Original | What moved | Notes |
|---------------|----------|------------|-------|
| `hello_portable` | `hello` | greet/done → MCP | smallest demo |
| `mcp_demo_portable` | `mcp_demo` | add + done → MCP, CalcGuard → LEP | both tools served by one MCP server |
| `planner_portable` | `planner` | done → MCP; nested `planner_child` twin | `subagent` builtin tool + LEP guard |
| `skillful_analyst_portable` | `skillful_analyst` | done/read_text/write_text → MCP, WriteScopeGuard → LEP | skill system via `search_skills`/`activate_skill` builtins + RUNTIME `skill_manager` |
| `dep_doctor_portable` | `dep_doctor` | 7 audit tools → MCP, RegistryGuard → LEP | `find_alternatives` degrades to empty (no host LLM in the MCP subprocess) |
| `threat_intel_portable` | `threat_intel` | 7 tools → MCP, FeedAllowlistGuard → LEP | regex IOC extraction preserved; LLM severity/risk degrade gracefully |
| `git_detective_portable` | `git_detective` | 10 tools → MCP, CouplingGuard → LEP | INPROCESS `repo_config` replaced by `$LOOPLET_PROJECT_ROOT` resolution; the server only needs `git` |

### The mechanical port recipe

Porting an in-process cartridge to a portable twin is a fixed procedure:

1. **Fold every `tools/<n>/execute.py` body into one `_mcp/tools_server.py`** -
   a stdlib-only stdio MCP server. Vendor any data tables (package DBs,
   threat feeds) into the server. Wire it in `config.yaml`:

   ```yaml
   mcp_servers:
     my_tools:
       command: "python3 ${runtime.cartridge_root}/_mcp/tools_server.py"
   ```

2. **Keep `kind: lep` hooks verbatim** - they are already PROTOCOL-tier.

3. **Keep `builtin_tools:` / `builtin_hooks:`** - already STDLIB-tier.

4. **Keep `resources/compact_service.py` (and similar)** - already
   RUNTIME-tier (host service), not a blocker.

5. **Replace author-owned INPROCESS resources** - e.g.
   `git_detective`'s `repo_config` (a shared path) becomes a
   `$LOOPLET_PROJECT_ROOT` env lookup inside the server; genuine shared
   *mutable* state would move to an SSP service instead.

6. **Anchor relative paths to the host root** - the MCP subprocess
   resolves relative paths against `os.getcwd()` (the host project
   root), the portable equivalent of `resolve_project_root(runtime)`.

!!! note "LLM-backed tools degrade, they don't break"
    A tool that calls the host LLM (`ctx.llm`) can't reach it from a
    separate MCP process. The faithful port returns the same fallback
    the original takes when `ctx.llm is None` (e.g.
    `find_alternatives` → empty list, `assess_risk` → severity-only
    summary). Deterministic logic (regex extraction, git statistics) is
    preserved in full. True host-LLM access from a server would require
    MCP `sampling/createMessage` callbacks - out of scope for v1.x.

## Inherently python-host cartridges

Two shipped cartridges are **deliberately** python-host. They are not
ported; their whole purpose is to host and execute Python.

### `coder`

A general software-engineering agent: ~16 substantial tools
(`bash`, `edit_file`, `grep`, `subagent`, `worktree`, …), 5 class
hooks, and 7 resources. Several resources (`file_cache`,
`workspace_config`, `eval_*`) are **genuine** `INPROCESS` shared
mutable state, and the tools execute arbitrary code in the host
environment. The mechanical recipe above *would* apply tool-by-tool
(bodies → an MCP server, class hooks → LEP, shared state → SSP), but the
result would still need a host that can run a shell and edit a live
workspace - the very thing that makes it useful. Porting it buys no
portability; it stays python-host by design.

### `agent_factory`

`agent_factory` does `extends: ../coder.cartridge`, so the analyser
reports its blockers as **its own** (`validate_workspace`) **plus all of
`coder`'s** (tagged `(inherited)`). Its one bespoke tool,
`validate_workspace`, exists to *import and validate a Python workspace*
and is python-host by definition. Like `coder`, it is documented as
inherently python-host rather than ported.

```python
report = analyse_cartridge("examples/agent_factory.cartridge")
assert report.profile == "python-host"
# blockers = validate_workspace + every coder component, via extends
```
