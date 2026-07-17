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
| --- | --- | --- | --- |
| `PROTOCOL` | ● | Pure data or out-of-process protocol - `config.yaml`, `prompts/`, `mcp_servers:` tools, `kind: lep` hooks. Runs on any conforming loader with no shared code. | no |
| `STDLIB` | ◐ | Declarative reference to a looplet-shipped archetype (`builtin_tools:` / `builtin_hooks:`). No Python body in the cartridge. | no |
| `RUNTIME` | ◈ | A `resources/*.py` whose only job is to wrap a host-provided *service* (compaction, the skill manager) via a looplet factory like `default_compact_service`. The service is a host responsibility every loader ships its own equivalent of. | no |
| `INPROCESS` | ○ | A Python body or author-owned shared object that pins the cartridge to a Python host - `tools/<n>/execute.py`, single-file `tools/<n>.py`, `hooks/<n>/` class hooks, author-owned `resources/*.py` (`@ref` shared mutable state). | **yes** |

`extends:` is resolved transitively: a child cartridge inherits its
parent's components (and therefore its parent's blockers), tagged
`(inherited)` in the report.

## The four protocol surfaces

A fully-portable cartridge moves every author-owned capability onto one
of looplet's four out-of-process protocol surfaces:

| Surface | Carries | Transport | Cardinality |
| --- | --- | --- | --- |
| **MCP** | tools | stdio JSON-RPC | 1 server : N tools |
| **LEP** (Loop Effect Protocol) | hooks (permission / done policies) | stdio JSON-RPC | 1 server : 1 hook |
| **SSP** (State Service Protocol) | shared mutable state | `AF_UNIX` SOCK_STREAM | 1 service : N clients |
| **MGP** (Model Gateway Protocol) | host model access | `AF_UNIX` SOCK_STREAM | 1 gateway : N clients |

The host ships only a declared **view** to a hook; the hook returns a
decision. The host spawns MCP and LEP servers, brokers shared state through
SSP, and exposes the active model through MGP. The loader does not import
author-owned tool, hook, or state code.

## The portable coder reference

`coder_portable.cartridge` is the flagship protocol composition. It moves the
complete 16-tool coding surface behind one MCP server, five policy/cache hooks
behind LEP, the shared `FileCache` behind SSP, and model-backed `web_fetch` and
`subagent` calls through MGP. Compaction remains a host-provided runtime
service. The analyzer reports `portable` with zero `INPROCESS` blockers.

```python
from looplet import bundled_cartridge_path
from looplet.cartridge import analyse_cartridge


report = analyse_cartridge(bundled_cartridge_path("coder_portable"))
assert report.profile == "portable"
assert report.blockers == ()
```

The wheel and sdist include this cartridge as a reference artifact. The agent
factory still extends `coder.cartridge`, the Python-host profile, because it
retains host-side `EvalHook` wiring and dynamic instruction/project memory.
Those behaviors have no portable protocol equivalent yet and are explicitly
omitted from the portable twin.

## Portable example cartridges

The repository ships portable twins for its major examples. They preserve the
documented tool and policy behavior that has a protocol equivalent and
classify as `portable`. Each twin has an end-to-end *cross-process* dogfood
test in `tests/test_example_*_portable_cartridge.py`.

| Portable twin | Original | What moved | Notes |
| --- | --- | --- | --- |
| `coder_portable` | `coder` | 16 tools → MCP, 5 hooks → LEP, FileCache → SSP, model access → MGP | bundled flagship; host eval and dynamic memory intentionally omitted |
| `hello_portable` | `hello` | greet/done → MCP | smallest demo |
| `mcp_demo_portable` | `mcp_demo` | add + done → MCP, CalcGuard → LEP | both tools served by one MCP server |
| `planner_portable` | `planner` | done → MCP; nested `planner_child` twin | `subagent` builtin tool + LEP guard |
| `skillful_analyst_portable` | `skillful_analyst` | done/read_text/write_text → MCP, WriteScopeGuard → LEP | skill system via `search_skills`/`activate_skill` builtins + RUNTIME `skill_manager` |
| `dep_doctor_portable` | `dep_doctor` | 7 audit tools → MCP, RegistryGuard → LEP | `find_alternatives` reaches the bound host model through MGP |
| `threat_intel_portable` | `threat_intel` | 7 tools → MCP, FeedAllowlistGuard → LEP | deterministic extraction plus MGP-backed severity/risk parity |
| `git_detective_portable` | `git_detective` | 10 tools → MCP, CouplingGuard → LEP | repo root via host env; LLM assessment through MGP |

### The mechanical port recipe

Porting an in-process cartridge to a portable twin is a fixed procedure:

1. **Fold every `tools/<n>/execute.py` body into one `_mcp/tools_server.py`** -
   a stdlib-only stdio MCP server. Vendor any data tables (package DBs,
   threat feeds) into the server. Wire it in `config.yaml`:

   ```yaml
   mcp_servers:
     my_tools:
      command: '"${runtime.python_executable}" "${runtime.cartridge_root}/_mcp/tools_server.py"'
   ```

2. **Keep `kind: lep` hooks verbatim** - they are already PROTOCOL-tier.

3. **Keep `builtin_tools:` / `builtin_hooks:`** - already STDLIB-tier.

4. **Keep `resources/compact_service.py` (and similar)** - already
   RUNTIME-tier (host service), not a blocker.

5. **Replace author-owned INPROCESS resources** - e.g.
   `git_detective`'s `repo_config` (a shared path) becomes a
   `$LOOPLET_PROJECT_ROOT` env lookup inside the server; genuine shared
   *mutable* state would move to an SSP service instead.

6. **Use MGP for model-backed tools** - the loader binds its active backend to
   the model gateway and exports `LOOPLET_LLM_SOCKET` to child processes.
   Tools degrade through their documented no-model branch when no backend is
   bound.

7. **Anchor relative paths to the host root** - the MCP subprocess
   resolves relative paths against `os.getcwd()` (the host project
   root), the portable equivalent of `resolve_project_root(runtime)`.

!!! note "Model access is explicit"
   MGP lets an out-of-process tool call the backend currently bound by the
   host. With no backend bound, the tool takes the same graceful-degradation
   branch as the in-process implementation with `ctx.llm is None`.

## Python-host defaults and portable twins

The default and portable profiles coexist. Portability is not automatically
better when a host explicitly wants in-process Python composition.

### `coder` and `coder_portable`

`coder.cartridge` is the Python-host source used by the agent factory. It keeps
tool bodies, several hooks, dynamic memory, and eval wiring in the host.
`coder_portable.cartridge` moves every capability with a protocol equivalent
out of process. Both still need an execution environment that permits shell
commands and workspace edits; the portability distinction is where authored
code runs, not whether the agent has side effects.

### `agent_factory`

`agent_factory` does `extends: ../coder.cartridge`, so the analyzer
reports its blockers as **its own** (`validate_workspace`) **plus all of
`coder`'s** (tagged `(inherited)`). Its one bespoke tool,
`validate_workspace`, exists to *import and validate a Python cartridge* and
is Python-host by definition.

```python
report = analyse_cartridge("examples/agent_factory.cartridge")
assert report.profile == "python-host"
# blockers = validate_workspace + every coder component, via extends
```

## What the evidence proves

The portability claim has three layers:

1. The static analyzer reports zero in-process blockers for each portable
   twin.
2. Dogfood tests execute MCP, LEP, SSP, and MGP boundaries as separate
   processes and verify shared-state and model-call behavior.
3. `examples/alt_runtime/tinyloop.py` is an independent loader that matches the
   reference loader on the spec-pinned declarative subset.

It does not yet prove that the full coder runs under a production Rust, Go, or
TypeScript loader. The bundled protocol servers are currently Python programs,
and SSP/MGP require `AF_UNIX` sockets. Therefore the precise claim is:

> The cartridge has no author-owned code that the loader must import. A
> conforming host can replace or execute the declared protocol services.
