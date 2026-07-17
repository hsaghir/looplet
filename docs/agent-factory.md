# The agent factory

`looplet new` is an optional authoring accelerator: describe a harness in one
paragraph and get a reviewable cartridge draft. It is useful for scaffolding,
but generation is not the product and generated code is not automatically a
release-ready harness. Review the files, run structural tests, add behavioral
cases, and gate the outcomes you care about.

The factory is built on the same primitives ([`extends:`](cartridge.md#extends),
[`builtin_tools:`](cartridge.md#builtin-tools),
[`scaffold_cartridge`](cartridge.md#builtin-tools)) you'd use to hand-roll an
agent. It is simply an agent that builds other agents.

---

## Three commands

```bash
# 1. Configure any OpenAI-compatible endpoint.
export OPENAI_BASE_URL=https://api.openai.com/v1   # or http://127.0.0.1:11434/v1 for Ollama, etc.
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-5.5                       # or claude-sonnet-4.6, llama3.1, …

# 2. Scaffold a cartridge draft.
looplet new "an agent that takes a URL and returns the page title and a 2-sentence summary" \
    ./url_summarizer.cartridge

# 3. Review its files, then run it on a real task.
looplet run-cartridge ./url_summarizer.cartridge "Summarize https://example.com"
```

No Python host code is required for this first run. The cartridge is still
ordinary source: inspect its prompt, tools, permissions, and tests before
giving it credentials or production access.

---

## What the factory writes

For each `looplet new` invocation, the produced cartridge contains:

```text
my_agent.cartridge/
├── cartridge.json          # {"name": "my_agent", "schema_version": 1}
├── config.yaml             # max_steps, max_tokens, temperature, done_tool
├── prompts/system.md       # role + tools + workflow + when to call done
├── tools/<name>/
│   ├── tool.yaml           # name, description, parameters, requires
│   └── execute.py          # def execute(ctx, *, ...) -> dict
├── tools/done/             # standard finalizer (always present)
└── tests/test_<name>.py    # at least one content assertion per pure-Python tool
```

The factory also writes one or more `tests/test_<agent>.py` files that run
during the build to verify structural loading and tool-level output shapes.
They are generated smoke tests, not independent evidence of task quality.

---

## CLI reference

### `looplet new <description> [target]`

Generate a cartridge from a brief.

| Flag | Default | Purpose |
| --- | --- | --- |
| `target` (positional) | `./agent.cartridge` | Where to write the produced cartridge |
| `--name` | derived from `target` | Cartridge name (becomes `cartridge.json.name`) |
| `--tool TOOL` | _(none)_ | Pre-scaffold a tool by name (repeatable). When omitted, the agent picks tools from the brief. |
| `--max-steps N` | `80` | Override the factory's max steps |
| `--quiet` | _(off)_ | Suppress per-step output |

When `--tool` is supplied, the host pre-scaffolds the skeleton (saving several
LLM turns). Otherwise the factory agent reads your brief and picks the tool
list itself.

### `looplet run-cartridge <path> <task>`

Load a cartridge and run it on a task.

`looplet run-workspace` remains a backward-compatible alias.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--max-steps N` | from cartridge's `config.yaml` | Override |
| `--quiet` | _(off)_ | Suppress per-step output |

---

## Grounding drafts in existing tools and data

Most useful agents aren't built on greenfield Python. They use tools and data
the team already has: an internal CLI, a vendor SDK, a helper module, or a shell
script. Naming one in the brief makes the factory inspect the real surface and
write thin wrappers instead of hallucinating signatures from training data.

The factory's planning phase recognises three patterns and uses bash + `inspect` to ground itself in the real source:

| Pattern in the brief | What the factory does |
| --- | --- |
| Mentions a CLI on `$PATH` (e.g. `gh`, `kubectl`, `aws`, an internal CLI) | Runs `<cli> --help` and a couple of `<cli> <subcommand> --help` calls; detects `--json` support; writes subprocess-based tool bodies that call the real subcommands |
| Mentions a Python dotted path (`pkg.module` or `pkg.module:Class`) | Imports it, runs `inspect.signature` on the public callables, and writes tool bodies that call the real methods. For class wraps it uses the cartridge `resources/` mechanism: `resources/<name>.py` builds the singleton, every tool declares `requires: [<name>]`, the body looks it up via `ctx.resources["<name>"]` |
| Mentions a local script (`./scripts/foo.sh`, `~/bin/bar.py`) | Reads the file and writes a subprocess- or import-based wrapper from the actual source |

### Example: wrap the GitHub CLI

```bash
looplet new "Wrap the gh CLI as a triage agent that surfaces my open PRs and issues that need attention today" \
    ./gh_triager.cartridge
```

Produces tools like:

```python
# tools/list_my_prs/execute.py
import json, subprocess

def execute(ctx, *, limit: int = 20) -> dict:
    result = subprocess.run(
        ["gh", "pr", "list", "--author", "@me", "--state", "open",
         "--limit", str(limit), "--json",
         "number,title,repository,updatedAt,reviewDecision,isDraft,url"],
        capture_output=True, text=True, check=True,
    )
    return {"prs": json.loads(result.stdout)}
```

The agent picked the right `--json` field set and the right `--author @me` flag because it ran `gh pr list --help` first.

### Example: wrap an existing Python class

```bash
looplet new "Wrap mycompany.search:SearchClient as a SOC investigator with search/pivot/scan tools, backed by DuckDBBackend(':memory:')" \
    ./soc_investigator.cartridge
```

Produces a cartridge with the `resources/` mechanism wired correctly:

```python
# resources/searchclient.py
from mycompany.search import SearchClient
from mycompany.backends.duckdb_backend import DuckDBBackend

def build():
    return SearchClient(DuckDBBackend(":memory:"))
```

```yaml
# tools/search/tool.yaml
name: search
parameters:
  pattern: { type: string }
  window: { type: array, default: null }
  tables: { type: array, default: null }
requires:
  - searchclient
```

```python
# tools/search/execute.py
def execute(ctx, *, pattern, window=None, tables=None) -> dict:
    ep = ctx.resources["searchclient"]
    hits = ep.search(pattern, window=window, tables=tables)
    return {"hits": [h.__dict__ for h in hits], "count": len(hits)}
```

Every signature matches the real class, including default values such as
`mode="full"` and `profile_top_k=10`, because the factory ran
`inspect.signature` first.

### Why this works without flags

The factory's system prompt tells the agent to introspect _first_ (before
scaffolding) whenever the brief mentions an existing CLI, module, or script.
The agent already has `bash`, `read_file`, and `multi_edit`; a one-line
`bash("python -c 'import inspect; ...'")` is enough. There is no special CLI
flag. Naming the dependency in the brief triggers inspection.

If the brief is purely greenfield ("an agent that takes a URL and returns the title…"), the factory falls through to ordinary scaffold-and-fill behaviour. The introspection step only fires when there's something concrete to wrap.

---

## What the factory does internally

`agent_factory.cartridge` is itself a cartridge; see
`examples/agent_factory.cartridge/`. It [`extends:`](cartridge.md#extends)
the bundled `coder.cartridge`, so it inherits all coding tools
(`read_file`, `write_file`, `multi_edit`, `bash`, `grep`, `glob`,
`list_dir`, `think`) and adds two factory-specific tools:

* **`scaffold_cartridge`:** a built-in tool that calls the
    public `looplet.scaffold_cartridge()` helper to write
  the cartridge skeleton in one step. The factory's prompt
  instructs the agent to call this FIRST so it skips the
  boilerplate of writing `cartridge.json` etc. by hand.
* **`validate_workspace`:** runs `cartridge_to_preset()` on the
  produced path and returns a structured success/error.

The factory's system prompt includes "robustness rules" that get
embedded in the produced agent's own system prompt:

1. **Tolerant JSON parsing.** Tools that ask the LLM for JSON
   should use a defensive `_extract_json()` helper that handles
   prose-wrapped or fenced output.
2. **Chained-tool data piping.** When the workflow chains tools,
   the second tool's args MUST come from the first tool's actual
   result. The prompt includes a worked example showing the
   fabrication failure mode and the right pattern.
3. **Defensive arg shapes.** Tools that consume another tool's
   list output should defensively unwrap a `dict` with the same
   key.

These rules avoid several common generation failures. They do not establish
that the produced harness is correct for your environment; only reviewed code
and outcome-grounded tests can do that.

---

## Pre-scaffold from the host

If you already know what tools the agent should have (e.g. when
calling the factory programmatically from a CLI you're building on
top of looplet), pre-scaffold the skeleton **host-side**, then load
the factory cartridge as normal. The agent's first
`scaffold_cartridge` call is idempotent and will treat the existing
skeleton as a no-op.

```python
from looplet import cartridge_to_preset, composable_loop, scaffold_cartridge
from looplet.types import DefaultState

target = "/path/to/your/project/my_agent.cartridge"
scaffold_cartridge(
    target,
    name="my_agent",
    tools=["fetch_url", "extract_title", "summarize_text"],
    overwrite=True,  # idempotent: existing files preserved
)

preset = cartridge_to_preset(
    "examples/agent_factory.cartridge",
    runtime={"workspace": "/path/to/your/project"},
)
```

This is exactly what `looplet new --tool fetch_url --tool extract_title …`
does under the hood. The factory cartridge itself is fully
declarative (`schema_version: 2`, no `setup.py`); filesystem side
effects belong in the host that invokes it, not in the cartridge.

---

## What validation means

Factory behavior is covered by `tests/test_cli_new_smoke.py`. Those tests
establish that generated cartridges load, include required structural pieces,
and report draft/error states consistently. They do **not** establish task
success, safe permissions, or production fitness.

Treat generated tests as structural checks. Add an `evals/` bundle with real
cases, independent outcome collectors, and required graders before release.
The [tutorial](tutorial.md) builds that contract, and the
[regression proof](regression-demo.md) shows the red-to-green workflow without
a model call.

---

## See also

* [Cartridge format](cartridge.md): what belongs in a cartridge directory
* [Composition with `extends:`](cartridge.md#extends): how the factory inherits `coder.cartridge`
* [Built-in tools](cartridge.md#builtin-tools): `subagent`, `scaffold_cartridge`
