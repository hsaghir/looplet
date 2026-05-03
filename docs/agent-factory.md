# The agent factory

Looplet's killer feature: describe the agent you want in one
paragraph, get a working workspace back in a few minutes. The factory
is built on the same primitives ([`extends:`](workspace.md#extends),
[`builtin_tools:`](workspace.md#builtin-tools),
[`scaffold_workspace`](#scaffold_workspace)) you'd use to hand-roll an
agent — it's just an agent that builds other agents.

![looplet new — agents from a paragraph, in one command](looplet_new.gif)

---

## Three commands

```bash
# 1. Configure any OpenAI-compatible endpoint.
export OPENAI_BASE_URL=https://api.openai.com/v1   # or http://127.0.0.1:11434/v1 for Ollama, etc.
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini                    # or claude-sonnet-4.6, llama3.1, …

# 2. Generate the agent.
looplet new "an agent that takes a URL and returns the page title and a 2-sentence summary" \
    ./url_summarizer.workspace

# 3. Run it on a real task.
looplet run-workspace ./url_summarizer.workspace "Summarize https://example.com"
```

That's the entire user-facing API. No Python code required to get a
working agent.

---

## What the factory writes

For each `looplet new` invocation, the produced workspace contains:

```
my_agent.workspace/
├── workspace.json          # {"name": "my_agent", "schema_version": 1}
├── config.yaml             # max_steps, max_tokens, temperature, done_tool
├── prompts/system.md       # role + tools + workflow + when to call done
├── tools/<name>/
│   ├── tool.yaml           # name, description, parameters, requires
│   └── execute.py          # def execute(ctx, *, ...) -> dict
├── tools/done/             # standard finalizer (always present)
└── tests/test_<name>.py    # at least one content assertion per pure-Python tool
```

The factory also writes one or more `tests/test_<agent>.py` files
that run during the build to verify the agent loads cleanly and
produces correctly-formatted output.

---

## CLI reference

### `looplet new <description> [target]`

Generate a workspace from a brief.

| Flag | Default | Purpose |
|---|---|---|
| `target` (positional) | `./agent.workspace` | Where to write the produced workspace |
| `--name` | derived from `target` | Workspace name (becomes `workspace.json.name`) |
| `--tool TOOL` | _(none)_ | Pre-scaffold a tool by name (repeatable). When omitted, the agent picks tools from the brief. |
| `--max-steps N` | `80` | Override the factory's max steps |
| `--quiet` | _(off)_ | Suppress per-step output |

When `--tool` is supplied, the factory's setup.py pre-scaffolds the
skeleton (saves ~5 LLM turns). Otherwise the agent reads your brief
and picks the tool list itself.

### `looplet run-workspace <path> <task>`

Load a workspace and run it on a task.

| Flag | Default | Purpose |
|---|---|---|
| `--max-steps N` | from workspace's `config.yaml` | Override |
| `--quiet` | _(off)_ | Suppress per-step output |

---

## What the factory does internally

`agent_factory.workspace` is a workspace itself — see
`examples/agent_factory.workspace/`. It [`extends:`](workspace.md#extends)
the bundled `coder.workspace`, so it inherits all coding tools
(`read_file`, `write_file`, `multi_edit`, `bash`, `grep`, `glob`,
`list_dir`, `think`) and adds two factory-specific tools:

* **`scaffold_workspace`** — a built-in tool that calls the
  Python helper `looplet.scaffold.scaffold_workspace()` to write
  the workspace skeleton in one step. The factory's prompt
  instructs the agent to call this FIRST so it skips the
  boilerplate of writing `workspace.json` etc. by hand.
* **`validate_workspace`** — runs `workspace_to_preset()` on the
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

These rules make the produced agents reliable enough to ship.

---

## Pre-scaffold from the host

If you already know what tools the agent should have (e.g. when
calling the factory programmatically from a CLI you're building on
top of looplet), you can pre-scaffold the skeleton via runtime
kwargs. The factory's setup.py honours these:

```python
from looplet import workspace_to_preset, composable_loop
from looplet.types import DefaultState

preset = workspace_to_preset(
    "examples/agent_factory.workspace",
    runtime={
        "workspace": "/path/to/your/project",
        "scaffold_to": "my_agent.workspace",
        "scaffold_name": "my_agent",
        "scaffold_tools": ["fetch_url", "extract_title", "summarize_text"],
    },
)
```

This is exactly what `looplet new --tool fetch_url --tool extract_title …`
does under the hood.

---

## Quality

The factory has been dogfood-tested on five distinct briefs:

| Agent | Tools | Verdict |
|---|---|---|
| `meeting_notes` | extract_decisions, extract_action_items, format_minutes | A — correct minutes from real transcripts |
| `recipe_finder` | brainstorm_recipes, pick_best | A — pipes real recipes through correctly |
| `haiku_writer` | brainstorm_imagery, compose_haiku | A — on-topic 5-7-5 haiku |
| `json_validator` | parse_json, check_required_fields, format_report | A — correct missing-field detection |
| `git_release_notes` | fetch_commits, group_by_type, format_notes | A — real commit shas, no fabrication |

All five reach `done` end-to-end on real LLM input and produce
correct, well-formatted output.

---

## See also

- [Workspace format](workspace.md) — what's in a workspace dir
- [Composition with `extends:`](workspace.md#extends) — how the factory inherits coder.workspace
- [Built-in tools](workspace.md#builtin-tools) — `subagent`, `scaffold_workspace`
