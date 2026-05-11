# 05 — Recursive editing (agents that build agents)

Cartridges are files. An agent that can read, write, and validate
files can build them. There is no separate "code generation"
abstraction needed.

This snippet **points at the existing `examples/agent_factory.cartridge/`**
shipped with looplet. The factory's tools are
`scaffold_workspace`, `write_file`, `read_file`, and
`validate_workspace`. Given a one-paragraph brief, it produces a
new cartridge directory and validates it.

```bash
# Requires: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL set.
# This is the same command the looplet CLI runs under the hood.
looplet new \
    "Build a URL summarizer that fetches a page and returns its title and a 2-sentence summary." \
    /tmp/url_summarizer.workspace
```

After the run, `/tmp/url_summarizer.workspace/` will be a valid
cartridge you can load with `workspace_to_preset(...)` or run with
`looplet run-workspace`.

## Why this matters

The factory does not need to know anything about code generation,
framework internals, or runtime construction. It manipulates a
static artifact, the same way infrastructure-as-code tools generate
Terraform modules and scaffolders generate npm packages. The agent
factory is itself a cartridge (`examples/agent_factory.cartridge/`);
recursion is just composition.
