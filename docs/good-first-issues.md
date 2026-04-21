# Good first issues

A curated list of small, well-scoped tasks for first-time contributors.
Each one is 1–3 hours of work and doesn't require deep familiarity
with the loop internals.

When you pick one up, please **open an issue** (or comment on an
existing one) so we don't duplicate work.

## Backends

### 1. Add a Gemini (Google) backend

**Where:** new file `src/openharness/backends_gemini.py`, exported from
`openharness/__init__.py` as `GeminiBackend`.

**What:** implement the `LLMBackend` protocol on top of the
`google-generativeai` Python SDK. Follow the shape of
`AnthropicBackend` / `OpenAIBackend` in
[`src/openharness/backends.py`](../src/openharness/backends.py).

**Acceptance:**
- Add `google-generativeai` to an optional extra `[gemini]`.
- Sync + async + streaming variants mirror the existing backends.
- New tests under `tests/test_backends_gemini.py` (mock the SDK — no
  real network).
- Extend the "How it compares" table in the README.

### 2. Add a Bedrock (AWS) backend

Same shape as #1, on top of `boto3` / `aiobotocore`. Extra is
`[bedrock]`.

### 3. Add a local `llama.cpp` / `llama-cpp-python` backend recipe

Documentation-only: extend [docs/recipes.md](recipes.md) with a fully
runnable example that uses the OpenAI-compatible server mode.

## Examples

### 4. Add `examples/ollama_hello.py`

Mirror `examples/hello_world.py` but with the Ollama base URL baked
in, zero API key, and a comment explaining `ollama pull llama3.1:8b`.

### 5. Add `examples/research_agent.py`

A small 3-tool research agent (web-search stub, note-taking,
summarise) that demonstrates `run_sub_loop` for a specialist subtask.

## Evals

### 6. Ship reusable eval recipes as `openharness.evals.recipes`

**Where:** new file `src/openharness/evals_recipes.py`.

**What:** port these common evals into stable importable functions:

- `eval_efficiency` — fewer steps = higher score
- `eval_no_tool_errors` — fraction of tool calls that didn't error
- `eval_parse_quality` — fraction of LLM outputs that parsed first try
- `eval_tool_diversity` — number of unique tools used / total tools
- `eval_completed` — did the agent call `done()`?

**Acceptance:** each has a docstring, default thresholds, and at least
one test against a fixture trajectory under `tests/fixtures/`.

## Docs & housekeeping

### 7. Fill in missing docstrings

Run `uv run pyright src/openharness/` and fix any public symbol that
is missing a docstring. Start with the smallest modules
(`events.py`, `flags.py`, `memory.py`).

### 8. Add a `make` target shim

A minimal `Makefile` with `test`, `lint`, `format`, `build`, `docs`
targets that wrap the `uv run` equivalents. Helps people coming from
other projects.

### 9. Write the "why not LangGraph" FAQ entry

A 300-word honest comparison for
[docs/faq.md](faq.md) (create new). Focus on *when* LangGraph is
actually the better choice — credibility comes from telling people
when not to use you.

### 10. Improve error messages for common misuse

Find three cases where a user mistake produces a confusing exception
(e.g. forgetting `state=`, registering a tool with no `execute=`,
passing a dict instead of `ToolSpec`) and turn each into a clear
actionable error with a link to the relevant doc section.

---

Don't see one that fits? Open an issue tagged `good first issue`
proposing your own — we're happy to scope it with you.
