# Recipes

Copy-paste recipes for the most common integrations. Each is a small,
self-contained snippet you can drop into your agent.

## Ollama (local models, zero API key)

```python
from looplet.backends import OpenAIBackend

llm = OpenAIBackend(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3.1:8b",
)
```

See [`examples/ollama_hello.py`](https://github.com/hsaghir/looplet/blob/master/src/looplet/examples/ollama_hello.py)
for a runnable end-to-end example.

## Groq / Together / any OpenAI-compatible endpoint

```python
import os
from looplet.backends import OpenAIBackend

llm = OpenAIBackend(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
)
```

## Diagnose your local setup

```bash
looplet doctor                 # checks env vars and probes backend if configured
looplet doctor --no-backend    # local-only checks, safe for CI
looplet doctor --json          # machine-readable diagnostics
```

`doctor` catches the common first-run issues: missing backend env vars,
unavailable OpenAI extras, and OpenAI-compatible proxies that claim tool
support but return plain text instead of native `tool_use` blocks.

## Decorator-first tool construction

```python
from looplet import tool, tools_from

@tool(description="Search the docs by keyword.", concurrent_safe=True)
def search_docs(query: str, limit: int = 5) -> dict:
    return {"results": search(query, limit)}

tools = tools_from([search_docs], include_done=True)
```

The decorator infers JSON Schema from type hints, treats parameters with
defaults as optional, uses the docstring when no description is provided,
and still produces a plain `ToolSpec` you can inspect or mutate.

## Anthropic

```python
from looplet.backends import AnthropicBackend

llm = AnthropicBackend(api_key="sk-ant-...", model="claude-sonnet-4-latest")
```

## OpenTelemetry

Wrap the built-in `Tracer` with an OTel exporter:

```python
from looplet import Tracer, TracingHook
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
otel_tracer = trace.get_tracer("looplet")

class OTelBridge:
    def __init__(self, otel): self.otel = otel
    def start_span(self, name, **kw):
        span = self.otel.start_span(name, attributes=kw)
        return span   # duck-typed; must support .end() / .set_attribute()

hooks = [TracingHook(OTelBridge(otel_tracer))]
```

## MCP server as a tool source

```python
from looplet import MCPToolAdapter, BaseToolRegistry

reg = BaseToolRegistry()
adapter = MCPToolAdapter.connect_stdio(
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)
for spec in adapter.list_tools():
    reg.register(spec)
```

No MCP SDK required — `looplet` speaks JSON-RPC over stdio directly.

## Cost accounting on top of provenance

```python
from looplet.provenance import ProvenanceSink

sink = ProvenanceSink(dir="traces/run_1/")
llm = sink.wrap_llm(my_llm)

for step in composable_loop(llm=llm, ...):
    ...
sink.flush()

# Post-hoc cost calculation
import json
total_in, total_out = 0, 0
for line in open("traces/run_1/manifest.jsonl"):
    rec = json.loads(line)
    total_in += rec.get("input_tokens", 0)
    total_out += rec.get("output_tokens", 0)
cost = total_in * 3e-6 + total_out * 15e-6      # $3/M in, $15/M out
print(f"${cost:.4f}")
```

## Golden-test a trajectory

```python
from looplet import eval_discover, eval_run, EvalContext

def eval_matches_golden(ctx):
    golden = open("golden/run_1/tool_sequence.txt").read().splitlines()
    return ctx.tool_sequence == golden

ctx = EvalContext.from_trajectory_dir("traces/run_1/")
print(eval_run([eval_matches_golden], ctx))
```

## Crash-resume with conversation preserved

```python
from looplet import LoopConfig

config = LoopConfig(checkpoint_dir="./checkpoints", max_steps=100)

# First run:
for step in composable_loop(llm=llm, tools=tools, config=config, task=task):
    ...                                  # Ctrl-C or crash

# Later, run the same loop with the same checkpoint_dir. looplet loads
# the latest checkpoint automatically when config.initial_checkpoint is unset.
for step in composable_loop(llm=llm, tools=tools, config=config, task=task):
    ...
```

## Deny-by-default shell tool

```python
from looplet import PermissionDecision, PermissionEngine, PermissionHook

engine = PermissionEngine(default=PermissionDecision.DENY)
engine.allow(
    "bash",
    arg_matcher=lambda args: args.get("command", "").startswith(("ls", "pwd", "cat")),
    reason="read-only shell commands are safe by default",
)
engine.ask(
    "bash",
    arg_matcher=lambda args: "rm " in args.get("command", ""),
    reason="destructive shell commands need approval",
)

hooks = [PermissionHook(engine)]
```

## Run a sub-loop with its own tools

```python
from looplet import run_sub_loop

result = run_sub_loop(
    llm=llm,
    tools=specialist_tools,              # scoped toolset
    task={"goal": "summarise the repo"},
    parent_tracer=tracer,                # shares telemetry
)
```

---

Have a recipe we should add? Open a PR against
[`docs/recipes.md`](recipes.md) — recipes under ~40 lines are welcome.
