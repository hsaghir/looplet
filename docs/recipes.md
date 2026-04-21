# Recipes

Copy-paste recipes for the most common integrations. Each is a small,
self-contained snippet you can drop into your agent.

## Ollama (local models, zero API key)

```python
from openharness.backends import OpenAIBackend
from openai import OpenAI

llm = OpenAIBackend(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    model="llama3.1:8b",
)
```

See [`examples/ollama_hello.py`](../src/openharness/examples/ollama_hello.py)
for a runnable end-to-end example.

## Groq / Together / any OpenAI-compatible endpoint

```python
import os
from openharness.backends import OpenAIBackend
from openai import OpenAI

llm = OpenAIBackend(
    OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"]),
    model=os.environ["OPENAI_MODEL"],
)
```

## Anthropic

```python
from openharness.backends import AnthropicBackend
from anthropic import Anthropic

llm = AnthropicBackend(Anthropic(), model="claude-sonnet-4-latest")
```

## OpenTelemetry

Wrap the built-in `Tracer` with an OTel exporter:

```python
from openharness import Tracer, TracingHook
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
otel_tracer = trace.get_tracer("openharness")

class OTelBridge:
    def __init__(self, otel): self.otel = otel
    def start_span(self, name, **kw):
        span = self.otel.start_span(name, attributes=kw)
        return span   # duck-typed; must support .end() / .set_attribute()

hooks = [TracingHook(OTelBridge(otel_tracer))]
```

## MCP server as a tool source

```python
from openharness import MCPToolAdapter, BaseToolRegistry

reg = BaseToolRegistry()
adapter = MCPToolAdapter.connect_stdio(
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)
for spec in adapter.list_tools():
    reg.register(spec)
```

No MCP SDK required — `openharness` speaks JSON-RPC over stdio directly.

## Cost accounting on top of provenance

```python
from openharness.provenance import ProvenanceSink

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
from openharness import eval_discover, eval_run, EvalContext

def eval_matches_golden(ctx):
    golden = open("golden/run_1/tool_sequence.txt").read().splitlines()
    return ctx.tool_sequence == golden

ctx = EvalContext.from_trajectory_dir("traces/run_1/")
print(eval_run([eval_matches_golden], ctx))
```

## Crash-resume with conversation preserved

```python
from openharness import LoopConfig, resume_loop_state

config = LoopConfig(checkpoint_dir="./checkpoints", max_steps=100)

# First run:
for step in composable_loop(llm=llm, tools=tools, config=config, task=task):
    ...                                  # Ctrl-C or crash

# Later, same arguments — resumes from last checkpoint:
state, offset = resume_loop_state("./checkpoints") or (None, 0)
for step in composable_loop(
    llm=llm, tools=tools, config=config,
    task=task, state=state, resume_step=offset,
):
    ...
```

## Deny-by-default shell tool

```python
from openharness import PermissionEngine, PermissionRule

engine = PermissionEngine(default="DENY")
engine.add(PermissionRule(tool="bash", args={"command": r"^(ls|pwd|cat)( |$)"}, decision="ALLOW"))
engine.add(PermissionRule(tool="bash", args={"command": r"rm "}, decision="ASK"))

config = LoopConfig(permissions=engine, approval_handler=my_ask_handler)
```

## Run a sub-loop with its own tools

```python
from openharness import run_sub_loop

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
