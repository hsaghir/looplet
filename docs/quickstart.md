# Quickstart

Build one loop you can inspect, capture it, and see how the same harness
becomes a behavioral contract.

## 1. Run the whole claim first

If you cloned the repository, start with the network-free proof:

```bash
uv sync
uv run python examples/regression_demo/run_demo.py
```

It holds two model responses constant, changes one tool line, executes
the fixed tool in a fresh workspace, and turns an independently
collected outcome from red to green. Read the
[walkthrough](regression-demo.md) when you want every artifact.

## 2. Install

```bash
pip install looplet
```

Core Looplet uses only the Python standard library. Install a provider
extra when you want the bundled adapter:

```bash
pip install "looplet[openai]"
# or
pip install "looplet[anthropic]"
```

Verify the loop locally without an API key:

```bash
python -m looplet.examples.hello_world --scripted
```

The output is a sequence of explicit tool steps followed by two live
eval results.

## 3. Write an owned loop

Configure any OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
```

```python title="my_agent.py"
from looplet import OpenAIBackend, composable_loop, tool, tools_from


@tool(description="Look up one service owner by name.")
def lookup_owner(service: str) -> dict:
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


llm = OpenAIBackend.from_env()
tools = tools_from([lookup_owner], include_done=True)

for step in composable_loop(
    llm=llm,
    tools=tools,
    task={"goal": "Find the owner of payments, then finish."},
    max_steps=5,
):
    print(step.pretty())
```

The important part is not the line count. It is the execution boundary:

- `tools_from(...)` builds the registry that validates and dispatches calls;
- `composable_loop(...)` owns prompt → model → tool → state;
- each dispatch is yielded as a `Step` before control returns to the loop;
- your caller can print, route, pause, approve, or stop on that object.

## 4. Add one exact interception point

Hooks are ordinary objects. Implement only the lifecycle methods you
need:

```python
from looplet import Block


class RequireKnownOwner:
    def check_done(self, state, session_log, context, step_num):
        looked_up = [
            step.tool_result.data
            for step in state.steps
            if step.tool_call.tool == "lookup_owner" and not step.tool_result.error
        ]
        if not any(item.get("owner") for item in looked_up):
            return Block("Look up a known owner before finishing.")
        return None
```

Attach it without subclassing or changing the loop:

```python
for step in composable_loop(..., hooks=[RequireKnownOwner()]):
    print(step.pretty())
```

This assertion is a **runtime guard**: it steers the live loop. Product
quality should usually be checked after the run by an independent
collector, as shown below.

## 5. Capture what the model saw

```python
from looplet import ProvenanceSink


sink = ProvenanceSink(dir="traces/owner_lookup")
recorded_llm = sink.wrap_llm(llm)

for step in composable_loop(
    llm=recorded_llm,
    tools=tools,
    hooks=[sink.trajectory_hook(), RequireKnownOwner()],
    task={"goal": "Find the owner of payments, then finish."},
    max_steps=5,
):
    print(step.pretty())

sink.flush()
```

Inspect the result from the shell:

```bash
python -m looplet show traces/owner_lookup
```

The directory contains exact prompt/response bodies, a model-call
manifest, the trajectory, per-step JSON, stop reason, and metadata.
Apply `redact=` when prompts or results may contain sensitive data.

Captured responses can later drive `replay_loop(...)` with fresh tools
or hooks. This avoids another model call; it does **not** freeze tool
side effects. See [capture and replay](provenance.md).

## 6. Make the outcome a contract

An evaluator should prefer world state over the model's route or claim:

```python
import json
from pathlib import Path

from looplet import EvalHook, eval_mark


def collect_assignment(state):
    path = Path("assignment.json")
    return {
        "assignment": json.loads(path.read_text()) if path.exists() else None,
    }


@eval_mark("required")
def eval_owner_is_correct(ctx):
    return ctx.artifacts["assignment"] == {
        "service": "payments",
        "owner": "fintech-platform",
    }


eval_hook = EvalHook(
    evaluators=[eval_owner_is_correct],
    collectors=[collect_assignment],
    verbose=True,
)
```

That collector assumes the real harness has a tool that writes
`assignment.json`; the host reads the artifact after the loop. It does
not require the model to call a particular tool sequence.

For a complete executable version of this pattern—including protected
expected data and persisted eval runs—use the
[failure-to-regression demo](regression-demo.md).

## 7. Put the harness under review

When a directory is a useful review and release unit, use a cartridge:

```text
owner_agent.cartridge/
├── cartridge.json
├── config.yaml
├── runtime.yaml
├── prompts/system.md
├── tools/lookup_owner/{tool.yaml, execute.py}
├── tools/done/{tool.yaml, execute.py}
└── evals/
    ├── cases/payments_owner.json
    ├── collect_assignment.py
    └── eval_correctness.py
```

```bash
looplet describe ./owner_agent.cartridge
looplet run-cartridge ./owner_agent.cartridge "Find the payments owner"
looplet eval run ./owner_agent.cartridge --out ./eval-runs --threshold 1.0
```

`looplet new` can scaffold a first cartridge from a brief, but generated
output is a starting point: review the files and add cases, collectors,
and required graders before shipping.

## Mental model

1. The model proposes a tool call.
2. The registry validates and dispatches it.
3. Hooks can observe or steer exact lifecycle phases.
4. State records the result.
5. The loop yields a `Step` to your code.
6. End-of-loop collectors inspect the resulting world.
7. Evals grade that evidence and CI enforces the contract.

## Next

- [Tutorial](tutorial.md) — build a colocated case, collector, and required grader.
- [Cartridges](cartridge.md) — file layout, refs, inheritance, and boundaries.
- [Provenance](provenance.md) — capture and controlled re-execution.
- [Evals](evals.md) — outcome philosophy, pytest helpers, and CLI gates.
- [Hooks](hooks.md) — every interception point and return type.
