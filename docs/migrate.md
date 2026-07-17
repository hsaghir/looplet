# Migrate an existing tool loop

You do not need to rewrite a working agent into a cartridge. The safest
migration keeps the task, provider, tool implementations, and product tests
fixed while replacing one control boundary at a time.

The first useful milestone is small: your existing tools run through
`composable_loop()`, and every dispatch is returned to your code as a `Step`.
Capture, hooks, cartridges, and eval gates can follow independently.

## Map what you already own

| Existing harness responsibility | Looplet surface |
| --- | --- |
| Provider client | An `LLMBackend` or bundled backend adapter |
| Tool schema and callable | `ToolSpec`, `@tool`, and `tools_from()` |
| Model-to-tool while-loop | `composable_loop()` |
| Cross-cutting runtime policy | A focused hook method |
| Run logs | `ProvenanceSink` and the yielded `Step` stream |
| Product acceptance check | Collector plus outcome grader |
| Versioned harness directory | Optional cartridge |

Keep domain services and business rules outside Looplet. A tool can continue
calling the same function it called before the migration.

## 1. Adapt one tool

Start with a read-only or otherwise low-risk tool. The decorator infers a JSON
Schema from type hints and returns an ordinary `ToolSpec`:

```python
from looplet import tool, tools_from


@tool(description="Look up one service owner by name.")
def lookup_owner(service: str) -> dict:
    return ownership_service.lookup(service)


tools = tools_from([lookup_owner], include_done=True)
```

Do not move service clients, credentials, or persistence into the tool merely
to satisfy Looplet. Close over existing dependencies or expose them through a
small host-owned resource.

If your harness already has a schema registry, adapt it to `ToolSpec` rather
than adding duplicate decorators. The [API map](api.md) links to the registry
and tool types.

## 2. Replace only the control loop

```python
from looplet import LoopConfig, composable_loop


config = LoopConfig(max_steps=8, use_native_tools=False)

for step in composable_loop(
    llm=current_backend,
    tools=tools,
    task={"goal": user_request},
    config=config,
):
    existing_logger.info(step.pretty())
```

Keep text-mode tools for the first parity check if the previous harness used a
text protocol. Native tool calling can be enabled and tested as a separate
change.

Run the same deterministic tool tests and a small set of representative tasks
before adding any hook. If behavior changes, the loop replacement is the only
new variable.

## 3. Keep your provider adapter

You can use an existing client through the backend protocol:

```python
class ExistingBackend:
    def __init__(self, client):
        self.client = client

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        return self.client.complete(
            system=system_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
```

Switching provider SDKs during the loop migration makes failures harder to
attribute. Adopt `OpenAIBackend` or `AnthropicBackend` later if they remove
real integration code.

## 4. Move policy into narrow hooks

Add a hook only when you can name the exact lifecycle boundary it owns. For
example, a release guard that prevents premature completion belongs in
`check_done`:

```python
from looplet import Block


class RequireArtifact:
    def check_done(self, state, session_log, context, step_num):
        if not artifact_exists(state):
            return Block("Create the required artifact before finishing.")
        return None
```

Do not create one hook that contains permissions, retries, metrics, domain
state, and acceptance logic. Compose small hooks and keep outcome acceptance in
post-run graders. See [hooks](hooks.md) and [runtime operations](operations.md).

## 5. Capture one real failure

```python
from looplet import ProvenanceSink


sink = ProvenanceSink(dir="traces/incident-001")
recorded_backend = sink.wrap_llm(current_backend)

for step in composable_loop(
    llm=recorded_backend,
    tools=tools,
    hooks=[sink.trajectory_hook()],
    task={"goal": user_request},
    config=config,
):
    existing_logger.info(step.pretty())

sink.flush()
```

Treat the trace as sensitive prompt evidence. Apply `redact=` before capture
when prompts or tool results may contain credentials or personal data.

Use captured-response replay only when holding model decisions fixed answers
the question you are testing. Tool code, clocks, networks, permissions, and
other side effects execute again. The [experiment guide](experiments.md)
separates replay from mocks and fresh model sampling.

## 6. Turn the failure into an outcome contract

Preserve the task as data, inspect the resulting world independently, and
grade that observation:

```python
def collect_result(state):
    result = read_product_artifact()
    return {"observed_status": result.status}


def eval_status_is_ready(ctx):
    return ctx.artifacts["observed_status"] == ctx.task["expected"]["status"]
```

Avoid requiring a historical tool sequence unless the sequence itself is a
runtime contract. A better model may take a different route to the same valid
outcome. See [behavioral evals](evals.md) for cases, required marks, pytest,
and host-owned holdouts.

## 7. Adopt a cartridge only when it helps review

Once the Python harness has parity and a useful regression contract, a
cartridge can colocate its prompt, tools, hooks, resources, and self-tests:

```bash
looplet describe ./agent.cartridge
looplet diff ./agent-v1.cartridge ./agent-v2.cartridge --show
looplet hash ./agent.cartridge
```

Cartridges are a file representation of the same `AgentPreset` used by the
Python API. They are not required to use the loop, hooks, provenance, replay,
or eval primitives.

## Migration sequence

Use separate commits or pull requests for these checkpoints:

1. adapt tools without changing their implementation;
2. replace the control loop and establish behavior parity;
3. capture run evidence with redaction;
4. add one failure-derived case, collector, and required grader;
5. add only the hooks needed by observed runtime policy;
6. package the harness as a cartridge if file-native review helps;
7. enable native tools, compaction, or other operational controls one at a
   time.

At every checkpoint, keep one cheap parity test that can disprove the current
migration assumption. The goal is not to use every Looplet feature. The goal
is to make the harness boundary inspectable and each future change testable.
