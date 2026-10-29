# Python Quickstart

Write Python functions and give them to a loop.
Add configuration only when your agent needs it.

## Install

The Python-first additions are **unreleased**, not in published `0.4.1`.
Use this exact preview revision in a virtual environment:

```bash
python -m venv .venv-looplet
. .venv-looplet/bin/activate
python -m pip install "looplet[openai] @ https://github.com/hsaghir/looplet/archive/cf642eba3a0d4a2239db7400611222797e791540.zip"
```

This non-editable install needs network access, but no Git installation or checkout.
Keep this environment active for the examples below.
Set `OPENAI_API_KEY` and `OPENAI_MODEL` for your provider.
For an OpenAI-compatible server, also set `OPENAI_BASE_URL`.
Do not put credentials in tool functions or source control.

For published `0.4.1`, use the separate [released walkthrough](quickstart.md).

## Write One Loop

```python title="first_loop.py"
from looplet import OpenAIBackend, composable_loop


def lookup_owner(service: str) -> str:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return owners.get(service, "unknown")


llm = OpenAIBackend.from_env()
for step in composable_loop(llm, task="Who owns payments?", tools=[lookup_owner]):
    print(step.pretty())
```

`lookup_owner` remains a normal Python function.
You can call or test it directly.
Its signature describes the tool arguments; its docstring describes the tool to the model.
The loop yields each tool call and result, including failures.

Looplet creates a fresh registry for each function sequence.
It validates top-level argument types, nullability, and enums before tool effects.
Nested validation and other constraints remain the function's responsibility.
It adds the standard `done(summary=...)` completion tool unless a configured terminal is already present.
Explicit assistant-text completion also suppresses that addition.

## Collect The Result

Replace the `for` loop when you only need the final result:

```python
from looplet import run_result

result = run_result(llm, task="Who owns payments?", tools=[lookup_owner])
print(result.output)
print(result.completed, result.termination_reason)
```

`result.steps` retains the steps from this run.
The standard completion output contains `status` and `summary`.
A custom completion function can return structured output instead.
Completion means the loop accepted an output, not that the output is correct.
Check important results independently.

## Change One Thing At A Time

Set a prompt or step limit without constructing other objects:

```python
result = run_result(
    llm,
    task="Who owns payments?",
    tools=[lookup_owner],
    system_prompt="Use the supplied functions and report the evidence.",
    max_steps=8,
)
```

Use `LoopConfig` when you need more control:

```python
from looplet import LoopConfig, resolve_harness

config = LoopConfig(max_model_calls=12, context_window=32_000)
print(resolve_harness(config).describe())
result = run_result(llm, task="Who owns payments?", tools=[lookup_owner], config=config)
```

`max_steps` limits charged tool steps, not total requests or spending.
`max_model_calls` limits admitted backend invocations, including recovery.
Provider billing still depends on the backend and provider.
Sampling settings remain unset unless you supply them.

Ordinary runs compose model retries, native tool handling, deterministic compaction, and private run evidence.
Automatic compaction is not lossless model memory.
Looplet does not blindly repeat tools with unknown effects.
Run files default to `.looplet/runs/<unique-id>`.
They can contain sensitive task data; the host controls access and retention.
Use `run_dir` to choose their location.

Supply a compaction service or [hook](hooks.md) to replace a policy.
Explicit components take precedence over matching defaults.
Inspect the effective policy with `resolve_harness(config, hooks).describe()`.
Use `disabled_defaults` for individual opt-outs or `harness_profile="minimal"` to omit automatic composition.
An explicit registry remains unchanged; Looplet does not add completion tools or type validation to it.
Use `tools_from()` when you need to construct and configure that registry yourself.

## Test Without A Model Service

Use the same functions and entry point with a scripted backend:

```python
from looplet import MockLLMBackend, run_result

mock = MockLLMBackend([
    '{"tool":"lookup_owner","args":{"service":"payments"}}',
    '{"tool":"done","args":{"summary":"fintech-platform"}}',
], cycle=False)
result = run_result(mock, task="Who owns payments?", tools=[lookup_owner])
assert result.completed
assert result.output["summary"] == "fintech-platform"
assert lookup_owner("payments") == "fintech-platform"
```

This checks loop behavior and your functions without paid requests.
It does not prove that a real model will solve the task.
`cycle=False` makes unexpected extra requests visible.

## Stop Early Or Use Async

Normal exhaustion runs cleanup automatically.
Python's `break` does not close a generator. Use `closing` for early exit:

```python
from contextlib import closing

with closing(composable_loop(llm, task="Who owns payments?", tools=[lookup_owner])) as steps:
    for step in steps:
        print(step.pretty())
        break
```

Early closure records `stopped` with reason `stream_closed`, not accepted completion.
Host-supplied clients and resources remain host-owned.

With an asynchronous backend, use `await run_result_async(...)` to collect a result.
Use `async_composable_loop()` and `aclosing` for early exit:

```python
from contextlib import aclosing
from looplet import async_composable_loop


async def inspect_async(async_llm):
    async with aclosing(
        async_composable_loop(async_llm, task="Who owns payments?", tools=[lookup_owner])
    ) as steps:
        async for step in steps:
            print(step.pretty())
            break
```

Use fresh configs, state, and mutable hooks for concurrent runs.
Function sequences get fresh registries, but supplied functions and clients are not cloned.
An explicit registry cannot serve overlapping runs.

## Grow The Same Agent

Add [hooks](hooks.md) for host policy and [configuration](api.md) for runtime controls.
Use custom state only when the default state is insufficient.
Use a [factory](agent-factory.md) when you need reusable dependency construction.
Use [cartridges](cartridge.md) when you need file-based packaging.
These layers use the same loop; none is required for the first program.