# looplet

[![CI](https://github.com/hsaghir/looplet/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/looplet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hsaghir/looplet/branch/master/graph/badge.svg)](https://codecov.io/gh/hsaghir/looplet)
[![PyPI version](https://img.shields.io/pypi/v/looplet.svg)](https://pypi.org/project/looplet/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hsaghir/looplet/blob/master/LICENSE)

**Test-driven harness engineering for Python agents.**

## Own the loop. Test every change.

Looplet is for Python teams maintaining a tool-calling agent that is
fundamentally one model in one loop. It helps answer the post-prototype
question: when a prompt, tool, hook, model, or permission changes, what behavior
broke?

- keep execution visible as an iterator of typed steps;
- capture prompts, responses, tool calls, and stop reasons as readable files;
- replay recorded model responses through changed harness code when that is a
  valid experiment;
- collect actual world state and gate required outcomes in pytest or CI.

No graph DSL, hosted control plane, or required third-party runtime
dependencies.

[Documentation](https://hsaghir.com/looplet/) | [Why Looplet](https://hsaghir.com/looplet/why-looplet/) | [Quickstart](https://hsaghir.com/looplet/quickstart/) | [Evals](https://hsaghir.com/looplet/evals/)

## Run the mechanism proof

Run the complete network-free proof directly from PyPI:

```bash
uvx --from looplet==0.4.0 looplet-proof
```

```text
1. CAPTURE v1 with fixed model responses
   model decisions: publish_report -> done
   collected profit: 200
   required eval: FAIL (0.00)

2. CHANGE one reviewable harness line
   - "profit": revenue + cost,
   + "profit": revenue - cost,

3. REPLAY captured responses with fresh v2 tool execution
   same decisions: true
   collected profit: 40
   required eval: PASS (1.00)
```

The example is a mechanism proof, not a claim that arithmetic needs an agent
eval. It persists the harness versions, model-call cassette, trajectories,
fresh workspaces, host-observed artifacts, and grader results so every layer is
inspectable.

> **Replay is controlled re-execution, not deterministic simulation.**
> Captured model responses stay fixed. Tools, clocks, networks, randomness,
> and other side effects are fresh unless the host isolates or mocks them.

[Read the walkthrough and its limits](https://hsaghir.com/looplet/regression-demo/).

## Start with one owned loop

```bash
pip install "looplet[openai]"

export OPENAI_API_KEY=...
export OPENAI_MODEL=...
```

```python
from looplet import OpenAIBackend, composable_loop, tool, tools_from


@tool(description="Look up one service owner by name.")
def lookup_owner(service: str) -> dict:
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


for step in composable_loop(
    llm=OpenAIBackend.from_env(),
    tools=tools_from([lookup_owner], include_done=True),
    task={"goal": "Find the owner of payments, then finish."},
    max_steps=5,
):
    print(step.pretty())
```

`composable_loop()` is a generator. Each dispatch returns to the caller as a
`Step`, so normal Python can log, route, pause, approve, or stop at the exact
tool boundary. Hooks are ordinary duck-typed objects; implement only the
lifecycle methods a policy needs.

For a zero-network hello world:

```bash
pip install looplet
python -m looplet.examples.hello_world --scripted
```

## Adopt it incrementally

A working agent does not need to become a cartridge on day one:

1. keep the current provider and tool implementations;
2. adapt one low-risk tool and replace only the control loop;
3. establish parity before adding capture, hooks, or eval gates;
4. preserve one real failure as a case, collector, and required grader;
5. use a file-native cartridge only when it improves review or distribution.

See [Migrate an existing loop](https://hsaghir.com/looplet/migrate/) for the
smallest-change sequence.

## Grade outcomes, not yesterday's trajectory

A smarter model may use different tools and still produce a better result.
Use trajectory checks for harness mechanics, such as whether a guard fired.
Use collectors for product outcomes, such as tests passing or a record being
written correctly.

```python
import subprocess

from looplet import eval_mark


def collect_tests(state):
    result = subprocess.run(["pytest", "-q"], check=False)
    return {"tests_passing": result.returncode == 0}


@eval_mark("required")
def eval_tests_pass(ctx):
    return ctx.artifacts["tests_passing"]
```

Expected values remain grader-only during a run. Colocated cartridge evals are
versioned self-tests, not protected holdouts. A promotion oracle must remain in
a host-owned runner outside candidate authority; arbitrary candidate code also
requires an OS or process isolation boundary.

[Read behavioral evals](https://hsaghir.com/looplet/evals/).

## Optional reviewable cartridges

A cartridge stores the runnable harness as ordinary files:

```text
agent.cartridge/
├── cartridge.json
├── config.yaml
├── runtime.yaml
├── prompts/system.md
├── tools/<name>/{tool.yaml, execute.py}
├── hooks/<order>_<name>/{config.yaml, hook.py}
├── resources/<name>.py
├── memory/*.md
└── evals/{cases/, collect_*.py, eval_*.py}
```

```bash
looplet describe ./agent.cartridge
looplet diff ./agent-v1.cartridge ./agent-v2.cartridge --show
looplet eval run ./agent.cartridge --out ./eval-runs --threshold 1.0
```

Cartridges are optional. They load into the same Python primitives used above.
See the [cartridge guide](https://hsaghir.com/looplet/cartridge/) for schema,
inheritance, trust boundaries, and protocol portability.

## When Looplet fits

Use Looplet when:

- one model calls tools until it is done;
- the team already works in Python, Git, pytest, and CI;
- harness changes need regression evidence;
- exact interception points and local artifacts matter;
- the team wants to own execution and behavioral contracts.

Use something else when:

- the application is naturally a branching, durable workflow graph;
- a managed dashboard or annotation system should be the source of truth;
- the requirement is a turnkey assistant rather than a toolkit;
- a small disposable loop is still enough.

Looplet can run inside a workflow engine and export evidence to observability
systems. It does not try to replace either one.

## Documentation

| Start here | Purpose |
| --- | --- |
| [Install](https://hsaghir.com/looplet/install/) | Provider extras, environment setup, and network-free checks |
| [Quickstart](https://hsaghir.com/looplet/quickstart/) | Build, capture, and test a first loop |
| [Migration](https://hsaghir.com/looplet/migrate/) | Adopt the loop boundary without rewriting domain tools |
| [Failure to regression](https://hsaghir.com/looplet/regression-demo/) | Inspect the executable proof and experiment limits |
| [Provenance](https://hsaghir.com/looplet/provenance/) | Capture and controlled re-execution |
| [Evals](https://hsaghir.com/looplet/evals/) | Outcome collectors, required graders, pytest, and CI |
| [Operations](https://hsaghir.com/looplet/operations/) | Async loops, retries, cancellation, permissions, and checkpoints |
| [CLI](https://hsaghir.com/looplet/cli/) | Commands and machine-readable output |
| [Python API](https://hsaghir.com/looplet/api/) | Curated public surface |

Advanced packaging, skills, and MCP/LEP/SSP/MGP boundaries remain documented
under [cartridges](https://hsaghir.com/looplet/cartridge/),
[skills](https://hsaghir.com/looplet/skills/), and
[portability](https://hsaghir.com/looplet/portability/), rather than the first
adoption path.

## Stability

Looplet follows SemVer. Before `1.0`, minor versions may make breaking
changes; pin the current minor line:

```toml
looplet>=0.4,<0.5
```

The current release is `0.4.0`. See the
[changelog](https://github.com/hsaghir/looplet/blob/master/CHANGELOG.md) and
[path to 1.0](https://github.com/hsaghir/looplet/blob/master/ROADMAP.md#path-to-10).

## Contributing

Bug reports, focused examples, backend adapters, and integrity fixes are
welcome. New core features must remain domain-neutral and include evidence for
the behavior they change.

See [CONTRIBUTING.md](https://github.com/hsaghir/looplet/blob/master/CONTRIBUTING.md).
Security issues go through
[SECURITY.md](https://github.com/hsaghir/looplet/blob/master/SECURITY.md).

## License

Apache License 2.0. See [LICENSE](https://github.com/hsaghir/looplet/blob/master/LICENSE).