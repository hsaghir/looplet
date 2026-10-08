# Install and configure

Looplet requires Python 3.11 or newer. The core package has no third-party
runtime dependencies. Provider SDKs are optional extras, so install only the
adapter you plan to use.

## Choose an installation

| Goal | Command |
| --- | --- |
| Scripted runs, custom backend, or core APIs only | `pip install looplet` |
| OpenAI or an OpenAI-compatible endpoint | `pip install "looplet[openai]"` |
| Anthropic | `pip install "looplet[anthropic]"` |
| Both bundled provider adapters | `pip install "looplet[all]"` |

Use a virtual environment and pin the current minor line in applications:

```toml title="pyproject.toml"
dependencies = ["looplet>=0.4,<0.5"]
```

Before `1.0`, a minor release may include breaking changes. Read the
[changelog](changelog.md) before moving to a new minor line.

## Verify without a provider

This command runs a scripted loop and its live evals. It does not need an API
key or network connection:

```bash
python -m looplet.examples.hello_world --scripted
```

For the complete capture, change, replay, and gate workflow, run the
network-free [regression proof](regression-demo.md) from the installed package:

```bash
looplet-proof
```

## Configure OpenAI

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o
```

```python
from looplet import OpenAIBackend

llm = OpenAIBackend.from_env()
```

`OPENAI_MODEL` is optional and defaults to `gpt-4o`. `OPENAI_BASE_URL` is
optional for the OpenAI cloud endpoint.

## Configure an OpenAI-compatible endpoint

Looplet uses the same adapter for Ollama, vLLM, llama.cpp servers, and hosted
OpenAI-compatible APIs:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.1:8b
export OPENAI_API_KEY=x
```

`OpenAIBackend.from_env()` supplies a non-empty sentinel key when a base URL is
set without a key. Setting `OPENAI_API_KEY=x` explicitly is still useful when
other clients inspect the same environment.

Compatible endpoints do not all implement native tool calling the same way.
Looplet enables native tool calling by default. Missing or rejected native
support fails visibly; it does not silently switch protocols. Inspect the
configured provider with:

```bash
looplet doctor
```

Keep the single mode setting explicit in applications that want to document
the default:

```python
from looplet import LoopConfig

config = LoopConfig(use_native_tools=True)
```

Set it to `False` only when deliberately forcing the text protocol for a
compatibility test.

See [provider recipes](recipes.md) for Ollama, Groq, Together, and explicit
client construction.

## Configure Anthropic

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

```python
from looplet import AnthropicBackend

llm = AnthropicBackend.from_env()
```

`ANTHROPIC_MODEL` is optional. Pin an explicit model in release environments
rather than relying on the library default.

## Bring your own backend

Any object with the small synchronous backend method can drive
`composable_loop()`:

```python
class MyBackend:
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        return my_client.complete(prompt)
```

Set `LoopConfig(use_native_tools=False)` for a backend that implements only
`generate(...)`. A backend that also exposes `generate_with_tools(...)` can
use the default native protocol. Missing or rejected native support remains an
explicit failure. See the [Python API map](api.md) for the backend protocol surface.

## Diagnose the environment

```bash
looplet doctor                 # local checks and backend probe
looplet doctor --no-backend    # no network, suitable for CI
looplet doctor --json          # machine-readable output
looplet doctor --strict        # warnings also produce a non-zero exit
```

The current source CLI uses `looplet.backends.make_backend()` for `doctor`,
live creation, runs, and evals. It selects Anthropic when an Anthropic key is
present, otherwise OpenAI when a cloud key or compatible base URL is present.
Set `LOOPLET_PROVIDER=openai` or `LOOPLET_PROVIDER=anthropic` to choose explicitly
when both are configured. Provider model variables remain optional. An explicit
`--base-url` on examples or evals selects an OpenAI-compatible endpoint.

These unified CLI setup changes are unreleased; PyPI `0.4.0` has the earlier
OpenAI-only doctor. See the [CLI reference](cli.md) for source installation
and compatibility command names.

## Next

- [Quickstart](quickstart.md): build, capture, and test one loop.
- [Migrate an existing loop](migrate.md): adopt Looplet without rewriting the
  whole harness.
- [Troubleshooting](troubleshooting.md): diagnose provider, tool, hook, replay,
  and eval failures.
