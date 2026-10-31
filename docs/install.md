# Install and configure

Looplet requires Python 3.11 or newer.
The core package has no third-party runtime dependencies.
Model provider adapters are optional extras.

## Install the current build

Use this **unreleased source preview** for the [Python quickstart](python.md).
It includes the Python-first API and current run ownership fixes.
Those additions are not in PyPI `0.4.1`.
Use the [published package](#install-the-published-package) when you need that released API.

```bash
python -m venv .venv-looplet
. .venv-looplet/bin/activate
python -m pip install "looplet @ https://github.com/hsaghir/looplet/archive/3ecf285ba98b1d3635af2c9711a1a73910147f8f.zip"
```

Installation needs network access, but no Git checkout.
The examples with fixed responses need no model key or model service.
On Windows, activate `.venv-looplet\Scripts\Activate.ps1` in PowerShell.

Both the preview and release report version `0.4.1`.
The version alone does not identify their APIs.
Record the source fingerprint to distinguish them:

```bash
looplet doctor --no-backend --json
```

Keep the revision and artifact hashes in your deployment lockfile.

## Choose provider extras

Install the adapter from the same preview revision.
For OpenAI or a compatible endpoint:

```bash
python -m pip install "looplet[openai] @ https://github.com/hsaghir/looplet/archive/3ecf285ba98b1d3635af2c9711a1a73910147f8f.zip"
```

Use `[anthropic]` for Anthropic or `[all]` for both adapters.
Choose your provider below, then keep the same Python functions and loop.

## Verify without a provider

This command runs a loop with fixed model responses and checks its results.
It does not need a model key or network connection:

```bash
python -m looplet.examples.hello_world --scripted
```

Use the [Python quickstart](python.md) to write a loop that you control.
Then run the [regression example](regression-demo.md) to check a tool change:

```bash
looplet-proof
```

## Configure OpenAI

```bash
export LOOPLET_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o
```

```python
from looplet import OpenAIBackend

llm = OpenAIBackend.from_env()
```

`OPENAI_MODEL` is optional.
Its default is `gpt-4o`.
You do not need `OPENAI_BASE_URL` for the OpenAI cloud endpoint.

## Configure an OpenAI-compatible endpoint

Looplet uses the OpenAI adapter for compatible endpoints.
These include Ollama, vLLM, and llama.cpp servers:

```bash
export LOOPLET_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.1:8b
export OPENAI_API_KEY=x
```

If a base URL has no key, `OpenAIBackend.from_env()` supplies a placeholder key.
You can set `OPENAI_API_KEY=x` explicitly if other clients read the same environment.

Some compatible endpoints do not support native tool calls.
Looplet uses native tool calls by default.
It reports an error if the backend does not support them.
It does not switch protocols without your choice.
Check the configured provider with:

```bash
looplet doctor
```

Use this configuration to state the default explicitly:

```python
from looplet import LoopConfig

config = LoopConfig(use_native_tools=True)
```

Set `use_native_tools=False` if your backend needs the JSON-text protocol.

See [provider recipes](recipes.md) for Ollama, Groq, Together, and explicit clients.

## Configure Anthropic

```bash
export LOOPLET_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

```python
from looplet import AnthropicBackend

llm = AnthropicBackend.from_env()
```

`ANTHROPIC_MODEL` is optional.
Set an explicit model in release environments instead of the library default.

## Bring your own backend

`composable_loop()` accepts a backend object with this synchronous method.
Replace `my_client` with the client that your application uses:

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

Set `LoopConfig(use_native_tools=False)` if the backend implements only `generate(...)`.
A backend with `generate_with_tools(...)` can use native tool calls.
The loop reports an error if the backend rejects native calls.
See the [Python API](api.md#backends) for the full backend interface.

## Diagnose the environment

```bash
looplet doctor
```

| Option         | Effect                                       |
| -------------- | -------------------------------------------- |
| `--no-backend` | Check local settings without network access  |
| `--json`       | Print a JSON report                          |
| `--strict`     | Return a non-zero exit code for warnings too |

The command-line interface shares backend selection across `doctor`, `new`, `run`, and `eval`.
It selects Anthropic if an Anthropic key is present.
Otherwise, it selects OpenAI if an OpenAI key or compatible base URL is present.
Set `LOOPLET_PROVIDER=openai` or `LOOPLET_PROVIDER=anthropic` to choose explicitly.
An explicit `--base-url` on examples or evals selects an OpenAI-compatible endpoint.
Automatic selection warns when Anthropic and OpenAI settings are both configured.
`doctor --strict` treats this ambiguity as a failed configuration check.
An explicit provider choice removes the warning without changing stored credentials.

Looplet `0.4.0` includes these selection rules for both providers.
See the [command-line guide](cli.md) for options and older command names.

## Install the published package

The latest PyPI release is `0.4.1`.
Use its [released walkthrough](quickstart.md), not the Python-first preview examples.
Install it in a separate virtual environment:

```bash
python -m venv .venv-looplet-release
. .venv-looplet-release/bin/activate
python -m pip install "looplet==0.4.1"
```

Add `[openai]`, `[anthropic]`, or `[all]` before `==0.4.1` for provider adapters.
For example, `looplet[openai]==0.4.1` installs the released OpenAI adapter.
Before version `1.0`, a minor release can change an API incompatibly.
Read the [changelog](changelog.md) before changing versions.

## Try unreleased factory commands

The preview above also includes the optional experiment starter and factory commands.
Use the same fixed revision; a second preview build is not needed.
See the [experiment starter](quickstart.md#optional-experiment-starter) for the shell workflow.
The [`looplet new` generator](agent-factory.md) uses a model to draft a cartridge.
A [Python factory](api.md#optional-preset-conveniences) is an ordinary function that builds fresh run dependencies.

<a id="unreleased-safety-fixes"></a>

## Safety fixes in 0.4.1

Use `0.4.0` only when reproducing that older release.
It lacks the fixes below.
Release `0.4.1` and the recommended preview include them.

| Behavior                  | Published `0.4.0`                                                               | Release `0.4.1`                                                                            |
| ------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Serial native batches     | A batch may execute before its first yielded `Step`.                            | Each serial call runs before its own step; stopping prevents later calls.                  |
| Native response redaction | `redact=` does not scrub native response blocks.                                | `redact=` scrubs nested string values and keys.                                            |
| Python replay defaults    | Uses model-call count as the budget and an empty task.                          | Uses saved tool steps and task when omitted.                                               |
| Tool timeout errors       | May report a blank, retryable timeout.                                          | Warns about unknown effects and requires reconciliation before retrying.                   |
| Rich Python annotations   | `Annotated`, `Any`, and multi-type unions may become string schemas.            | Preserves supported contracts and rejects unsupported annotations clearly.                 |
| Callable preset typing    | Accepts functions at runtime but declares only `ToolSpec` inputs.               | Type annotations match callable inputs and reusable sequences.                             |
| Failed terminal dispatch  | May report accepted completion with no output, even with `--require-completed`. | Rejects failed calls and permits correction within the remaining budget.                   |
| Free tool budget          | `DefaultState` charges every recorded step despite `free=True`.                 | Successful free calls preserve budget; failures and terminal attempts still consume steps. |
| CLI failure details       | Completion output may omit the underlying error.                                | Human and JSON output include bounded last-error details for incomplete runs.              |
| Missing terminal setup    | Unknown terminal calls consume turns before failure.                            | Warns before the first model call when no configured terminal tool is registered.          |

Do not treat published `0.4.0` native captures as redacted.
Keep them private and review evidence before sharing it.
Explicit concurrent dispatch still runs a batch before yielding its results.

## Develop Looplet

Clone the repository only when changing Looplet itself:

```bash
git clone https://github.com/hsaghir/looplet.git
cd looplet
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Install `.[openai]`, `.[anthropic]`, or `.[all]` from that checkout for provider adapters.
See [Contributing](contributing.md) for development checks.

## Prepare a package release

Maintainers can qualify a distribution without publishing.
Use the version declared in the selected revision:

```bash
gh workflow run publish.yml --repo hsaghir/looplet --ref master -f version=0.4.1
```

A non-tag dispatch runs tests, builds distributions, and checks an installed wheel.
It uploads build artifacts but does not publish to PyPI or create a GitHub release.
This qualifies a source candidate; it does not publish a new package version.
Publication requires a matching release tag and the configured PyPI environment.
Before tagging, update the versioned changelog and obtain release approval.
Publish only the qualified commit after its pull request has merged.

## Next

- [Python quickstart](python.md): run, inspect, and extend one loop.
- [Migration](migrate.md): replace a loop without new tool implementations.
- [Troubleshooting](troubleshooting.md): find causes of provider, tool, hook, replay, and eval errors.
