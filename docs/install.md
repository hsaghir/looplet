# Install and configure

Looplet requires Python 3.11 or newer.
The core package has no third-party runtime dependencies.
Model provider adapters are optional extras.

## Install the source version

Use this immutable source snapshot for the [Quickstart](quickstart.md) and the CLI guide.
Install it in your own project without cloning the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "looplet @ git+https://github.com/hsaghir/looplet.git@8fe1196fe27160f737896029621b86c14c9b8a5d"
```

Add `[openai]`, `[anthropic]`, or `[all]` after `looplet` for provider adapters in the same snapshot.
This needs Git and network access during installation.
The fixed-response tutorial itself needs neither a model key nor a model service.
This one install includes result helpers, typed-validation options, capture safeguards, and the unified CLI commands.
PyPI `0.3.0` remains supported, but does not contain these additions.

## Develop Looplet

Clone the repository only when you want to change Looplet itself.
Run these commands in a directory where you want the repository:

```bash
git clone https://github.com/hsaghir/looplet.git
cd looplet
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell.

Install a provider extra from that checkout when you need a real model:

```bash
python -m pip install -e ".[openai]"
```

Use `.[anthropic]` for Anthropic or `.[all]` for both adapters.

## Install the published package

Use a published package when you need a released version.
The latest PyPI release is `0.3.0`.
Use a source install for the current Quickstart and new APIs.
Run its install command inside a virtual environment.

| Goal | Command |
| --- | --- |
| Core APIs or fixed model responses | `python -m pip install looplet` |
| OpenAI or a compatible endpoint | `python -m pip install "looplet[openai]"` |
| Anthropic | `python -m pip install "looplet[anthropic]"` |
| Both provider adapters | `python -m pip install "looplet[all]"` |

Pin the minor version in your application:

```toml title="pyproject.toml"
dependencies = ["looplet>=0.3,<0.4"]
```

Before version `1.0`, a minor release can change an API incompatibly.
Read the [changelog](changelog.md) before you change the minor version.
Source-only features have a note in their API or command guide.

The pinned source install is an interim path, not a new PyPI release.
Keep it until a published version contains the tutorial APIs.
After that release, replace the VCS dependency with its published version pin and rerun your tests.
This guide does not publish a package or change release tags.

## Verify without a provider

This command runs a loop with fixed model responses and checks its results.
It does not need a model key or network connection:

```bash
python -m looplet.examples.hello_world --scripted
```

Use the [Quickstart](quickstart.md) to write a loop that you control.
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

| Option | Effect |
| --- | --- |
| `--no-backend` | Check local settings without network access |
| `--json` | Print a JSON report |
| `--strict` | Return a non-zero exit code for warnings too |

The source command-line interface shares backend selection across `doctor`, `new`, `run`, and `eval`.
It selects Anthropic if an Anthropic key is present.
Otherwise, it selects OpenAI if an OpenAI key or compatible base URL is present.
Set `LOOPLET_PROVIDER=openai` or `LOOPLET_PROVIDER=anthropic` to choose explicitly.
An explicit `--base-url` on examples or evals selects an OpenAI-compatible endpoint.
Automatic selection warns when Anthropic and OpenAI settings are both configured.
`doctor --strict` treats this ambiguity as a failed configuration check.
An explicit provider choice removes the warning without changing stored credentials.

These selection changes are not in PyPI `0.3.0`.
That release has an OpenAI-only `doctor` command.
See the [command-line guide](cli.md) for source-only features and older command names.

## Prepare a package release

The source version is `0.4.0`; it is not yet the published PyPI version.
Maintainers can qualify its distribution without publishing:

```bash
gh workflow run publish.yml --repo hsaghir/looplet --ref master -f version=0.4.0
```

A non-tag dispatch runs tests, builds distributions, and checks an installed wheel.
It uploads build artifacts but does not publish to PyPI or create a GitHub release.
Publication requires a matching release tag and the configured PyPI environment.
Before tagging, update the versioned changelog and obtain release approval.
Keep the source install above until the compatible package is available on PyPI.

## Next

- [Quickstart](quickstart.md): run, inspect, and stop one loop.
- [Migration](migrate.md): replace a loop without new tool implementations.
- [Troubleshooting](troubleshooting.md): find causes of provider, tool, hook, replay, and eval errors.
