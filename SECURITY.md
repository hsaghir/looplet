# Security Policy

## Supported versions

`looplet` is pre-1.0; only the latest minor release receives
security fixes. Please upgrade to the current version before filing a
report.

## Reporting a vulnerability

**Please do not report security issues via public GitHub issues.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the repo's **Security** tab.
2. Click **Report a vulnerability**.
3. Provide:
   - A description of the issue and its impact.
   - Steps to reproduce (ideally a minimal PoC).
   - Affected version(s).
   - Your suggested remediation, if any.

If private vulnerability reporting is not available, email the
maintainers listed in `pyproject.toml` with the same information and
the subject line `[looplet security]`.

## What to expect

- **Acknowledgement** within 5 business days.
- **Initial assessment** within 10 business days (severity, whether the
  issue is in-scope, rough timeline).
- **Fix + advisory** coordinated with the reporter. We aim to publish a
  patched release and a GitHub Security Advisory within 30 days for
  high-severity issues, faster when possible.
- **Credit**: reporters are credited in the advisory and CHANGELOG
  unless they prefer to remain anonymous.

## Scope

In scope:

- Bugs in the harness that cause unsafe permission bypass.
- Bugs that cause a cancelled loop or tool to keep running.
- Bugs that cause structured errors (`ToolError`) to leak sensitive
  data into logs or traces unintentionally.
- Deserialization issues in `Conversation` / `Checkpoint` restore.

Out of scope:

- Issues in downstream agents built on top of `looplet` (report to
  that project).
- Issues in optional LLM SDKs (`anthropic`, `openai`) - report
  upstream.
- Prompt-injection vectors against LLM backends (these are a property
  of the backend / your deployment, not the harness).
- Issues that require untrusted code already running inside your
  process.

An enforced host-authorization bypass remains in scope when model output
triggers it. This policy does not promise prompt-injection resistance.

## Threat model

### Assets and actors

Protected assets include host files, credentials, external services, provider
budgets, checkpoint integrity, and private evaluation oracles.

Treat task text, retrieved documents, model responses, and tool results as
untrusted data. Attackers can influence requested tool names and arguments.
They may also control files inside an assigned candidate workspace.

The host owns tool registration, permission policies, approval callbacks, and
storage selection. Python factories, cartridges, skill bundles, hooks, graders,
resource builders, and MCP servers are executable code. Loading them can execute
code before tool permissions apply.

### Trust boundaries

| Threat | Boundary | Required control | Remaining responsibility |
| --- | --- | --- | --- |
| Unauthorized tool execution | Model request to host tool | Restricted registry, argument validation, explicit default-deny | An allowed tool must not expose broader authority |
| Changed approved actions | Approval to dispatch | Bind approval to the exact action and policy | Trust the approval service and policy callbacks |
| Checkpoint path confusion | Run identity or key to filesystem | Validate IDs and literal keys; reject cross-run write collisions | Keep the store private and host-owned |
| Saved artifact path escape | Evidence directory to component reads | Resolve selected files inside the artifact root; reject escaping or broken links | Use stable host-owned storage and authenticate shared evidence |
| Terminal control injection | Saved trace to human display | Escape C0/C1 controls except tabs and newlines | Review remaining layout and Unicode; other commands are not covered |
| Forged recovery state | Saved JSON to live state | Validate identity, structure, and recovery semantics | Authenticate records crossing a trust boundary |
| Secret disclosure | Arguments to logs and saved evidence | Bounded diagnostics and configured redaction | Protect raw state, denial records, and application logs |
| Host code execution | Extension loading to process | Review code and isolate execution | Dependency metadata does not establish code safety |
| Resource exhaustion | Agent execution to compute and provider spend | Step limits, shared request budgets, timeouts, cancellation | Enforce hard deadlines outside the worker |
| Repeated external effects | Retry or resume to tool dispatch | Preserve pending-effect checkpoints and reconcile uncertainty | Supply idempotency or application-specific recovery |
| Oracle contamination | Candidate workspace to host grading | Keep protected oracles outside candidate capabilities | Colocated cartridge evals are editable self-tests |
| Dependency compromise | Optional SDK or extension to process | Pin versions and review dependency advisories | Audit the complete deployed environment |

### Explicit limits

`looplet` is a library, not an OS sandbox or an authenticated tenant service.
A run ID identifies state; it does not authorize access to that state.
Use separate private stores for separate trust principals.

A fresh Git worktree isolates edits, not credentials, processes, networking,
or filesystem permissions. Separate directories do not contain arbitrary code
running as the same OS user. Path validation cannot secure a checkpoint directory
that an attacker can rewrite or replace with symlinks.

File checkpoint stores reject a write when the destination already belongs to
another run. They preserve corrupt existing records rather than replacing them.
The filesystem must support hard links and atomic replacement. Unsupported
operations fail without falling back to an unsafe overwrite. Cooperating writers
can update their own run; arbitrary directory mutation remains outside this control.

Implicit recovery without a run ID stops if any checkpoint is unreadable or malformed.
Legacy keys can contain `__`, so filenames cannot establish a corrupt record's run identity.
Configure an explicit run ID to keep unrelated run namespaces isolated.
Preserve damaged records and reconcile possible effects before retrying.

Saved eval, replay, and trace inspection readers resolve selected files inside the artifact directory.
This includes legacy payloads, optional case data, manifests, and indexed model-call files.
Links to files inside that directory remain supported.
Readers reject escaping or broken selected links before opening their targets.
Broken metadata links cannot enable legacy fallback.
These checks do not authenticate records or detect hard-link aliases.
They do not protect directories that change during validation or reads.
Use a private, stable copy when inspecting evidence from an untrusted source.
Replay still executes fresh tools and effects under the host's policy and isolation controls.

Human `show` and saved-trace `inspect` display C0/C1 controls as visible escapes.
Tabs and newlines remain layout characters.
Ordinary Unicode, JSON values, and saved evidence remain unchanged.
This is not a general defense against misleading text, Unicode formatting, or terminal output from other commands.

Permission rules use first-match order. The default remains `ALLOW` for
compatibility; secure deployments must select `DENY` explicitly.
Matching `DEFAULT` rules resolve through that configured default.
Do not use shell-command substring filters as a sandbox.

Approval binds to argument values and types, including nested values.
Changing an approved action or policy requires a new authorization decision.

Cancellation is cooperative. It cannot undo completed effects or reliably kill
arbitrary Python threads. Provider timeouts and process supervision remain
host responsibilities. A step limit alone does not bound retries or helper
model requests.

JSON serialization does not authenticate a checkpoint or conversation.
Do not resume arbitrary uploaded state. Validate its source before parsing,
and verify its identity before applying it to a run.

## Hardening recommendations

When deploying `looplet`:

- Run candidate code as an unprivileged worker with an isolated filesystem.
- Mount only required inputs, and restrict network egress to reviewed destinations.
- Keep host credentials and protected oracles outside candidate mounts and tools.
- Register only reviewed tools, and validate arguments against their actual effects.
- Configure an explicit default-deny policy before accepting untrusted tasks.
- Review rule order; a later deny does not override an earlier matching allow.
- Keep checkpoints, traces, and evaluation records in private host-owned storage.
- Set restrictive file permissions before starting the process, such as `umask 077` on Unix.
- Redact captured evidence before sharing it; denial records can contain complete arguments.
- Configure provider deadlines, shared model-call budgets, and a host execution deadline.
- Thread a `CancelToken` through tools and backends that support cooperative cancellation.
- Reconcile uncertain effects before resuming; do not discard pending checkpoints to force a retry.
- Pin Looplet, optional SDKs, and extension dependencies in your lock file.
- Review dependency advisories and the changelog before upgrading.

Optional provider extras require `certifi>=2026.10.19` for the corrected CA bundle.
See [GHSA-6q48-6c7w-9gw2](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2).
Refresh existing environments and rebuild images that include an older bundle.

A minimal policy configuration uses existing public APIs:

```python
from looplet import CancelToken, LoopConfig, PermissionDecision, PermissionEngine, PermissionHook

engine = PermissionEngine(default=PermissionDecision.DENY)
engine.allow("read_status")
engine.allow("done")

config = LoopConfig(max_steps=5, cancel_token=CancelToken())
hooks = [PermissionHook(engine)]
```

Register reviewed implementations for the allowed tools. Pass `config` and
`hooks` to the loop. This configuration restricts dispatch; it does not isolate
the process or cap provider spend.

## Security regression checks

The local checks use mock providers and disposable files. They verify specific
controls, not universal resistance to prompt injection or hostile extensions.

```bash
uv run pytest -q \
  tests/test_permission_engine.py tests/test_permission_hook_smoke.py \
  tests/test_checkpoint.py tests/test_bundles_smoke.py \
  tests/test_conversation.py tests/test_artifact_compatibility.py \
  tests/test_replay_and_cli_smoke.py \
  tests/test_provenance_smoke.py tests/test_native_tools.py \
  tests/test_llm_cancel_token.py tests/test_rpc_cancel.py
```

The tests cover denied effects across sync/async and native/text execution,
approval integrity, diagnostic privacy, and checkpoint lookup and write boundaries.
They also cover saved-artifact containment, replay effect boundaries, bundle entrypoint
containment, recovery safety, redaction, and cooperative cancellation.
Trace inspection tests check inert display controls, link containment, and unchanged JSON and saved evidence.
Approval tests include fixed-seed nested mutations and unchanged-action controls.
Checkpoint tests include synchronized writers and preservation on storage failures.
Run `make check` before publishing changes.

For deployment review, also use OWASP's
[Excessive Agency guidance](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/).
It complements these specific regression checks; it does not certify a deployment.
