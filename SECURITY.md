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
- Issues in optional LLM SDKs (`anthropic`, `openai`) — report
  upstream.
- Prompt-injection vectors against LLM backends (these are a property
  of the backend / your deployment, not the harness).
- Issues that require untrusted code already running inside your
  process.

## Hardening recommendations

When deploying `looplet`:

- Always configure a `PermissionEngine` with an explicit `default`
  (usually `DENY`) instead of relying on rule coverage.
- Thread a `CancelToken` through long-running loops so you can stop
  them cleanly.
- Treat checkpoint files as sensitive — they contain prompt and tool
  result history.
- Pin the `looplet` version in your lock file and review
  `CHANGELOG.md` before upgrading.
