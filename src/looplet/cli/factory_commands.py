"""``looplet new`` and ``looplet run-workspace`` CLI implementation.

These two commands are the user-facing entry points to the agent
factory. The vision is: a developer with one hour to spare can pip
install looplet, write one paragraph describing the agent they want,
run a single command, and have a working agent that operates on
their input.

## ``looplet new <description>``

Runs the bundled :mod:`agent_factory.workspace` against a brief and
writes the produced workspace to a directory. After this completes,
``./<name>.workspace/`` contains a fully-loaded looplet workspace.

Required env vars (any OpenAI-compatible endpoint):

* ``OPENAI_BASE_URL`` — e.g. ``http://127.0.0.1:19823/v1`` for a
  local proxy or ``https://api.openai.com/v1`` for direct OpenAI.
* ``OPENAI_API_KEY`` — your key (or any string for proxies that
  don't validate it).
* ``OPENAI_MODEL`` — model id, e.g. ``gpt-4o-mini`` or
  ``claude-sonnet-4.6``.

## ``looplet run-workspace <path> <task>``

Runs an existing workspace on a task and prints the final result.
Same env vars as above. This is the "watch the agent work" command
once ``looplet new`` has produced a workspace.

## Why split into two modules?

``__main__.py`` already has ten subcommands. This module isolates the
two factory-facing ones so they can evolve independently of the
bundle / trace / eval CLI machinery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if sys.stdout.isatty() else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if sys.stdout.isatty() else s


def _check_env() -> int:
    """Verify required env vars are set. Returns 0 on success."""
    missing: list[str] = []
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(
            _red(f"error: missing required env vars: {', '.join(missing)}"),
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print("Set them to point at any OpenAI-compatible endpoint:", file=sys.stderr)
        print(
            _dim('    export OPENAI_BASE_URL="http://127.0.0.1:19823/v1"'),
            file=sys.stderr,
        )
        print(_dim('    export OPENAI_API_KEY="sk-..."'), file=sys.stderr)
        print(_dim('    export OPENAI_MODEL="gpt-4o-mini"'), file=sys.stderr)
        print(file=sys.stderr)
        print(
            _dim("Run ``looplet doctor`` to verify connectivity."),
            file=sys.stderr,
        )
        return 1
    return 0


def _build_backend():
    """Construct an OpenAIBackend from env vars."""
    from looplet.backends import OpenAIBackend  # noqa: PLC0415

    return OpenAIBackend(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
    )


def _factory_workspace_path() -> Path:
    """Locate the bundled ``examples/agent_factory.workspace`` directory.

    Looplet ships with this workspace in the repo's ``examples/``
    folder. When installed via ``pip install``, the examples may not
    be co-packaged; in that case we fall back to ``LOOPLET_FACTORY_DIR``
    or print a clear error.
    """
    # Walk up from this file looking for examples/agent_factory.workspace.
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "examples" / "agent_factory.workspace"
        if candidate.is_dir():
            return candidate
    env_override = os.environ.get("LOOPLET_FACTORY_DIR")
    if env_override and Path(env_override).is_dir():
        return Path(env_override)
    raise FileNotFoundError(
        "Could not locate examples/agent_factory.workspace. "
        "Set LOOPLET_FACTORY_DIR to point at it, or run from the looplet repo."
    )


# ── ``looplet new`` ─────────────────────────────────────────────


def cmd_new(args: argparse.Namespace) -> int:
    if _check_env() != 0:
        return 1

    description: str = args.description
    target_dir: Path = args.target.resolve()
    name: str = args.name or target_dir.name.replace(".workspace", "").replace("-", "_")
    tools: list[str] = args.tool or []

    print(f"{_bold('looplet new')} → {target_dir}")
    print(_dim(f"  brief:  {description[:80]}{'…' if len(description) > 80 else ''}"))
    print(_dim(f"  name:   {name}"))
    if tools:
        print(_dim(f"  tools:  {', '.join(tools)} (pre-scaffolded)"))
    print(_dim(f"  model:  {os.environ['OPENAI_MODEL']}"))
    print()

    try:
        from looplet import composable_loop, workspace_to_preset  # noqa: PLC0415
        from looplet.types import DefaultState  # noqa: PLC0415

        factory = _factory_workspace_path()
    except Exception as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # The factory's setup.py honours scaffold_to + scaffold_tools at
    # load time. When the user passed --tool flags, pre-scaffold the
    # skeleton so the agent skips the boilerplate; otherwise the agent
    # picks the tool list itself from the brief.
    runtime: dict = {"workspace": str(target_dir.parent)}
    if tools:
        runtime["scaffold_to"] = target_dir.name
        runtime["scaffold_name"] = name
        runtime["scaffold_tools"] = tools

    try:
        backend = _build_backend()
        preset = workspace_to_preset(str(factory), runtime=runtime)
    except Exception as exc:
        print(_red(f"error: factory load failed: {exc}"), file=sys.stderr)
        return 1

    state = DefaultState(max_steps=args.max_steps or preset.config.max_steps)
    brief_for_factory = description
    if not tools:
        brief_for_factory = (
            f"Build a workspace at ./{target_dir.name}/ for the following agent:\n\n"
            f"{description}\n\n"
            f"Workspace name should be: {name}"
        )

    t0 = time.time()
    n_steps = 0
    n_denies = 0
    last_done_summary: str | None = None
    try:
        for step in composable_loop(
            llm=backend,
            config=preset.config,
            tools=preset.tools,
            state=state,
            hooks=preset.hooks,
            task={"goal": brief_for_factory},
        ):
            n_steps += 1
            tool_call = step.tool_call
            tool_result = step.tool_result
            if tool_call is None:
                continue
            err = (tool_result and tool_result.error) or (
                tool_result.data.get("error") if tool_result and tool_result.data else None
            )
            if err:
                n_denies += 1
            if not args.quiet:
                tag = _red("✗") if err else _green("✓")
                short = json.dumps(tool_call.args, default=str)[:80]
                print(f"  {tag} step {n_steps:>2}: {tool_call.tool}({short})")
            if (
                tool_call.tool == "done"
                and tool_result
                and tool_result.data
                and "summary" in tool_result.data
            ):
                last_done_summary = str(tool_result.data["summary"])
    except KeyboardInterrupt:
        print(_red("\ninterrupted"), file=sys.stderr)
        return 130
    except Exception as exc:
        print(_red(f"error during build: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print()
    print(f"{_green('✓')} built in {elapsed:.1f}s — {n_steps} steps, {n_denies} denies")

    # Verify the workspace actually loads.
    if not target_dir.is_dir():
        print(_red(f"\nerror: workspace not created at {target_dir}"), file=sys.stderr)
        return 1
    try:
        sub_preset = workspace_to_preset(str(target_dir))
        produced_tools = sorted(sub_preset.tools._tools.keys())
        n_tools = len(produced_tools)
        sys_prompt_chars = len(sub_preset.config.system_prompt or "")
    except Exception as exc:
        print(
            _red(f"\nerror: produced workspace failed to load: {exc}"),
            file=sys.stderr,
        )
        return 1

    print()
    print(f"{_bold('produced workspace:')} {target_dir}")
    print(f"  tools:  {', '.join(produced_tools)}  ({n_tools})")
    print(f"  prompt: {sys_prompt_chars} chars")
    if last_done_summary:
        print(f"  agent says: {last_done_summary[:120]}")
    print()
    print(_bold("next:"))
    print(f'  looplet run-workspace {target_dir} "<your task>"')
    return 0


# ── ``looplet run-workspace`` ───────────────────────────────────


def cmd_run_workspace(args: argparse.Namespace) -> int:
    if _check_env() != 0:
        return 1

    workspace_path: Path = args.workspace.resolve()
    task: str = args.task

    if not workspace_path.is_dir():
        print(_red(f"error: workspace not found at {workspace_path}"), file=sys.stderr)
        return 1

    try:
        from looplet import composable_loop, workspace_to_preset  # noqa: PLC0415
        from looplet.types import DefaultState  # noqa: PLC0415
    except Exception as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1

    print(f"{_bold('looplet run')} {workspace_path}")
    print(_dim(f"  task:  {task[:100]}{'…' if len(task) > 100 else ''}"))
    print(_dim(f"  model: {os.environ['OPENAI_MODEL']}"))
    print()

    try:
        backend = _build_backend()
        preset = workspace_to_preset(
            str(workspace_path), runtime={"workspace": str(workspace_path.parent)}
        )
    except Exception as exc:
        print(_red(f"error: workspace load failed: {exc}"), file=sys.stderr)
        return 1

    state = DefaultState(max_steps=args.max_steps or preset.config.max_steps)
    t0 = time.time()
    n_steps = 0
    final_summary: str | None = None
    final_data: dict | None = None
    try:
        for step in composable_loop(
            llm=backend,
            config=preset.config,
            tools=preset.tools,
            state=state,
            hooks=preset.hooks,
            task={"goal": task},
        ):
            n_steps += 1
            tool_call = step.tool_call
            tool_result = step.tool_result
            if tool_call is None:
                continue
            if not args.quiet:
                err = (tool_result and tool_result.error) or (
                    tool_result.data.get("error") if tool_result and tool_result.data else None
                )
                tag = _red("✗") if err else _green("✓")
                short = json.dumps(tool_call.args, default=str)[:80]
                print(f"  {tag} step {n_steps:>2}: {tool_call.tool}({short})")
            if tool_call.tool == "done" and tool_result and tool_result.data:
                final_data = dict(tool_result.data)
                final_summary = str(tool_result.data.get("summary", ""))
    except KeyboardInterrupt:
        print(_red("\ninterrupted"), file=sys.stderr)
        return 130

    elapsed = time.time() - t0
    print()
    print(f"{_green('✓')} done in {elapsed:.1f}s — {n_steps} steps")
    print()
    if final_summary:
        print(_bold("result:"))
        print(final_summary)
    elif final_data:
        print(_bold("result:"))
        print(json.dumps(final_data, indent=2, default=str)[:2000])
    return 0


# ── argparse wiring (called from __main__.main) ─────────────────


def add_subparsers(sub: "argparse._SubParsersAction") -> None:
    """Register ``new`` and ``run-workspace`` on the top-level parser.

    Called from :mod:`looplet.__main__` so the two commands are
    available as ``looplet new ...`` and ``looplet run-workspace ...``.
    """
    new_p = sub.add_parser(
        "new",
        help="Generate a new agent workspace from a brief (uses agent_factory)",
        description=(
            "Generate a working looplet agent workspace from a one-paragraph "
            "English brief. Requires OPENAI_BASE_URL / OPENAI_API_KEY / "
            "OPENAI_MODEL env vars (any OpenAI-compatible endpoint)."
        ),
    )
    new_p.add_argument(
        "description",
        help="Plain-English description of what the agent should do",
    )
    new_p.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=Path("./agent.workspace"),
        help="Target directory for the produced workspace (default: ./agent.workspace)",
    )
    new_p.add_argument(
        "--name",
        help="Workspace name (default: derived from target directory)",
    )
    new_p.add_argument(
        "--tool",
        action="append",
        help="Pre-scaffold a tool by name (repeatable). When omitted, the agent picks tools from the brief.",
    )
    new_p.add_argument(
        "--max-steps",
        type=int,
        help="Override the factory's default max_steps (default: 80)",
    )
    new_p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    new_p.set_defaults(_handler=cmd_new)

    run_p = sub.add_parser(
        "run-workspace",
        help="Run a workspace on a task and print the final result",
        description=(
            "Load an existing looplet workspace and run it against a task. "
            "Requires the same env vars as ``looplet new``."
        ),
    )
    run_p.add_argument("workspace", type=Path, help="Path to a workspace directory")
    run_p.add_argument("task", help="Task to give the agent")
    run_p.add_argument(
        "--max-steps",
        type=int,
        help="Override the workspace's default max_steps",
    )
    run_p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    run_p.set_defaults(_handler=cmd_run_workspace)


__all__ = ["add_subparsers", "cmd_new", "cmd_run_workspace"]
