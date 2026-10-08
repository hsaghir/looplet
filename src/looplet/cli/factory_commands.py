"""Cartridge creation and execution adapters for the Looplet CLI.

``new`` runs the bundled factory to produce a reviewable harness draft;
``new --offline`` writes placeholders without constructing a backend.
Review tool code and add outcome contracts before releasing either draft.

Live creation and execution use ``looplet.backends.make_backend()``:
an OpenAI cloud key, a compatible base URL, or an Anthropic key is enough.
Optional model defaults match Python; ``LOOPLET_PROVIDER`` selects a
provider explicitly when multiple providers are configured.

The unified ``run`` command delegates cartridge execution here.
``run-cartridge`` and ``run-workspace`` remain compatible entry points.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

from looplet.bundled import bundled_cartridge_path
from looplet.cli import completion_payload, execution_output


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if sys.stdout.isatty() else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if sys.stdout.isatty() else s


def _check_env() -> int:
    """Validate the same provider configuration used by Python and CLI runs."""
    try:
        _build_backend()
    except Exception as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        print("Run `looplet doctor` to inspect provider configuration.", file=sys.stderr)
        return 1
    return 0


def _build_backend():
    """Construct the environment-selected backend through the shared resolver."""
    from looplet.backends import make_backend  # noqa: PLC0415

    return make_backend()


def _factory_workspace_path() -> Path:
    """Locate the bundled ``agent_factory.cartridge`` directory.

    Source checkouts use ``examples/``; wheels and sdists install the
    complete factory dependency tree under ``looplet/_bundled``.
    ``LOOPLET_FACTORY_DIR`` remains an explicit override.
    """
    env_override = os.environ.get("LOOPLET_FACTORY_DIR")
    if env_override and Path(env_override).is_dir():
        return Path(env_override)

    try:
        return bundled_cartridge_path("agent_factory")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not locate the bundled agent_factory.cartridge. "
            "Reinstall looplet or set LOOPLET_FACTORY_DIR to an explicit factory cartridge."
        ) from exc


# ── ``looplet new`` ─────────────────────────────────────────────
def cmd_new(args: argparse.Namespace) -> int:
    description: str = args.description
    target_dir: Path = args.target.resolve()
    name: str = args.name or target_dir.name.replace(".cartridge", "").replace(
        ".workspace", ""
    ).replace("-", "_")
    tools: list[str] = args.tool or []

    if getattr(args, "offline", False):
        from looplet.cartridge.scaffold import scaffold_cartridge  # noqa: PLC0415

        try:
            root = scaffold_cartridge(target_dir, name=name, tools=tools)
            prompt = root / "prompts" / "system.md"
            prompt.write_text(
                prompt.read_text(encoding="utf-8").replace(
                    "<TODO: one-paragraph mission statement - what does this agent do?>",
                    description,
                ),
                encoding="utf-8",
            )
        except (OSError, ValueError) as exc:
            print(_red(f"error: {exc}"), file=sys.stderr)
            return 1
        print(f"scaffolded cartridge draft: {root}")
        return 0

    if args.max_steps is not None and args.max_steps < 1:
        print(_red("error: --max-steps must be positive"), file=sys.stderr)
        return 1
    try:
        backend = _build_backend()
    except Exception as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1
    model_label = str(getattr(backend, "_model", type(backend).__name__))

    print(f"{_bold('looplet new')} → {target_dir}")
    if not getattr(args, "pretty", False):
        # In --pretty mode the printer renders its own header below;
        # avoid printing the duplicate banner.
        print(_dim(f"  brief:  {description[:80]}{'…' if len(description) > 80 else ''}"))
        print(_dim(f"  name:   {name}"))
        if tools:
            print(_dim(f"  tools:  {', '.join(tools)} (pre-scaffolded)"))
        print(_dim(f"  model:  {model_label}"))
        print()
    try:
        from looplet import RunResult, cartridge_to_preset  # noqa: PLC0415

        factory = _factory_workspace_path()
    except Exception as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # When the user passed --tool flags, pre-scaffold the skeleton
    # host-side so the agent's first ``scaffold_cartridge`` call is a
    # no-op (idempotent - existing files are preserved). This used to
    # live in agent_factory.cartridge/setup.py; v2 cartridges can't
    # ship executable Python at the root, so the scaffolding moves to
    # the host CLI where filesystem side effects belong.
    runtime: dict = {"workspace": str(target_dir.parent)}
    if tools:
        from looplet.cartridge.scaffold import scaffold_cartridge  # noqa: PLC0415

        scaffold_cartridge(target_dir, name=name, tools=list(tools), overwrite=True)

    try:
        preset = cartridge_to_preset(str(factory), runtime=runtime)
    except Exception as exc:
        print(_red(f"error: factory load failed: {exc}"), file=sys.stderr)
        return 1

    if args.max_steps is not None:
        preset.config.max_steps = args.max_steps
        preset.state.max_steps = args.max_steps
    state = preset.state
    brief_for_factory = description
    if not tools:
        brief_for_factory = (
            f"Scaffold a cartridge draft at ./{target_dir.name}/ for this harness:\n\n"
            f"{description}\n\n"
            f"Cartridge name should be: {name}"
        )

    pretty = None
    if getattr(args, "pretty", False) and not args.quiet:
        from looplet.cli._pretty import PrettyPrinter  # noqa: PLC0415

        pretty = PrettyPrinter(
            title=f"looplet new · building {name}",
            max_steps=preset.config.max_steps,
        )
        pretty.header(
            [
                f"  brief:  {description[:80]}{'…' if len(description) > 80 else ''}",
                f"  target: {target_dir}",
                f"  model:  {model_label}",
            ]
        )

    t0 = time.time()
    n_steps = 0
    n_denies = 0
    last_done_summary: str | None = None
    try:
        for step in preset.run(
            backend,
            task={"goal": brief_for_factory},
        ):
            n_steps += 1
            tool_call = step.tool_call
            tool_result = step.tool_result
            if tool_call is None:
                if pretty is not None:
                    pretty.step(step)
                continue
            data = tool_result.data if tool_result is not None else None
            err = (tool_result and tool_result.error) or (
                data.get("error") if isinstance(data, dict) else None
            )
            if err:
                n_denies += 1
            if pretty is not None:
                pretty.step(step)
            elif not args.quiet:
                tag = _red("✗") if err else _green("✓")
                short = json.dumps(tool_call.args, default=str)[:80]
                print(f"  {tag} step {n_steps:>2}: {tool_call.tool}({short})")
            if (
                tool_call.tool == "done"
                and tool_result
                and isinstance(data, dict)
                and "summary" in data
            ):
                last_done_summary = str(data["summary"])
    except KeyboardInterrupt:
        print(_red("\ninterrupted"), file=sys.stderr)
        return 130
    except Exception as exc:
        print(_red(f"error during build: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1
    finally:
        preset.close()

    result = RunResult.from_state(state)
    if result.failed:
        print(_red(f"error: factory stopped ({result.termination_reason})"), file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print()
    print(f"{_green('✓')} draft built in {elapsed:.1f}s - {n_steps} steps, {n_denies} denies")

    # Verify the cartridge actually loads.
    if not target_dir.is_dir():
        print(_red(f"\nerror: cartridge not created at {target_dir}"), file=sys.stderr)
        return 1
    try:
        sub_preset = cartridge_to_preset(str(target_dir))
        try:
            produced_tools = sorted(sub_preset.tools._tools.keys())
            n_tools = len(produced_tools)
            sys_prompt_chars = len(sub_preset.config.system_prompt or "")
        finally:
            sub_preset.close()
    except Exception as exc:
        print(
            _red(f"\nerror: produced cartridge failed to load: {exc}"),
            file=sys.stderr,
        )
        return 1

    print()
    print(f"{_bold('produced cartridge draft:')} {target_dir}")
    print(f"  tools:  {', '.join(produced_tools)}  ({n_tools})")
    print(f"  prompt: {sys_prompt_chars} chars")
    if last_done_summary:
        print(f"  agent says: {last_done_summary[:120]}")
    print()
    print(_bold("next:"))
    print(f'  looplet run {shlex.quote(str(target_dir))} "<your task>"')
    return 0


# ── ``looplet run-cartridge`` (``run-workspace`` alias) ────────
def cmd_run_workspace(args: argparse.Namespace) -> int:
    output = sys.stdout
    with execution_output(getattr(args, "json", False)):
        return _run_cartridge(args, output=output)


def _run_cartridge(args: argparse.Namespace, *, output: TextIO) -> int:
    json_output = getattr(args, "json", False)
    if json_output and getattr(args, "pretty", False):
        print(_red("error: --json cannot be used with --pretty"), file=sys.stderr)
        return 1
    if args.max_steps is not None and args.max_steps < 1:
        print(_red("error: --max-steps must be positive"), file=sys.stderr)
        return 1

    workspace_path: Path = args.workspace.resolve()

    if not workspace_path.is_dir():
        print(_red(f"error: cartridge not found at {workspace_path}"), file=sys.stderr)
        return 1

    task: str = args.task
    if task == "-":
        task = sys.stdin.read()
        if not task.strip():
            print(_red("error: task read from stdin is empty"), file=sys.stderr)
            return 1

    try:
        from looplet import (  # noqa: PLC0415
            ProvenanceSink,
            RunResult,
            cartridge_to_preset,
        )
        from looplet.cartridge.runtime_helpers import resolve_project_root  # noqa: PLC0415
    except Exception as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1

    project_root = getattr(args, "project_root", None)
    runtime: dict | None = None
    if project_root is not None:
        runtime = {"project_root": str(project_root.expanduser().resolve())}
    effective_project_root = Path(resolve_project_root(runtime))
    if not effective_project_root.is_dir():
        print(
            _red(f"error: project root is not a directory: {effective_project_root}"),
            file=sys.stderr,
        )
        return 1

    trajectory_metadata: dict[str, str] = {}
    parent_trace = getattr(args, "parent_trace", None)
    if parent_trace is not None:
        if getattr(args, "no_trace", False):
            print(_red("error: --parent-trace cannot be used with --no-trace"), file=sys.stderr)
            return 1
        parent_trajectory = parent_trace.expanduser().resolve() / "trajectory.json"
        try:
            parent_data = json.loads(parent_trajectory.read_text(encoding="utf-8"))
            if not isinstance(parent_data, dict):
                raise ValueError("trajectory.json must contain an object")
            parent_run_id = parent_data.get("run_id")
            if not isinstance(parent_run_id, str) or not parent_run_id.strip():
                raise ValueError("trajectory.json has no run_id")
        except (OSError, ValueError) as exc:
            print(_red(f"error: invalid parent trace {parent_trace}: {exc}"), file=sys.stderr)
            return 1
        trajectory_metadata["parent_run_id"] = parent_run_id

    try:
        scripted_responses = getattr(args, "scripted_response", None)
        if scripted_responses:
            from looplet.testing import MockLLMBackend  # noqa: PLC0415

            if any(not response.strip() for response in scripted_responses):
                raise ValueError("--scripted-response must not be empty")
            backend = MockLLMBackend(responses=scripted_responses)
        else:
            backend = _build_backend()
        preset = cartridge_to_preset(str(workspace_path), runtime=runtime)
    except Exception as exc:
        print(_red(f"error: cartridge load failed: {exc}"), file=sys.stderr)
        return 1

    model_label = str(getattr(backend, "_model", type(backend).__name__))
    if not json_output:
        print(f"{_bold('looplet run-cartridge')} {workspace_path}")
        if not getattr(args, "pretty", False):
            print(_dim(f"  task:  {task[:100]}{'…' if len(task) > 100 else ''}"))
            print(_dim(f"  model: {model_label}"))
        print()

    if args.max_steps is not None:
        preset.config.max_steps = args.max_steps
        preset.state.max_steps = args.max_steps
    state = preset.state
    sink = None
    effective_trace_dir = None
    if not getattr(args, "no_trace", False):
        effective_trace_dir = getattr(args, "trace_dir", None)
        if effective_trace_dir is None:
            cartridge_name = workspace_path.name.removesuffix(".cartridge")
            effective_trace_dir = (
                effective_project_root
                / ".looplet"
                / "traces"
                / f"{cartridge_name}-{uuid.uuid4().hex[:12]}"
            )
        sink = ProvenanceSink(dir=effective_trace_dir, metadata=trajectory_metadata)
        backend = sink.wrap_llm(backend)

    hooks = []
    if sink is not None:
        hooks.append(sink.trajectory_hook())
    pretty = None
    if getattr(args, "pretty", False) and not args.quiet:
        from looplet.cli._pretty import PrettyPrinter  # noqa: PLC0415

        pretty = PrettyPrinter(
            title=f"looplet run-cartridge · {workspace_path.name}",
            max_steps=preset.config.max_steps,
        )
        pretty.header(
            [
                f"  task:  {task[:80]}{'…' if len(task) > 80 else ''}",
                f"  model: {model_label}",
            ]
        )
    t0 = time.time()
    n_steps = 0
    final_summary: str | None = None
    final_data: Any = None
    interrupted = False
    saved_trace_dir = None
    trace_failed = False
    try:
        for step in preset.run(
            backend,
            extra_hooks=hooks,
            task={"goal": task},
        ):
            n_steps += 1
            tool_call = step.tool_call
            tool_result = step.tool_result
            if tool_call is None:
                if pretty is not None:
                    pretty.step(step)
                continue
            if pretty is not None:
                pretty.step(step)
            elif not args.quiet and not json_output:
                _data = tool_result.data if tool_result else None
                err = (tool_result and tool_result.error) or (
                    _data.get("error") if isinstance(_data, dict) else None
                )
                tag = _red("✗") if err else _green("✓")
                short = json.dumps(tool_call.args, default=str)[:80]
                print(f"  {tag} step {n_steps:>2}: {tool_call.tool}({short})")
    except KeyboardInterrupt:
        print(_red("\ninterrupted"), file=sys.stderr)
        interrupted = True
    except Exception as exc:
        print(_red(f"error during run: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1
    finally:
        try:
            if sink is not None:
                saved_trace_dir = sink.flush()
        except Exception as exc:
            print(_red(f"error: could not save trace: {exc}"), file=sys.stderr)
            trace_failed = True
        finally:
            preset.close()

    if trace_failed:
        return 1
    if interrupted:
        if saved_trace_dir is not None and not json_output:
            print(f"  Trace: {saved_trace_dir}")
        return 130

    elapsed = time.time() - t0
    result = RunResult.from_state(state)
    termination_reason = result.termination_reason
    final_data = result.output
    if isinstance(final_data, dict):
        final_summary = str(final_data.get("summary", ""))
    if json_output:
        print(
            json.dumps(
                completion_payload(
                    result,
                    steps=n_steps,
                    duration_ms=elapsed * 1000,
                    trace_dir=saved_trace_dir,
                ),
                indent=2,
                default=str,
            ),
            file=output,
        )
        return 1 if result.failed else 0

    print()
    if result.completed:
        status = _green("✓ done")
    elif result.failed:
        status = _red(f"✗ failed ({termination_reason or 'unknown'})")
    else:
        status = _dim(f"· stopped ({termination_reason or 'unknown'})")
    print(f"{status} in {elapsed:.1f}s - {n_steps} steps")
    print()
    if final_summary:
        print(_bold("result:"))
        print(final_summary)
    elif final_data is not None:
        print(_bold("result:"))
        print(json.dumps(final_data, indent=2, default=str)[:2000])
    if saved_trace_dir is not None:
        print(f"\n  Trace: {saved_trace_dir}")
    return 1 if result.failed else 0


def cmd_portability(args: argparse.Namespace) -> int:
    """Statically report a cartridge's cross-runtime portability."""
    cartridge_path: Path = args.cartridge.resolve()
    if not cartridge_path.is_dir():
        print(
            _red(f"error: cartridge not found at {cartridge_path}"),
            file=sys.stderr,
        )
        return 1

    try:
        from looplet.cartridge import analyse_cartridge  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1

    try:
        report = analyse_cartridge(cartridge_path)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"error: analysis failed: {exc}"), file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    # Non-zero exit when the cartridge is NOT fully portable and the
    # caller asked us to enforce it (useful in CI conformance gates).
    if getattr(args, "require_portable", False) and report.profile != "portable":
        return 2
    return 0


# ── argparse wiring (called from __main__.main) ─────────────────
def add_subparsers(sub: "argparse._SubParsersAction") -> None:
    """Register ``new`` and ``run-cartridge`` on the top-level parser.

    Called from :mod:`looplet.__main__` so the two commands are
    available as ``looplet new ...`` and ``looplet run-cartridge ...``.
    """
    new_p = sub.add_parser(
        "new",
        help="Scaffold a reviewable cartridge draft from a brief",
        description=(
            "Scaffold a reviewable Looplet cartridge draft from a one-paragraph "
            "English brief. Review and add behavioral contracts before release. "
            "Uses the provider configuration reported by looplet doctor, "
            "or --offline for placeholders without a model or API key."
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
        default=Path("./agent.cartridge"),
        help="Target directory for the cartridge draft (default: ./agent.cartridge)",
    )
    new_p.add_argument(
        "--name",
        help="Cartridge name (default: derived from target directory)",
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
    new_p.add_argument(
        "--offline", action="store_true", help="Scaffold a draft without a model or API key"
    )
    new_p.add_argument(
        "--pretty",
        action="store_true",
        help="Render the build as a live human-friendly trace (boxed header, per-step reasoning + result summary, colored).",
    )
    new_p.set_defaults(_handler=cmd_new)

    run_p = sub.add_parser(
        "run-cartridge",
        aliases=["run-workspace"],
        help="Run a cartridge on a task and print the final result",
        description=(
            "Load an existing looplet cartridge and run it against a task. "
            "Uses the provider configuration reported by looplet doctor. "
            "``run-workspace`` is a back-compat alias."
        ),
    )
    run_p.add_argument(
        "workspace",
        metavar="cartridge",
        type=Path,
        help="Path to a cartridge directory",
    )
    run_p.add_argument("task", help="Task to give the agent, or '-' to read it from stdin")
    run_p.add_argument(
        "--max-steps",
        type=int,
        help="Override the cartridge's default max_steps",
    )
    run_p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Directory the agent operates on (its tools' working directory). "
            "Defaults to $LOOPLET_PROJECT_ROOT, then the current git repo, "
            "then the current working directory."
        ),
    )
    run_p.add_argument(
        "--trace-dir",
        type=Path,
        help="Write provenance trace output here",
    )
    run_p.add_argument(
        "--parent-trace",
        type=Path,
        help="Link this run to a parent trace directory",
    )
    run_p.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable default provenance capture",
    )
    run_p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    run_p.add_argument(
        "--scripted-response",
        action="append",
        default=[],
        help="Mock LLM response; repeat for a network-free run",
    )
    run_p.add_argument(
        "--pretty",
        action="store_true",
        help="Render the run as a live human-friendly trace (boxed header, per-step reasoning + result summary, colored).",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable completion object",
    )
    run_p.set_defaults(_handler=cmd_run_workspace)

    portab_p = sub.add_parser(
        "portability",
        help="Report a cartridge's cross-runtime portability (static analysis)",
        description=(
            "Statically classify every component of a cartridge as "
            "protocol-portable (config/prompts/MCP tools/LEP hooks), "
            "stdlib-declarative (builtin_tools/builtin_hooks), or "
            "Python-pinned (Python tool bodies, class hooks, resources). "
            "Prints an overall portable / python-host profile verdict and "
            "names the exact components that pin the cartridge to a Python "
            "host. No env vars required - reads the cartridge directory only."
        ),
    )
    portab_p.add_argument("cartridge", type=Path, help="Path to a cartridge directory")
    portab_p.add_argument("--json", action="store_true", help="Emit the report as JSON")
    portab_p.add_argument(
        "--require-portable",
        action="store_true",
        help="Exit non-zero (2) if the cartridge is not in the portable profile (CI gate).",
    )
    portab_p.set_defaults(_handler=cmd_portability)


__all__ = ["add_subparsers", "cmd_new", "cmd_portability", "cmd_run_workspace"]
