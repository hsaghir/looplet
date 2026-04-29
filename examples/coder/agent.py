#!/usr/bin/env python3
"""looplet coder — CLI entrypoint.

This file is intentionally small. It hosts the command-line wrapper
and re-exports the building blocks defined in the sibling modules:

* :mod:`examples.coder.tools`   — file/bash tools and ``FileCache``
* :mod:`examples.coder.hooks`   — observability hooks
* :mod:`examples.coder.wiring`  — composition (system prompt,
  default hook stack, memory sources, eval hook)

The CLI calls the same helpers that the runnable cartridge in
``examples/coder/skill/`` calls, so ``python examples/coder/agent.py
"…"`` and ``python -m looplet run examples/coder/skill "…"`` produce
the same behavior. Modify behavior in :mod:`wiring` and both the
library and the cartridge pick up the change.

Re-exports below preserve back-compat for callers that imported
``FileCache``, ``LinterHook``, ``TestGuardHook``, etc. from
``examples.coder.agent`` before the split.

Usage::

    python examples/coder/agent.py "Add type hints to utils.py"
    python examples/coder/agent.py --workspace /path/to/proj "..."
    python examples/coder/agent.py "Create add() with tests" --scripted
    OPENAI_BASE_URL=http://localhost:11434/v1 python examples/coder/agent.py "..."
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Re-exports — keep the public surface stable for existing importers.
from examples.coder.hooks import (  # noqa: F401
    FileCacheHook,
    LinterHook,
    StaleFileHook,
    TestGuardHook,
)
from examples.coder.tools import (  # noqa: F401
    _EXIT_CODE_MAP,
    FileCache,
    _fuzzy_find,
    _interpret_exit_code,
    _is_path_inside,
    _resolve_safe_path,
    _run,
    make_tools,
)
from examples.coder.wiring import (  # noqa: F401
    SYSTEM_PROMPT,
    _discover_instructions,
    _project_context,
    _tool_call,
    build_default_hooks,
    build_default_memory_sources,
    build_eval_hook,
    make_test_collector,
    scripted_responses,
)
from looplet import (
    Conversation,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    OpenAIBackend,
    TrajectoryRecorder,
    composable_loop,
    probe_native_tool_support,
)
from looplet.compact import PruneToolResults, TruncateCompact, compact_chain
from looplet.provenance import RecordingLLMBackend
from looplet.resilient import ResilientBackend
from looplet.session import SessionLog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="looplet coder — AI coding agent")
    parser.add_argument("task", help="What to build or fix")
    parser.add_argument("--workspace", "-w", default=os.getcwd(), help="Project directory")
    parser.add_argument("--max-steps", type=int, default=30, help="Max tool calls")
    parser.add_argument("--no-tests", action="store_true", help="Skip test guard")
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run a deterministic local demo with MockLLMBackend instead of a real model.",
    )
    args = parser.parse_args(argv)
    workspace = os.path.abspath(args.workspace)

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "x")
    model = os.environ.get("OPENAI_MODEL", "llama3.1")

    if args.scripted:
        llm = MockLLMBackend(responses=scripted_responses())
        model_label = "scripted MockLLMBackend"
    else:
        llm = ResilientBackend(
            OpenAIBackend(base_url=base_url, api_key=api_key, model=model),
            retries=2,
            timeout_s=120,
        )
        model_label = model
    recording = RecordingLLMBackend(llm)
    protocol_probe = probe_native_tool_support(recording)

    file_cache = FileCache(workspace)
    tools = make_tools(workspace, file_cache)

    hooks = build_default_hooks(
        workspace,
        file_cache,
        require_tests=not args.no_tests,
    )
    eval_hook = build_eval_hook(workspace)
    hooks.append(eval_hook)

    memory_sources = build_default_memory_sources(workspace, args.max_steps)
    instructions = _discover_instructions(workspace)
    project_ctx = _project_context(workspace)

    config = LoopConfig(
        max_steps=args.max_steps,
        temperature=0.2,
        system_prompt=SYSTEM_PROMPT,
        compact_service=compact_chain(
            PruneToolResults(keep_recent=10), TruncateCompact(keep_recent=5)
        ),
        memory_sources=memory_sources,
        use_native_tools=protocol_probe.supported,
    )
    state = DefaultState(max_steps=args.max_steps)
    session_log = SessionLog()
    conv = Conversation()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              looplet coder                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Task: {args.task}")
    print(f"  Workspace: {workspace}")
    print(f"  Context: {project_ctx}")
    if instructions:
        print(f"  Instructions: {len(instructions)} chars")
    print(f"  Model: {model_label} | Budget: {args.max_steps} steps")
    print(f"  Tool protocol: {'native' if protocol_probe.supported else 'json-text'}")
    print(f"  Probe: {protocol_probe.reason}\n")

    with tempfile.TemporaryDirectory() as traj_dir:
        recorder = TrajectoryRecorder(recording_llm=recording, output_dir=traj_dir)
        hooks.append(recorder)

        for step in composable_loop(
            llm=recording,
            task={"description": args.task},
            tools=tools,
            state=state,
            config=config,
            hooks=hooks,
            session_log=session_log,
            conversation=conv,
        ):
            print(_render_step(step))

        scoped = [c for c in recording.calls if c.scope]
        print(
            f"\n  Steps: {len(state.steps)} | LLM calls: {len(recording.calls)} "
            f"({len(scoped)} tool-internal)"
        )
        if eval_hook.results:
            print("  Evals:")
            for result in eval_hook.results:
                print(f"    {result.pretty()}")
        print()
    return 0


def _render_step(step) -> str:
    """Format a single composable_loop step for the CLI."""
    tool_name = step.tool_call.tool
    err = step.tool_result.error
    data = step.tool_result.data or {}
    if tool_name == "done":
        return f"\n  ✓ Done: {data.get('summary', data.get('status', ''))[:120]}"
    if tool_name == "think":
        return f"  💭 #{step.number} {step.tool_call.args.get('analysis', '')[:100]}..."
    if tool_name == "bash":
        marker = "✓" if data.get("exit_code") == 0 else "✗"
        cmd = step.tool_call.args.get("command", "")[:60]
        return f"  {marker} #{step.number} bash: {cmd}  [exit {data.get('exit_code', '?')}]"
    if tool_name == "read_file":
        return (
            f"  📖 #{step.number} read: {step.tool_call.args.get('file_path', '?')} "
            f"({data.get('total_lines', '?')} lines)"
        )
    if tool_name == "write_file":
        return f"  ✏️  #{step.number} write: {data.get('written', '?')} ({data.get('lines', '?')} lines)"
    if tool_name == "edit_file":
        suffix = " ✓" if not err else f" — {str(err)[:50]}"
        return (
            f"  {'✏️ ' if not err else '✗ '}#{step.number} edit: "
            f"{step.tool_call.args.get('file_path', '?')}{suffix}"
        )
    if tool_name == "list_dir":
        return f"  📂 #{step.number} list_dir: {data.get('count', '?')} entries"
    if tool_name == "glob":
        return f"  🔍 #{step.number} glob: {len(data.get('matches', []))} files"
    if tool_name == "grep":
        return f"  🔍 #{step.number} grep: {data.get('count', '?')} matches"
    return f"  → #{step.number} {tool_name}"


if __name__ == "__main__":
    sys.exit(main())
