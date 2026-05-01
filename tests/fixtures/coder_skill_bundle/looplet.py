"""Runnable skill bundle for the looplet coder example.

The bundle is intentionally thin: it loads the same composition
helpers as ``examples/coder/agent.py`` and delegates to them. That
means the bundle and the library entrypoint are *literally* the
same agent, configured identically. To change behavior, edit
``examples/coder/wiring.py`` once.

Loading note: bundles ship as plain files inside an installed
package (``site-packages/examples/coder/skill/``). A downstream user
may already have their own ``examples`` namespace package on
``sys.path`` that shadows ours, so we resolve the three sibling
modules (``tools.py``, ``hooks.py``, ``wiring.py``) by absolute file
path rather than relying on ``import examples.coder.*``. See
``test_distributions_include_coder_bundle_and_dependency``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

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
from looplet.presets import AgentPreset
from looplet.provenance import RecordingLLMBackend
from looplet.resilient import ResilientBackend
from looplet.session import SessionLog


def _ensure_parent_package() -> None:
    """Make ``examples.coder`` importable even when ``examples`` is shadowed.

    A downstream user may already have an ``examples`` package on
    sys.path that doesn't contain ``coder``. Synthesize the parent so
    ``from examples.coder.tools import FileCache`` works inside our
    sibling modules below.
    """
    parent = Path(__file__).resolve().parent  # bundle dir
    pkg_name = "examples.coder"
    if pkg_name in sys.modules:
        return
    init_file = parent / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_file,
        submodule_search_locations=[str(parent)],
    )
    if spec is None or spec.loader is None:
        return
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)


def _load_sibling(name: str) -> ModuleType:
    """Load ``examples/coder/<name>.py`` by file path, regardless of sys.path."""
    module_name = f"examples.coder.{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    sibling = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, sibling)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load coder bundle module {sibling}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_ensure_parent_package()
_tools = _load_sibling("tools")
_hooks = _load_sibling("hooks")  # noqa: F841 — preloaded so wiring import works
_wiring = _load_sibling("wiring")

FileCache = _tools.FileCache
make_tools = _tools.make_tools
SYSTEM_PROMPT = _wiring.SYSTEM_PROMPT
build_default_hooks = _wiring.build_default_hooks
build_default_memory_sources = _wiring.build_default_memory_sources
build_eval_hook = _wiring.build_eval_hook
_discover_instructions = _wiring._discover_instructions
_project_context = _wiring._project_context
_scripted = _wiring.scripted_responses


def scripted_responses() -> list[str]:
    """Return the deterministic coder demo responses."""
    return _scripted()


def render_step(step: Any) -> str:
    """Render one coder step for the generic bundle runner."""
    tool_name = step.tool_call.tool
    error = step.tool_result.error
    data = step.tool_result.data or {}
    if tool_name == "done":
        return f"\n  Done: {data.get('summary', data.get('status', ''))[:120]}"
    if tool_name == "think":
        return f"  think #{step.number}: {step.tool_call.args.get('analysis', '')[:100]}"
    if tool_name == "bash":
        status = "ok" if data.get("exit_code") == 0 else "fail"
        command = step.tool_call.args.get("command", "")[:60]
        return f"  {status} #{step.number} bash: {command} [exit {data.get('exit_code', '?')}]"
    if tool_name == "read_file":
        return (
            f"  read #{step.number}: {step.tool_call.args.get('file_path', '?')} "
            f"({data.get('total_lines', '?')} lines)"
        )
    if tool_name == "write_file":
        return (
            f"  write #{step.number}: {data.get('written', '?')} ({data.get('lines', '?')} lines)"
        )
    if tool_name == "edit_file":
        suffix = "ok" if not error else f"error: {str(error)[:50]}"
        return f"  edit #{step.number}: {step.tool_call.args.get('file_path', '?')} {suffix}"
    if tool_name == "list_dir":
        return f"  list_dir #{step.number}: {data.get('count', '?')} entries"
    if tool_name == "glob":
        return f"  glob #{step.number}: {len(data.get('matches', []))} files"
    if tool_name == "grep":
        return f"  grep #{step.number}: {data.get('count', '?')} matches"
    return f"  {tool_name} #{step.number}: {'error' if error else 'ok'}"


def run(
    *,
    task: str,
    workspace: str | Path,
    max_steps: int,
    scripted: bool,
    scripted_responses: list[str],
    require_tests: bool,
    trace_dir: str | Path | None,
    provenance: bool,
) -> int:
    """Run the coder bundle with byte-for-byte-compatible terminal output."""
    workspace_str = os.path.abspath(os.fspath(workspace))
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "x")
    model = os.environ.get("OPENAI_MODEL", "llama3.1")

    if scripted or scripted_responses:
        llm = MockLLMBackend(responses=scripted_responses or _scripted())
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
    file_cache = FileCache(workspace_str)
    tools = make_tools(workspace_str, file_cache)

    hooks: list[Any] = build_default_hooks(
        workspace_str,
        file_cache,
        require_tests=require_tests,
    )
    eval_hook = build_eval_hook(workspace_str)
    hooks.append(eval_hook)

    instructions = _discover_instructions(workspace_str)
    project_ctx = _project_context(workspace_str)
    memory_sources: list[Any] = build_default_memory_sources(workspace_str, max_steps)

    config = LoopConfig(
        max_steps=max_steps,
        temperature=0.2,
        system_prompt=SYSTEM_PROMPT,
        compact_service=compact_chain(
            PruneToolResults(keep_recent=10), TruncateCompact(keep_recent=5)
        ),
        memory_sources=memory_sources,
        use_native_tools=protocol_probe.supported,
    )
    state = DefaultState(max_steps=max_steps)
    session_log = SessionLog()
    conv = Conversation()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              looplet coder                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Task: {task}")
    print(f"  Workspace: {workspace_str}")
    print(f"  Context: {project_ctx}")
    if instructions:
        print(f"  Instructions: {len(instructions)} chars")
    print(f"  Model: {model_label} | Budget: {max_steps} steps")
    print(f"  Tool protocol: {'native' if protocol_probe.supported else 'json-text'}")
    print(f"  Probe: {protocol_probe.reason}\n")

    effective_trace_dir = None
    if provenance:
        effective_trace_dir = trace_dir or (
            Path(workspace_str) / ".looplet" / "traces" / f"coder-{uuid.uuid4().hex[:12]}"
        )
    _run_loop(
        recording,
        trace_dir=effective_trace_dir,
        hooks=hooks,
        task=task,
        tools=tools,
        state=state,
        config=config,
        session_log=session_log,
        conversation=conv,
    )
    calls = getattr(recording, "calls", [])
    if isinstance(calls, int):
        call_count = calls
        scoped_count = 0
    else:
        call_count = len(calls)
        scoped_count = len([call for call in calls if getattr(call, "scope", None)])
    print(f"\n  Steps: {len(state.steps)} | LLM calls: {call_count} ({scoped_count} tool-internal)")
    if eval_hook.results:
        print("  Evals:")
        for r in eval_hook.results:
            print(f"    {r.pretty()}")
    print()
    return 0


def _run_loop(
    recording: Any,
    *,
    trace_dir: str | Path | None,
    hooks: list[Any],
    task: str,
    tools: Any,
    state: Any,
    config: Any,
    session_log: Any,
    conversation: Any,
) -> None:
    run_hooks = list(hooks)
    if trace_dir is not None:
        run_hooks.append(TrajectoryRecorder(recording_llm=recording, output_dir=trace_dir))
    for step in composable_loop(
        llm=recording,
        task={"description": task},
        tools=tools,
        state=state,
        config=config,
        hooks=run_hooks,
        session_log=session_log,
        conversation=conversation,
    ):
        print(_render_original_step(step))


def _render_original_step(step: Any) -> str:
    tool_name = step.tool_call.tool
    error = step.tool_result.error
    data = step.tool_result.data or {}
    if tool_name == "done":
        return f"\n  ✓ Done: {data.get('summary', data.get('status', ''))[:120]}"
    if tool_name == "think":
        return f"  💭 #{step.number} {step.tool_call.args.get('analysis', '')[:100]}..."
    if tool_name == "bash":
        return (
            f"  {'✓' if data.get('exit_code') == 0 else '✗'} #{step.number} bash: "
            f"{step.tool_call.args.get('command', '')[:60]}  [exit {data.get('exit_code', '?')}]"
        )
    if tool_name == "read_file":
        return (
            f"  📖 #{step.number} read: {step.tool_call.args.get('file_path', '?')} "
            f"({data.get('total_lines', '?')} lines)"
        )
    if tool_name == "write_file":
        return f"  ✏️  #{step.number} write: {data.get('written', '?')} ({data.get('lines', '?')} lines)"
    if tool_name == "edit_file":
        return (
            f"  {'✏️ ' if not error else '✗ '}#{step.number} edit: "
            f"{step.tool_call.args.get('file_path', '?')}"
            f"{' ✓' if not error else ' — ' + str(error)[:50]}"
        )
    if tool_name == "list_dir":
        return f"  📂 #{step.number} list_dir: {data.get('count', '?')} entries"
    if tool_name == "glob":
        return f"  🔍 #{step.number} glob: {len(data.get('matches', []))} files"
    if tool_name == "grep":
        return f"  🔍 #{step.number} grep: {data.get('count', '?')} matches"
    return f"  → #{step.number} {tool_name}"


def build(runtime: Any) -> AgentPreset:
    """Build the coder agent as normal looplet primitives."""
    workspace = str(Path(runtime.workspace).resolve())
    max_steps = runtime.max_steps
    file_cache = FileCache(workspace)
    tools = make_tools(workspace, file_cache)

    events = runtime.option("events", [])
    hooks: list[Any] = build_default_hooks(
        workspace,
        file_cache,
        require_tests=bool(runtime.option("require_tests", True)),
        test_strict=bool(runtime.option("test_strict", False)),
        stagnation_threshold=int(runtime.option("stagnation_threshold", 6)),
        per_tool_limit=int(runtime.option("per_tool_limit", 100)),
        events=events if isinstance(events, list) else None,
    )
    if bool(runtime.option("eval_hook", True)):
        hooks.append(build_eval_hook(workspace))

    memory_sources = build_default_memory_sources(workspace, max_steps)

    config = LoopConfig(
        max_steps=max_steps,
        temperature=0.2,
        system_prompt=SYSTEM_PROMPT,
        compact_service=compact_chain(
            PruneToolResults(keep_recent=10),
            TruncateCompact(keep_recent=5),
        ),
        memory_sources=memory_sources,
        use_native_tools=bool(runtime.option("use_native_tools", False)),
    )

    return AgentPreset(
        config=config,
        hooks=hooks,
        tools=tools,
        state=DefaultState(max_steps=max_steps),
    )
