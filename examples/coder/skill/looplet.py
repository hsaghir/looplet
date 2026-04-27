"""Runnable skill bundle for the looplet coder example."""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

from looplet import (
    CallableMemorySource,
    Conversation,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    OpenAIBackend,
    StaticMemorySource,
    StreamingHook,
    TrajectoryRecorder,
    composable_loop,
    probe_native_tool_support,
)
from looplet.compact import PruneToolResults, TruncateCompact, compact_chain
from looplet.limits import PerToolLimitHook
from looplet.presets import AgentPreset
from looplet.provenance import RecordingLLMBackend
from looplet.resilient import ResilientBackend
from looplet.session import SessionLog
from looplet.stagnation import StagnationHook, tool_call_fingerprint
from looplet.streaming import CallbackEmitter


def _coder() -> Any:
    agent_path = Path(__file__).resolve().parent.parent / "agent.py"
    module_name = "examples.coder.agent"
    module = _loaded_agent_module(module_name, agent_path)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load coder agent from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _loaded_agent_module(module_name: str, agent_path: Path) -> ModuleType | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    try:
        if Path(module_file).resolve() == agent_path:
            return module
    except OSError:
        return None
    return None


def scripted_responses() -> list[str]:
    """Return the deterministic coder demo responses."""
    coder = _coder()
    return coder.scripted_responses()


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
    """Run the coder cartridge with byte-for-byte-compatible terminal output."""
    coder = _coder()
    workspace_str = os.path.abspath(os.fspath(workspace))
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "x")
    model = os.environ.get("OPENAI_MODEL", "llama3.1")

    if scripted or scripted_responses:
        llm = MockLLMBackend(responses=scripted_responses or coder.scripted_responses())
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
    file_cache = coder.FileCache(workspace_str)
    tools = coder.make_tools(workspace_str, file_cache)

    hooks: list[Any] = []
    if require_tests:
        hooks.append(coder.TestGuardHook())
    hooks.append(coder.FileCacheHook(file_cache))
    hooks.append(coder.StaleFileHook(file_cache))
    hooks.append(coder.LinterHook(workspace_str))
    hooks.append(
        StagnationHook(
            fingerprint=tool_call_fingerprint,
            threshold=4,
            nudge="[stagnation] Re-read the file, try a different approach, or think().",
        )
    )
    hooks.append(coder.PerToolLimitHook(default_limit=25, limits={"bash": 20, "read_file": 20}))
    events: list[Any] = []
    hooks.append(StreamingHook(CallbackEmitter(events.append)))

    instructions = coder._discover_instructions(workspace_str)
    project_ctx = coder._project_context(workspace_str)
    memory_sources: list[Any] = []
    if instructions:
        memory_sources.append(StaticMemorySource(instructions))
    memory_sources.append(
        CallableMemorySource(
            lambda state: f"[{project_ctx}] step {getattr(state, 'step_count', 0)}/{max_steps}"
        )
    )

    config = LoopConfig(
        max_steps=max_steps,
        temperature=0.2,
        system_prompt=coder.SYSTEM_PROMPT,
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
    print(
        f"\n  Steps: {len(state.steps)} | LLM calls: {call_count} ({scoped_count} tool-internal)\n"
    )
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
    coder = _coder()
    workspace = str(Path(runtime.workspace).resolve())
    max_steps = runtime.max_steps
    file_cache = coder.FileCache(workspace)
    tools = coder.make_tools(workspace, file_cache)

    hooks: list[Any] = []
    if bool(runtime.option("require_tests", True)):
        hooks.append(coder.TestGuardHook())
    hooks.append(coder.FileCacheHook(file_cache))
    hooks.append(coder.StaleFileHook(file_cache))
    hooks.append(coder.LinterHook(workspace))
    hooks.append(
        StagnationHook(
            fingerprint=tool_call_fingerprint,
            threshold=int(runtime.option("stagnation_threshold", 4)),
            nudge="[stagnation] Re-read the file, try a different approach, or think().",
        )
    )
    hooks.append(PerToolLimitHook(default_limit=25, limits={"bash": 20, "read_file": 20}))

    events = runtime.option("events", [])
    if isinstance(events, list):
        hooks.append(StreamingHook(CallbackEmitter(events.append)))

    instructions = coder._discover_instructions(workspace)
    project_ctx = coder._project_context(workspace)
    memory_sources: list[Any] = []
    if instructions:
        memory_sources.append(StaticMemorySource(instructions))
    memory_sources.append(
        CallableMemorySource(
            lambda state: f"[{project_ctx}] step {getattr(state, 'step_count', 0)}/{max_steps}"
        )
    )

    config = LoopConfig(
        max_steps=max_steps,
        temperature=0.2,
        system_prompt=coder.SYSTEM_PROMPT,
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
