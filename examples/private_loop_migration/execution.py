"""Offline raw, owned-loop, and replay execution for the migration recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from looplet import (
    BaseToolRegistry,
    DefaultState,
    EvalHook,
    LoopConfig,
    ProvenanceSink,
    ToolCall,
    composable_loop,
    replay_loop,
    tools_from,
)
from looplet.testing import MockLLMBackend

from .handoff_contract import live_task
from .handoff_tools import ToolSuite

DONE_TOOL = "done"
MAX_STEPS = 4
SYSTEM_PROMPT = "You hand an on-call shift over. Record the work the next owner still carries."

MODEL_RESPONSES = [
    json.dumps(
        {
            "tool": "list_incidents",
            "args": {},
            "reasoning": "read the shift log before writing anything",
        }
    ),
    json.dumps(
        {
            "tool": "publish_handoff",
            "args": {"owner": "day-shift"},
            "reasoning": "write the handoff file",
        }
    ),
    json.dumps(
        {
            "tool": DONE_TOOL,
            "args": {"summary": "handoff file written for day-shift"},
            "reasoning": "the requested file exists",
        }
    ),
]


def build_registry(suite: ToolSuite) -> BaseToolRegistry:
    """Build the one registry reused across the pre-fix stages of a run."""
    return tools_from(list(suite.callables.values()), include_done=True)


def run_raw_loop(
    suite: ToolSuite,
    *,
    registry: BaseToolRegistry | None = None,
    workspace: str | Path | None = None,
) -> tuple[str, ...]:
    """Run the private while-loop through the supplied registry."""
    llm = _scripted_backend()
    active_registry = registry or build_registry(suite)
    suite.bind_workspace(workspace or suite.workspace)
    task = live_task()
    decisions: list[str] = []
    observations: list[str] = []
    for _ in range(MAX_STEPS):
        response = llm.generate(_raw_prompt(task, observations), system_prompt=SYSTEM_PROMPT)
        call = _parse_call(response)
        decisions.append(call.tool)
        if call.tool == DONE_TOOL:
            break
        result = active_registry.dispatch(call)
        if result.error is not None:
            raise RuntimeError(f"raw loop tool {call.tool!r} failed: {result.error}")
        observations.append(f"{call.tool} -> {json.dumps(result.data)}")
    return tuple(decisions)


def run_owned_loop(
    suite: ToolSuite,
    *,
    registry: BaseToolRegistry | None = None,
    workspace: str | Path | None = None,
    sink: ProvenanceSink | None = None,
    eval_hook: EvalHook | None = None,
) -> tuple[str, ...]:
    """Run the same tools through ``composable_loop()``."""
    llm: Any = _scripted_backend()
    hooks: list[Any] = []
    if sink is not None:
        llm = sink.wrap_llm(llm)
        hooks.append(sink.trajectory_hook())
    if eval_hook is not None:
        hooks.append(eval_hook)
    active_registry = registry or build_registry(suite)
    suite.bind_workspace(workspace or suite.workspace)
    steps = list(
        composable_loop(
            llm=llm,
            tools=active_registry,
            hooks=hooks,
            task=live_task(),
            config=_loop_config(),
            state=DefaultState(max_steps=MAX_STEPS),
        )
    )
    if sink is not None:
        sink.flush()
    return tuple(step.tool_call.tool for step in steps)


def replay_with_fixed_tools(
    suite: ToolSuite,
    trace_dir: Path,
    eval_hook: EvalHook,
    *,
    registry: BaseToolRegistry | None = None,
    workspace: str | Path | None = None,
) -> tuple[str, ...]:
    """Re-execute recorded decisions against the fixed implementation."""
    active_registry = registry or build_registry(suite)
    suite.bind_workspace(workspace or suite.workspace)
    steps = list(
        replay_loop(
            trace_dir,
            tools=active_registry,
            state=DefaultState(max_steps=MAX_STEPS),
            hooks=[eval_hook],
            config=_loop_config(),
            task=live_task(),
        )
    )
    return tuple(step.tool_call.tool for step in steps)


def _scripted_backend() -> MockLLMBackend:
    return MockLLMBackend(responses=list(MODEL_RESPONSES), cycle=False)


def _parse_call(response: str) -> ToolCall:
    payload = json.loads(response)
    return ToolCall(
        tool=str(payload["tool"]),
        args=dict(payload.get("args", {})),
        reasoning=str(payload.get("reasoning", "")),
    )


def _loop_config() -> LoopConfig:
    return LoopConfig(max_steps=MAX_STEPS, use_native_tools=True, system_prompt=SYSTEM_PROMPT)


def _raw_prompt(task: dict[str, Any], observations: list[str]) -> str:
    lines = [
        f"goal: {task['goal']}",
        "",
        "tools: list_incidents(), publish_handoff(owner), done(summary)",
    ]
    if observations:
        lines += ["", "observations:", *observations]
    lines += ["", 'Reply with {"tool": ..., "args": {...}}.']
    return "\n".join(lines)
