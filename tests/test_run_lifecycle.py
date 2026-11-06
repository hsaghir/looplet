"""Tests for the shared run lifecycle transition policy."""

from __future__ import annotations

import pytest

from looplet import (
    InvalidRunTransition,
    RunPhase,
    RunStatus,
    validate_run_transition,
)


def test_valid_run_transition_chain() -> None:
    status = RunStatus.CREATED
    phase = RunPhase.STARTING
    transitions = [
        (RunStatus.RUNNING, RunPhase.PROMPTING),
        (RunStatus.RUNNING, RunPhase.LLM),
        (RunStatus.RUNNING, RunPhase.DISPATCHING),
        (RunStatus.RUNNING, RunPhase.FINALIZING),
        (RunStatus.COMPLETED, RunPhase.TERMINAL),
    ]

    for next_status, next_phase in transitions:
        validate_run_transition(
            status,
            phase,
            next_status=next_status,
            next_phase=next_phase,
        )
        status, phase = next_status, next_phase


@pytest.mark.parametrize(
    ("current_status", "current_phase", "next_status", "next_phase"),
    [
        (RunStatus.CREATED, RunPhase.STARTING, RunStatus.COMPLETED, RunPhase.TERMINAL),
        (RunStatus.RUNNING, RunPhase.LLM, RunStatus.RUNNING, RunPhase.STARTING),
        (RunStatus.COMPLETED, RunPhase.TERMINAL, RunStatus.RUNNING, RunPhase.PROMPTING),
        (RunStatus.RUNNING, RunPhase.TERMINAL, RunStatus.RUNNING, RunPhase.TERMINAL),
    ],
)
def test_invalid_run_transition_is_explicit(
    current_status: RunStatus,
    current_phase: RunPhase,
    next_status: RunStatus,
    next_phase: RunPhase,
) -> None:
    with pytest.raises(InvalidRunTransition):
        validate_run_transition(
            current_status,
            current_phase,
            next_status=next_status,
            next_phase=next_phase,
        )


def test_terminal_status_requires_terminal_phase() -> None:
    with pytest.raises(InvalidRunTransition):
        validate_run_transition(
            RunStatus.RUNNING,
            RunPhase.LLM,
            next_status=RunStatus.STOPPED,
            next_phase=RunPhase.STOPPING,
        )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_yield_boundary_records_history_and_checkpoint_before_early_close(
    tmp_path,
    asynchronous: bool,
) -> None:
    from looplet import (
        AsyncMockLLMBackend,
        BaseToolRegistry,
        DefaultState,
        FileCheckpointStore,
        LifecycleEvent,
        LoopConfig,
        MockLLMBackend,
        SessionLog,
        ToolSpec,
        async_composable_loop,
        composable_loop,
        register_done_tool,
    )

    events: list[tuple[str, str | None]] = []
    cleanup_calls = 0

    class LifecycleObserver:
        def on_event(self, payload):
            if payload.event is LifecycleEvent.STOP:
                events.append(("stop", payload.termination_reason))

        def on_loop_end(self, state, session_log, context, llm):
            nonlocal cleanup_calls
            cleanup_calls += 1

    tools = BaseToolRegistry()
    register_done_tool(tools)
    tools.register(ToolSpec("ping", "Ping", {}, lambda: {"pong": True}))
    ckpt_dir = tmp_path / ("ckpt_async" if asynchronous else "ckpt_sync")
    state = DefaultState(max_steps=3)
    log = SessionLog()
    config = LoopConfig(
        max_steps=3,
        use_native_tools=False,
        checkpoint_dir=str(ckpt_dir),
    )
    responses = [
        '{"tool":"ping","args":{}}',
        '{"tool":"done","args":{"summary":"ok"}}',
    ]

    if asynchronous:
        iterator = async_composable_loop(
            llm=AsyncMockLLMBackend(responses=responses),
            tools=tools,
            state=state,
            session_log=log,
            config=config,
            hooks=[LifecycleObserver()],
            task={"goal": "ping"},
        )
        step = await iterator.__anext__()
        entries_at_yield = len(log.entries)
        ckpt_at_yield = FileCheckpointStore(ckpt_dir).load("step_1")
        await iterator.aclose()
    else:
        iterator = composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=tools,
            state=state,
            session_log=log,
            config=config,
            hooks=[LifecycleObserver()],
            task={"goal": "ping"},
        )
        step = next(iterator)
        entries_at_yield = len(log.entries)
        ckpt_at_yield = FileCheckpointStore(ckpt_dir).load("step_1")
        iterator.close()

    assert step.tool_call.tool == "ping"
    assert entries_at_yield == 1
    assert ckpt_at_yield is not None
    assert ckpt_at_yield.step_number == 1
    assert cleanup_calls == 1
    assert events == [("stop", "stream_closed")]
    assert getattr(state, "run_status", None) == "stopped"
    assert getattr(state, "termination_reason", None) == "stream_closed"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_close_at_accepted_done_preserves_completed_status_and_checkpoint(
    tmp_path,
    asynchronous: bool,
) -> None:
    from looplet import (
        AsyncMockLLMBackend,
        BaseToolRegistry,
        DefaultState,
        FileCheckpointStore,
        LoopConfig,
        MockLLMBackend,
        SessionLog,
        async_composable_loop,
        composable_loop,
        register_done_tool,
    )

    tools = BaseToolRegistry()
    register_done_tool(tools)
    ckpt_dir = tmp_path / ("done_async" if asynchronous else "done_sync")
    state = DefaultState(max_steps=2)
    log = SessionLog()
    config = LoopConfig(
        max_steps=2,
        use_native_tools=False,
        checkpoint_dir=str(ckpt_dir),
    )
    responses = ['{"tool":"done","args":{"summary":"verified"}}']

    if asynchronous:
        iterator = async_composable_loop(
            llm=AsyncMockLLMBackend(responses=responses),
            tools=tools,
            state=state,
            session_log=log,
            config=config,
            task={},
        )
        step = await iterator.__anext__()
        status_at_yield = getattr(state, "run_status", None)
        ckpt_at_yield = FileCheckpointStore(ckpt_dir).load("step_1_done")
        await iterator.aclose()
    else:
        iterator = composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=tools,
            state=state,
            session_log=log,
            config=config,
            task={},
        )
        step = next(iterator)
        status_at_yield = getattr(state, "run_status", None)
        ckpt_at_yield = FileCheckpointStore(ckpt_dir).load("step_1_done")
        iterator.close()

    assert step.tool_call.tool == "done"
    assert status_at_yield == "completed"
    assert ckpt_at_yield is not None
    assert ckpt_at_yield.metadata.get("status") == "done"
    assert getattr(state, "run_status", None) == "completed"
    assert getattr(state, "termination_reason", None) == "done"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_terminal_tool_error_and_post_dispatch_rejection_normalize_reason(
    asynchronous: bool,
) -> None:
    from looplet import (
        AsyncMockLLMBackend,
        BaseToolRegistry,
        DefaultState,
        HookDecision,
        LoopConfig,
        MockLLMBackend,
        ToolResult,
        ToolSpec,
        async_composable_loop,
        composable_loop,
    )
    from looplet.done_steps import last_rejected_done

    attempts = 0

    def submit(answer: str) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("evidence missing")
        return {"accepted": True, "answer": answer}

    class RejectSecondAttemptHook:
        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            if tool_call.tool == "submit" and step_num == 2:
                return HookDecision(
                    updated_result=ToolResult(
                        tool="submit",
                        args_summary="rejected",
                        data=None,
                        error="post_dispatch verification failed",
                    )
                )
            return None

    tools = BaseToolRegistry()
    tools.register(ToolSpec("submit", "Submit final answer", {"answer": "str"}, submit))
    config = LoopConfig(
        max_steps=4,
        use_native_tools=False,
        done_tool="submit",
    )
    state = DefaultState(max_steps=4)
    responses = [
        '{"tool":"submit","args":{"answer":"draft-1"}}',
        '{"tool":"submit","args":{"answer":"draft-2"}}',
        '{"tool":"submit","args":{"answer":"final"}}',
    ]

    if asynchronous:
        steps = [
            s
            async for s in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=responses),
                tools=tools,
                state=state,
                hooks=[RejectSecondAttemptHook()],
                config=config,
                task={},
            )
        ]
    else:
        steps = list(
            composable_loop(
                llm=MockLLMBackend(responses=responses),
                tools=tools,
                state=state,
                hooks=[RejectSecondAttemptHook()],
                config=config,
                task={},
            )
        )

    assert len(steps) == 3
    assert "evidence missing" in (steps[0].tool_result.error or "")
    assert steps[0].tool_result.data["rejected"] is True
    assert "evidence missing" in steps[0].tool_result.data["reason"]
    assert steps[0].metadata.get("terminal_accepted") is False
    assert steps[1].tool_result.error == "post_dispatch verification failed"
    assert steps[1].tool_result.data == {
        "rejected": True,
        "reason": "post_dispatch verification failed",
    }
    assert steps[1].metadata.get("terminal_accepted") is False
    rejected = last_rejected_done(state, tool_name="submit")
    assert rejected == steps[1]
    assert steps[2].tool_result.error is None
    assert steps[2].metadata.get("terminal_accepted") is True
    assert getattr(state, "run_status", None) == "completed"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_portable_async_tool_uses_ctx_async_llm_in_both_loops(
    tmp_path,
    asynchronous: bool,
) -> None:
    from looplet import (
        AsyncMockLLMBackend,
        BaseToolRegistry,
        DefaultState,
        LoopConfig,
        MockLLMBackend,
        ProvenanceSink,
        ToolSpec,
        async_composable_loop,
        composable_loop,
        register_done_tool,
    )

    async def summarize_with_async_llm(ctx) -> dict:
        text = await ctx.async_llm.generate("Summarize this evidence")
        return {"summary_text": text}

    tools = BaseToolRegistry()
    register_done_tool(tools)
    tools.register(ToolSpec("summarize", "Summarize", {}, summarize_with_async_llm))
    sink = ProvenanceSink(tmp_path / ("trace_async" if asynchronous else "trace_sync"))
    responses = [
        '{"tool":"summarize","args":{}}',
        "concise helper summary",
        '{"tool":"done","args":{"summary":"complete"}}',
    ]
    state = DefaultState(max_steps=3)
    config = LoopConfig(max_steps=3, use_native_tools=False)

    if asynchronous:
        llm = sink.wrap_llm(AsyncMockLLMBackend(responses=responses))
        steps = [
            s
            async for s in async_composable_loop(
                llm=llm,
                tools=tools,
                state=state,
                config=config,
                hooks=[sink.trajectory_hook()],
                task={},
            )
        ]
    else:
        llm = sink.wrap_llm(MockLLMBackend(responses=responses))
        steps = list(
            composable_loop(
                llm=llm,
                tools=tools,
                state=state,
                config=config,
                hooks=[sink.trajectory_hook()],
                task={},
            )
        )

    assert steps[0].tool_result.error is None
    assert steps[0].tool_result.data == {"summary_text": "concise helper summary"}
    assert [call.scope for call in llm.calls] == [None, "tool:summarize", None]
