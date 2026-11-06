"""Tests for Phase-2 SDK & framework improvements:
1. OpenAIResponsesBackend & AsyncOpenAIResponsesBackend (with previous_response_id & 404 fallback)
2. Wire-level stream_with_tools & LoopConfig(streaming_tool_execution=True)
3. Prompt-prefix stability (CachePolicy.stable / stable_prefix=True)
4. DeferredToolViewHook & search_tools for large tool catalogs
5. Multimodal image ContentBlock translation across OpenAI Chat, Responses, and Anthropic
6. Process-group subprocess helper (run_subprocess_tool) with timeout, CancelToken, and output spill
7. Session branching from checkpoint (branch_from_checkpoint & ExecutionSession.branch_checkpoint)
8. Eval expected_graders manifest deletion detection & replay_loop(stub_tools=...)
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from looplet import (
    AnthropicBackend,
    AsyncOpenAIResponsesBackend,
    CachePolicy,
    CancelToken,
    Conversation,
    DefaultState,
    DeferredToolViewHook,
    EvalContext,
    EvalReport,
    EvalResult,
    EvalRunRecord,
    ExecutionSession,
    FileCheckpointStore,
    LoopConfig,
    Message,
    MockLLMBackend,
    OpenAIBackend,
    OpenAIResponsesBackend,
    ProvenanceSink,
    RunEnvelope,
    RunResult,
    ToolCancelledError,
    async_composable_loop,
    branch_from_checkpoint,
    composable_loop,
    eval_assess,
    eval_mark,
    replay_loop,
    run_result,
    run_result_async,
    run_subprocess_tool,
    tool,
    tools_from,
)
from looplet.backends import AsyncOpenAIStreamingBackend, OpenAIStreamingBackend
from looplet.conversation import ContentBlock, MessageRole
from looplet.cost import extract_usage
from looplet.streaming import CallbackEmitter, LLMChunkEvent

# ── 1. OpenAIResponsesBackend & AsyncOpenAIResponsesBackend ─────


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_openai_responses_backend_stateful_chaining_and_404_fallback(
    asynchronous: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    def create_sync(**kwargs: Any) -> Any:
        calls.append(kwargs)
        call_num = len(calls)
        if call_num == 1:
            assert "previous_response_id" not in kwargs
            return SimpleNamespace(
                id="resp_turn_1",
                status="completed",
                usage=SimpleNamespace(
                    input_tokens=120,
                    output_tokens=30,
                    input_tokens_details=SimpleNamespace(cached_tokens=80),
                ),
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_lookup_1",
                        name="lookup",
                        arguments='{"key": "alpha"}',
                    )
                ],
            )
        if call_num == 2:
            # Verify stateful previous_response_id was sent with only delta function_call_output
            assert kwargs.get("previous_response_id") == "resp_turn_1"
            assert any(
                item.get("type") == "function_call_output"
                and item.get("call_id") == "call_lookup_1"
                for item in kwargs["input"]
            )
            # Simulate expired previous_response_id 404 on the server
            raise RuntimeError("Error code: 404 - unknown previous_response_id: resp_turn_1")
        # Call 3 is the automatic stateless fallback retry
        assert "previous_response_id" not in kwargs
        assert any(item.get("type") == "function_call" for item in kwargs["input"])
        assert any(item.get("type") == "function_call_output" for item in kwargs["input"])
        return SimpleNamespace(
            id="resp_turn_2",
            status="completed",
            usage=SimpleNamespace(
                input_tokens=150,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=100),
            ),
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_done_1",
                    name="done",
                    arguments='{"summary": "alpha=42"}',
                )
            ],
        )

    async def create_async(**kwargs: Any) -> Any:
        return create_sync(**kwargs)

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_async if asynchronous else create_sync)
    )
    backend_cls = AsyncOpenAIResponsesBackend if asynchronous else OpenAIResponsesBackend
    llm = backend_cls(client, model="gpt-6-luna")

    @tool(concurrent_safe=True, read_only=True)
    def lookup(key: str) -> dict[str, Any]:
        return {"key": key, "value": 42}

    cfg = LoopConfig(max_steps=4, disabled_defaults=("persistence",))
    if asynchronous:
        res = await run_result_async(
            llm,
            task="Find alpha",
            tools=tools_from([lookup], include_done=True),
            config=cfg,
        )
    else:
        res = run_result(
            llm,
            task="Find alpha",
            tools=tools_from([lookup], include_done=True),
            config=cfg,
        )

    assert res.completed
    assert res.output == {"status": "completed", "summary": "alpha=42"}
    assert len(calls) == 3
    assert llm.previous_response_id == "resp_turn_2"
    assert llm.last_usage == {"input": 150, "output": 20, "cache_read": 100, "cache_write": 0}


def test_extract_usage_supports_responses_input_tokens_details() -> None:
    u = extract_usage(
        {
            "usage": {
                "input_tokens": 500,
                "output_tokens": 120,
                "input_tokens_details": {"cached_tokens": 350},
            }
        }
    )
    assert u == {"input": 500, "output": 120, "cache_read": 350, "cache_write": 0}


# ── 2. Wire-Level stream_with_tools & StreamingToolExecutor ─────


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_loop_streaming_tool_execution_overlaps_tool_with_stream(
    asynchronous: bool,
) -> None:
    tool_started = threading.Event()
    turn_count = 0

    def _make_chunks(turn: int) -> list[Any]:
        if turn == 1:
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content="Thinking...",
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_fast_1",
                                        function=SimpleNamespace(
                                            name="fast_tool", arguments='{"x": 1}'
                                        ),
                                    )
                                ],
                            ),
                        )
                    ]
                ),
                # Second tool starts at index=1 -> triggers immediate execution of index=0!
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="tool_calls",
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=1,
                                        id="call_fast_2",
                                        function=SimpleNamespace(
                                            name="fast_tool", arguments='{"x": 2}'
                                        ),
                                    )
                                ],
                            ),
                        )
                    ]
                ),
            ]
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content="Done!",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_done",
                                    function=SimpleNamespace(
                                        name="done", arguments='{"summary": "both ran"}'
                                    ),
                                )
                            ],
                        ),
                    )
                ]
            )
        ]

    def create_sync(**kwargs: Any) -> Any:
        nonlocal turn_count
        turn_count += 1
        chunks = _make_chunks(turn_count)

        def _gen():
            yield chunks[0]
            if turn_count == 1:
                yield chunks[1]
                # Wait briefly to prove index=0 tool started BEFORE stream generator exits!
                assert tool_started.wait(timeout=2.0), "Tool 0 did not start during stream!"

        return _gen()

    async def create_async(**kwargs: Any) -> Any:
        nonlocal turn_count
        turn_count += 1
        chunks = _make_chunks(turn_count)

        async def _agen():
            yield chunks[0]
            if turn_count == 1:
                yield chunks[1]
                await asyncio.sleep(0.05)
                assert tool_started.is_set()

        return _agen()

    @tool(concurrent_safe=True, read_only=True)
    def fast_tool(x: int) -> dict[str, int]:
        tool_started.set()
        return {"val": x * 10}

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_async if asynchronous else create_sync)
        )
    )
    llm = (
        AsyncOpenAIStreamingBackend(client, model="test")
        if asynchronous
        else OpenAIStreamingBackend(client, model="test")
    )
    events: list[Any] = []
    cfg = LoopConfig(
        max_steps=5,
        concurrent_dispatch=True,
        streaming_tool_execution=True,
        disabled_defaults=("persistence", "compaction", "stagnation", "tool_budget"),
    )
    if asynchronous:
        steps = [
            s
            async for s in async_composable_loop(
                llm,
                task="run fast tools",
                tools=tools_from([fast_tool], include_done=True),
                config=cfg,
                stream=CallbackEmitter(events.append),
            )
        ]
    else:
        steps = list(
            composable_loop(
                llm,
                task="run fast tools",
                tools=tools_from([fast_tool], include_done=True),
                config=cfg,
                stream=CallbackEmitter(events.append),
            )
        )

    assert len(steps) == 3
    assert [s.tool_result.data for s in steps[:2]] == [{"val": 10}, {"val": 20}]
    chunk_events = [e for e in events if isinstance(e, LLMChunkEvent)]
    assert [e.chunk for e in chunk_events] == ["Thinking...", "Done!"]


# ── 3. Prompt-Prefix Stability for Typed History Caching ────────


def test_stable_prefix_cache_policy_keeps_first_user_message_identical() -> None:
    sent_messages: list[list[dict[str, Any]]] = []

    def create(**kwargs: Any) -> Any:
        sent_messages.append(json.loads(json.dumps(kwargs["messages"])))
        step = len(sent_messages)
        if step < 3:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id=f"c_{step}",
                                    function=SimpleNamespace(
                                        name="step_tool", arguments=json.dumps({"n": step})
                                    ),
                                )
                            ],
                        ),
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="c_done",
                                function=SimpleNamespace(
                                    name="done", arguments='{"summary": "cached"}'
                                ),
                            )
                        ],
                    ),
                )
            ]
        )

    class BriefingHook:
        def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> str:
            return f"Dynamic briefing for step {step_num}"

    @tool(concurrent_safe=True)
    def step_tool(n: int) -> dict[str, int]:
        return {"n": n}

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = OpenAIBackend(client, model="test", use_prompt_cache_key=True)
    res = run_result(
        llm,
        task="Test prefix cache stability",
        tools=tools_from([step_tool], include_done=True),
        hooks=[BriefingHook()],
        config=LoopConfig(
            max_steps=5,
            cache_policy=CachePolicy.stable(),
            disabled_defaults=("persistence",),
        ),
    )
    assert res.completed
    assert len(sent_messages) == 3
    # messages[0] is the initial task/memory user message and MUST be byte-identical across all 3 turns!
    first_user_msg = sent_messages[0][0]
    assert first_user_msg["role"] == "user"
    assert sent_messages[1][0] == first_user_msg
    assert sent_messages[2][0] == first_user_msg
    # Dynamic briefing is appended at the tail of messages instead of mutating messages[0]
    assert "Dynamic briefing for step 3" in sent_messages[2][-1]["content"]


# ── 4. DeferredToolViewHook & search_tools ──────────────────────


def test_deferred_tool_view_hook_discovers_and_activates_schemas() -> None:
    @tool(description="Query PostgreSQL billing ledger by customer_id.", concurrent_safe=True)
    def query_billing_ledger(customer_id: str) -> dict[str, Any]:
        return {"customer_id": customer_id, "balance": 125}

    @tool(description="Send Slack notification to channel.")
    def send_slack_alert(channel: str, text: str) -> dict[str, Any]:
        return {"sent": True}

    registry = tools_from([query_billing_ledger, send_slack_alert], include_done=True)
    deferred_hook = DeferredToolViewHook(registry, pinned=())

    # Initially only search_tools and done are active
    initial_view = registry.tool_view(deferred_hook.select_tools())
    assert set(initial_view.names) == {"search_tools", "done"}
    assert set(deferred_hook.deferred_tool_names) == {"query_billing_ledger", "send_slack_alert"}

    observed_schemas_per_turn: list[set[str]] = []

    class InspectingBackend:
        def __init__(self) -> None:
            self.turn = 0

        def generate(self, prompt: str, **kwargs: Any) -> str:
            return ""

        def generate_with_tools(
            self, prompt: str, *, tools: list[dict[str, Any]], **kwargs: Any
        ) -> list[dict[str, Any]]:
            self.turn += 1
            observed_schemas_per_turn.append({t["name"] for t in tools})
            if self.turn == 1:
                return [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "search_tools",
                        "input": {"query": "billing customer_id"},
                    }
                ]
            if self.turn == 2:
                return [
                    {
                        "type": "tool_use",
                        "id": "call_2",
                        "name": "query_billing_ledger",
                        "input": {"customer_id": "cust_99"},
                    }
                ]
            return [
                {
                    "type": "tool_use",
                    "id": "call_3",
                    "name": "done",
                    "input": {"summary": "balance=125"},
                }
            ]

    res = run_result(
        InspectingBackend(),
        task="Check cust_99 billing balance",
        tools=registry,
        hooks=[deferred_hook],
        config=LoopConfig(max_steps=5, disabled_defaults=("persistence",)),
    )
    assert res.completed
    assert observed_schemas_per_turn[0] == {"search_tools", "done"}
    assert "query_billing_ledger" in observed_schemas_per_turn[1]
    assert "send_slack_alert" not in observed_schemas_per_turn[1]


# ── 5. Multimodal Image ContentBlock Support ────────────────────


def test_multimodal_image_blocks_in_openai_chat_anthropic_and_responses() -> None:
    b64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    data_uri = f"data:image/png;base64,{b64_png}"
    conv = Conversation(
        [
            Message(
                role=MessageRole.USER,
                content=[
                    ContentBlock(kind="text", data={"text": "Inspect this chart"}),
                    ContentBlock(kind="image", data={"url": data_uri}),
                ],
            )
        ]
    )
    snapshot = {"conversation": conv.serialize(), "prompt": "Initial task"}

    # 1. OpenAI Chat Completions
    chat_captured: list[Any] = []
    chat_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: (
                    chat_captured.append(kw)
                    or SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                    )
                )
            )
        )
    )
    oai_chat = OpenAIBackend(chat_client, model="gpt-6-luna")
    oai_chat.set_conversation_input(snapshot)
    oai_chat.generate("Initial task")
    multimodal_msg = chat_captured[0]["messages"][1]
    assert multimodal_msg["role"] == "user"
    assert multimodal_msg["content"] == [
        {"type": "text", "text": "Inspect this chart"},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]

    # 2. Anthropic Messages
    anth_captured: list[Any] = []
    anth_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kw: (
                anth_captured.append(kw) or SimpleNamespace(content=[SimpleNamespace(text="ok")])
            )
        )
    )
    anth = AnthropicBackend(anth_client, model="claude-sonnet-4-20250514")
    anth.set_conversation_input(snapshot)
    anth.generate("Initial task")
    anth_blocks = anth_captured[0]["messages"][0]["content"]
    assert any(
        b.get("type") == "image"
        and b.get("source") == {"type": "base64", "media_type": "image/png", "data": b64_png}
        for b in anth_blocks
    )

    # 3. OpenAI Responses API
    resp_captured: list[Any] = []
    resp_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kw: (
                resp_captured.append(kw)
                or SimpleNamespace(id="r1", status="completed", output_text="ok", output=[])
            )
        )
    )
    oai_resp = OpenAIResponsesBackend(resp_client, model="gpt-6-luna")
    oai_resp.set_conversation_input(snapshot)
    oai_resp.generate("Initial task")
    resp_msg = resp_captured[0]["input"][1]
    assert resp_msg["content"] == [
        {"type": "input_text", "text": "Inspect this chart"},
        {"type": "input_image", "image_url": data_uri},
    ]


# ── 6. Process-Group Subprocess Helper (run_subprocess_tool) ────


def test_run_subprocess_tool_timeout_cancellation_and_spill(tmp_path: Path) -> None:
    # 1. Timeout terminates process group cleanly
    res_timeout = run_subprocess_tool(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_s=0.1,
    )
    assert res_timeout["timed_out"] is True
    assert res_timeout["process_group_terminated"] is True
    assert res_timeout["returncode"] == -1

    # 2. CancelToken terminates in-flight process group and raises ToolCancelledError
    token = CancelToken()

    def _cancel_soon() -> None:
        time.sleep(0.08)
        token.cancel()

    threading.Thread(target=_cancel_soon, daemon=True).start()
    with pytest.raises(ToolCancelledError):
        run_subprocess_tool(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_s=10.0,
            cancel_token=token,
        )

    # 3. Output spilling when stdout exceeds max_output_chars
    res_spill = run_subprocess_tool(
        [sys.executable, "-c", "print('A' * 2000)"],
        max_output_chars=200,
        spill_dir=tmp_path / "spill",
    )
    assert res_spill["stdout_truncated"] is True
    assert Path(res_spill["persisted_output_path"]).is_file()
    assert res_spill["persisted_output_chars"] >= 2000


# ── 7. Session Branching from Checkpoint ────────────────────────


def test_branch_from_checkpoint_creates_new_run_and_steers(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    store = FileCheckpointStore(ckpt_dir)

    @tool(concurrent_safe=True)
    def compute(val: int) -> dict[str, int]:
        return {"doubled": val * 2}

    reg = tools_from([compute], include_done=True)
    state1 = DefaultState(max_steps=4)
    cfg1 = LoopConfig(
        max_steps=4,
        use_native_tools=False,
        checkpoint_dir=str(ckpt_dir),
        run_envelope=RunEnvelope(run_id="run-root"),
        disabled_defaults=("persistence",),
    )
    llm1 = MockLLMBackend(
        responses=[
            '{"tool": "compute", "args": {"val": 5}}',
            '{"tool": "done", "args": {"summary": "10"}}',
        ]
    )
    res1 = run_result(llm1, task="Compute", tools=reg, config=cfg1, state=state1)
    assert res1.completed

    # Branch from step 1 into a new run with an injected steering message
    session = ExecutionSession(run_ids=["run-root"])
    branched_cp = session.branch_checkpoint(
        store,
        step_number=1,
        new_run_id="run-branch-1",
        inject_message="Now compute val=21 instead before finishing.",
    )
    assert branched_cp.run_id == "run-branch-1"
    assert branched_cp.run_envelope is not None
    assert branched_cp.run_envelope["parent_run_id"] == "run-root"
    assert "run-branch-1" in session.run_ids

    state2 = DefaultState(max_steps=4)
    cfg2 = LoopConfig(
        max_steps=4,
        use_native_tools=False,
        checkpoint_dir=str(ckpt_dir),
        initial_checkpoint=branched_cp,
        run_envelope=RunEnvelope.from_dict(branched_cp.run_envelope),
        disabled_defaults=("persistence",),
    )
    llm2 = MockLLMBackend(
        responses=[
            '{"tool": "compute", "args": {"val": 21}}',
            '{"tool": "done", "args": {"summary": "42"}}',
        ]
    )
    res2 = run_result(llm2, task="Compute", tools=reg, config=cfg2, state=state2)
    assert res2.completed
    assert res2.output == {"status": "completed", "summary": "42"}


# ── 8. Eval expected_graders Manifest & replay_loop(stub_tools) ─


def test_eval_expected_graders_manifest_detects_deleted_required_grader(
    tmp_path: Path,
) -> None:
    @eval_mark("required")
    def eval_present(ctx: EvalContext) -> bool:
        return True

    ctx = EvalContext(steps=[], task={}, final_output={"summary": "ok"}, stop_reason="done")
    # Without expected_graders, eval_present alone passes
    report_ok = eval_assess([eval_present], ctx)
    assert report_ok.valid is True
    assert report_ok.passed is True

    # With expected_graders manifest requiring both eval_present and eval_deleted_oracle:
    manifest_file = tmp_path / "expected_graders.json"
    manifest_file.write_text(
        json.dumps({"required_graders": ["eval_present", "eval_deleted_oracle"]}),
        encoding="utf-8",
    )
    report_missing = eval_assess([eval_present], ctx, expected_graders=manifest_file)
    assert report_missing.valid is False
    assert report_missing.passed is False
    assert any("eval_deleted_oracle" in msg for msg in report_missing.integrity_failures)


def test_replay_loop_stub_tools_uses_recorded_output_for_stubbed_tools(
    tmp_path: Path,
) -> None:
    external_calls = 0
    transform_calls = 0

    @tool
    def external_api(query: str) -> dict[str, Any]:
        nonlocal external_calls
        external_calls += 1
        return {"raw_value": 100}

    @tool
    def local_transform(value: int) -> dict[str, Any]:
        nonlocal transform_calls
        transform_calls += 1
        return {"transformed": value + 5}

    trace_dir = tmp_path / "trace"
    sink = ProvenanceSink(trace_dir)
    reg = tools_from([external_api, local_transform], include_done=True)
    llm = sink.wrap_llm(
        MockLLMBackend(
            responses=[
                '{"tool": "external_api", "args": {"query": "q1"}}',
                '{"tool": "local_transform", "args": {"value": 100}}',
                '{"tool": "done", "args": {"summary": "105"}}',
            ]
        )
    )
    list(
        composable_loop(
            llm,
            task={"goal": "test replay stubbing"},
            tools=reg,
            hooks=[sink.trajectory_hook()],
            config=LoopConfig(
                max_steps=4,
                use_native_tools=False,
                disabled_defaults=("persistence",),
            ),
        )
    )
    sink.flush()
    assert external_calls == 1
    assert transform_calls == 1

    # Now modify local_transform and replay with stub_tools=["external_api"]
    @tool
    def local_transform_v2(value: int) -> dict[str, Any]:
        nonlocal transform_calls
        transform_calls += 1
        return {"transformed": value * 3}

    local_transform_v2.name = "local_transform"
    replay_reg = tools_from([external_api, local_transform_v2], include_done=True)
    replayed_steps = list(
        replay_loop(
            trace_dir,
            tools=replay_reg,
            stub_tools=["external_api"],
        )
    )
    # external_api was NOT re-executed live (still 1 call), while local_transform_v2 executed live!
    assert external_calls == 1
    assert transform_calls == 2
    assert replayed_steps[0].tool_result.data == {"raw_value": 100}
    assert replayed_steps[0].tool_result.metadata.get("replayed_from_trace") is True
    assert replayed_steps[1].tool_result.data == {"transformed": 300}
