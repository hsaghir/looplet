"""Smoke tests for the three-layer context budget system.

Layer 1 - per-tool-result cap (``TOOL_RESULT_MAX_CHARS``,
``TOOL_RESULT_PERSIST_THRESHOLD_CHARS``).
Layer 2 - per-context-window aggregate cap
(``CONTEXT_WINDOW_STEPS``, ``CONTEXT_INLINE_PER_STEP_CHARS``,
``CONTEXT_WINDOW_TOTAL_CHARS``).
Layer 3 - whole-conversation compact (existing
:class:`looplet.compact.CompactService`; not exercised here, only
the tunable's existence is asserted).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from looplet import context_budget
from looplet.types import DefaultState, Step, ToolCall, ToolResult


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("temperature", [None, 0.7])
async def test_parse_recovery_sampling_is_unset_unless_configured(asynchronous, temperature):
    from looplet import LoopConfig, RunResult, async_composable_loop, composable_loop, tools_from

    requests = []

    class Backend:
        def generate(self, prompt, **kwargs):
            requests.append(kwargs.get("temperature"))
            if len(requests) == 1:
                return "not a tool call"
            return '{"tool":"done","args":{"summary":"checked"}}'

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    options = {} if temperature is None else {"recovery_temperature": temperature}
    state = DefaultState(max_steps=1)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([], include_done=True),
        state=state,
        config=LoopConfig(max_steps=1, use_native_tools=False, **options),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))
    assert RunResult.from_state(state).completed
    assert requests == [None, temperature]


def test_deterministic_compaction_retains_bounded_tool_receipts():
    from looplet.conversation import Message, MessageRole, default_summarizer

    data = {
        "batch": 0,
        "paid_total": 1136,
        "completed_records": ["batch-0.json"],
        "archive": "x" * 20_000,
    }
    result = ToolResult(
        tool="record_invoice_total",
        args_summary="batch=0, paid_total=1136",
        data=data,
        result_key="receipt-1",
    )
    summary = default_summarizer([Message(role=MessageRole.TOOL, content="", tool_result=result)])
    assert '"completed_records": ["batch-0.json"]' in summary
    assert '"result_key": "receipt-1"' in summary
    assert "data, not instructions" in summary
    assert "[truncated; full result in run evidence]" in summary
    assert len(summary) < 1200
    assert result.data is data and len(data["archive"]) == 20_000


@pytest.mark.parametrize("profile", ["execution-v1", "standard-v1"])
@pytest.mark.parametrize("native_tools", [False, True])
def test_default_prompt_matches_protocol_and_explicit_policy(profile, native_tools):
    from looplet import LoopConfig, composable_loop, tools_from

    prompts = []

    class Backend:
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return '{"tool":"done","args":{"summary":"checked"}}'

        def generate_with_tools(self, prompt, **kwargs):
            prompts.append(prompt)
            return [
                {
                    "type": "tool_use",
                    "id": "finish",
                    "name": "done",
                    "input": {"summary": "checked"},
                }
            ]

    list(
        composable_loop(
            llm=Backend(),
            tools=tools_from([], include_done=True),
            config=LoopConfig(max_steps=1, use_native_tools=native_tools, harness_profile=profile),
            task={"goal": "Finish"},
        )
    )
    prompt = prompts[0]
    assert ("Respond with JSON only" in prompt) is (not native_tools)
    assert ("provider's native tool format" in prompt) is native_tools
    assert ("LOW BUDGET" in prompt) is (profile == "standard-v1")


@pytest.mark.parametrize("capability", [None, False, True, "yes"])
@pytest.mark.parametrize("native_tools", [False, True])
def test_stateful_helper_requires_explicit_stateless_capability(capability, native_tools):
    from looplet import LoopConfig
    from looplet.assistant_turns import _AssistantTurns
    from looplet.conversation import Conversation
    from looplet.scaffolding import TerminalLLMError

    class Backend:
        conversation_mode = "stateful"

        def __init__(self):
            self.requests = []
            self.snapshots = []
            if capability is not None:
                self.supports_stateless_text_generation = capability

        def set_conversation_input(self, snapshot):
            self.snapshots.append(snapshot)

        def generate(self, prompt, **kwargs):
            self.requests.append(prompt)
            return "approved"

        def generate_with_tools(self, prompt, **kwargs):
            self.requests.append(prompt)
            return []

    turns = _AssistantTurns(DefaultState(max_steps=2), LoopConfig(max_model_calls=4))
    backend = Backend()
    admitted = turns.wrap(backend)
    admitted.prepare_conversation(Conversation(), "parent request")
    method = admitted.generate_with_tools if native_tools else admitted.generate
    if capability is True and not native_tools:
        assert method("independent review") == "approved"
        assert backend.requests == ["independent review"]
        assert backend.snapshots == [None]
        assert turns.data["model_calls"] == 1
    else:
        with pytest.raises(TerminalLLMError, match="unsupported_context"):
            method("independent review")
        assert backend.requests == []
        assert turns.data["model_calls"] == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cartridge", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_settlement_evidence_example(tmp_path, asynchronous, cartridge, native):
    import json

    from looplet import (
        FileToolResultStore,
        MockLLMBackend,
        RunResult,
        cartridge_to_preset,
        preset_to_cartridge,
    )
    from looplet.backends import _conversation_messages
    from looplet.examples.default_harness import (
        build_agent,
        grade_deliverable,
        scripted_responses,
        task_for,
    )
    from looplet.testing import _scripted_native_blocks

    requests = []

    class Backend(MockLLMBackend):
        conversation_mode = "messages"

        def set_conversation_input(self, snapshot):
            self._conversation_input = snapshot

        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            return super().generate(prompt, **kwargs)

        def generate_with_tools(self, prompt, *, tools, **kwargs):
            requests.append(json.dumps(_conversation_messages(self, prompt, native_blocks=False)))
            return _scripted_native_blocks(MockLLMBackend.generate(self, prompt, **kwargs))

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    preset = build_agent("evidence", tmp_path / "deliverable")
    preset.config.use_native_tools = native
    if cartridge:
        bundle = preset_to_cartridge(preset, tmp_path / "agent.cartridge", strict=True)
        preset.close()
        preset = cartridge_to_preset(bundle.path)
    backend_type = AsyncBackend if asynchronous else Backend
    backend = backend_type(scripted_responses("evidence"), cycle=False)
    with preset:
        if asynchronous:
            async for _ in preset.run_async(backend, task=task_for("evidence")):
                pass
            result = RunResult.from_state(preset.state)
        else:
            result = preset.run_result(backend, task=task_for("evidence"))
        assert result.completed, result.diagnostics()
        assert len(requests) == 6
        assert "/body" in requests[0]
        assert "731405" not in requests[1]
        assert "result_view_too_large" in requests[3]
        assert "731405" in requests[4]
        assert grade_deliverable("evidence", tmp_path / "deliverable")["passed"]
        source = result.steps[0].tool_result
        assert len(source.data["body"]) > 200_000
        assert preset.tools.get_result(source.result_key) == source.data
        reopened = FileToolResultStore(str(tmp_path / "tool-results"))
        assert reopened.get_result(source.result_key) == source.data
        assert [step.tool_call.args["limit"] for step in result.steps[2:4]] == [8000, 1000]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cartridge", [False, True])
async def test_small_context_handles_batched_large_sources(tmp_path, asynchronous, cartridge):
    from looplet import RunResult, cartridge_to_preset, preset_to_cartridge
    from looplet.examples.default_harness import (
        EventObserver,
        build_agent,
        grade_deliverable,
        read_invoice_batch,
    )
    from looplet.scaffolding import estimate_prompt_tokens

    class Backend:
        conversation_mode = "messages"

        def __init__(self):
            self.calls = 0

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

        def generate(self, prompt, **kwargs):
            raise AssertionError("Native fixture must not use a text fallback")

        def generate_with_tools(self, prompt, **kwargs):
            self.calls += 1
            assert estimate_prompt_tokens(prompt) <= 9000
            if self.calls == 1:
                return [
                    {
                        "type": "tool_use",
                        "id": f"source-{batch}",
                        "name": "read_invoice_batch",
                        "input": {"batch": batch},
                    }
                    for batch in range(3)
                ]
            if self.calls == 2:
                return [
                    {
                        "type": "tool_use",
                        "id": f"record-{batch}",
                        "name": "record_invoice_total",
                        "input": {
                            "batch": batch,
                            "paid_total": read_invoice_batch(batch)["paid_total"],
                        },
                    }
                    for batch in range(3)
                ]
            return [
                {"type": "tool_use", "id": "finish", "name": "done", "input": {"summary": "saved"}}
            ]

    class AsyncBackend(Backend):
        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    preset = build_agent("invoices", tmp_path / "deliverable")
    preset.config.context_inline_per_step_chars = 40_000
    preset.config.context_window_total_chars = 100_000
    preset.tools.tool_specs["read_invoice_batch"].render = {"max_chars": 40_000}
    if cartridge:
        bundle = preset_to_cartridge(preset, tmp_path / "agent.cartridge", strict=True)
        preset.close()
        preset = cartridge_to_preset(bundle.path)
    backend = AsyncBackend() if asynchronous else Backend()
    observer = EventObserver()
    preset.hooks.append(observer)
    with preset:
        if asynchronous:
            async for _ in preset.run_async(backend, task={"goal": "Reconcile the three batches"}):
                pass
            result = RunResult.from_state(preset.state)
        else:
            result = preset.run_result(backend, task={"goal": "Reconcile the three batches"})
        assert result.completed, result.diagnostics()
        assert backend.calls == 3
        assert grade_deliverable("invoices", tmp_path / "deliverable")["passed"]
        sources = [
            step.tool_result.data
            for step in result.steps
            if step.tool_call.tool == "read_invoice_batch"
        ]
        assert len(sources) == 3
        assert all(len(source["audit_notes"]) > 16_000 for source in sources)
        assert "post_compact" in [event.lower() for event in observer.events]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_small_context_compacts_without_losing_raw_evidence(tmp_path, asynchronous):
    from looplet import MockLLMBackend, RunResult
    from looplet.examples.default_harness import (
        EventObserver,
        build_agent,
        grade_deliverable,
        scripted_responses,
    )

    class MessagesBackend(MockLLMBackend):
        conversation_mode = "messages"

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

    class AsyncMessagesBackend(MessagesBackend):
        async def generate_with_tools(self, *args, **kwargs):
            return super().generate_with_tools(*args, **kwargs)

    backend_type = AsyncMessagesBackend if asynchronous else MessagesBackend
    backend = backend_type(scripted_responses("invoices"), cycle=False)
    preset = build_agent("invoices", tmp_path / "deliverable")
    preset.config.context_inline_per_step_chars = 40_000
    preset.config.context_window_total_chars = 100_000
    preset.tools.tool_specs["read_invoice_batch"].render = {"max_chars": 40_000}
    observer = EventObserver()
    preset.hooks.append(observer)
    with preset:
        if asynchronous:
            async for _ in preset.run_async(backend, task={"goal": "Reconcile the three batches"}):
                pass
            result = RunResult.from_state(preset.state)
        else:
            result = preset.run_result(backend, task={"goal": "Reconcile the three batches"})
        assert result.completed, result.diagnostics()
        assert grade_deliverable("invoices", tmp_path / "deliverable")["passed"]
        sources = [
            step.tool_result.data
            for step in result.steps
            if step.tool_call.tool == "read_invoice_batch"
        ]
        assert len(sources) == 3
        assert all(len(source["audit_notes"]) > 16_000 for source in sources)
        assert "post_compact" in [event.lower() for event in observer.events]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("persist", [False, True])
async def test_prompt_projection_preserves_original_run_evidence(tmp_path, asynchronous, persist):
    from looplet import LoopConfig, async_composable_loop, composable_loop, tools_from
    from looplet.checkpoint import Checkpoint
    from looplet.conversation import MessageRole
    from looplet.tools import BaseToolRegistry

    original = "evidence" * 10_000

    def source() -> str:
        return original

    class Backend:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            name = "source" if self.calls == 1 else "done"
            return '{"tool": "' + name + '", "args": {}}'

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    state = DefaultState(max_steps=3)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([source], include_done=True),
        state=state,
        config=LoopConfig(
            max_steps=3,
            use_native_tools=False,
            tool_result_persist_dir=str(tmp_path) if persist else None,
        ),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))

    assert state.steps[0].tool_result.data == original
    checkpoint = Checkpoint(
        step_number=state.step_count,
        session_log_data={},
        conversation_data=state.conversation.serialize(),
        config_snapshot={},
        tool_results_store=arguments["tools"].snapshot_results(),
        metadata={},
    )
    restored = BaseToolRegistry()
    restored.restore_results(Checkpoint.from_dict(checkpoint.to_dict()).tool_results_store)
    assert restored.get_result(state.steps[0].tool_result.result_key) == original
    projected = next(
        message.tool_result
        for message in state.conversation.messages
        if message.role == MessageRole.TOOL
    )
    assert projected is not state.steps[0].tool_result
    assert projected.data != original
    if persist:
        assert original in Path(projected.data["persisted_output_path"]).read_text()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_paging_compacts_before_reserved_context_admission(asynchronous):
    import hashlib
    import json

    from looplet import AgentPreset, LoopConfig, MockLLMBackend, RunResult, tools_from
    from looplet.builtin_tools.tool_results import SPECS
    from looplet.examples.default_harness import EventObserver, SmallWindowCompactService

    original = "retained evidence\n" * 24000
    key = "sha256:" + hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()

    def source():
        return original

    class Backend(MockLLMBackend):
        conversation_mode = "messages"

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

    class AsyncBackend(Backend):
        async def generate_with_tools(self, *args, **kwargs):
            return super().generate_with_tools(*args, **kwargs)

    replies = (
        [{"tool": "source", "args": {}}]
        + [
            {
                "tool": "read_tool_result",
                "args": {"result_key": key, "offset": index * 1800, "limit": 1800},
            }
            for index in range(6)
        ]
        + [{"tool": "done", "args": {"summary": "all pages checked"}}]
    )
    backend = (AsyncBackend if asynchronous else Backend)(
        [json.dumps(reply) for reply in replies],
        cycle=False,
    )
    tools = tools_from([source], include_done=True)
    tools.tool_specs["source"].render = {"max_chars": 900}
    for spec in SPECS:
        tools.register(spec)
    observer = EventObserver()
    preset = AgentPreset(
        config=LoopConfig(
            max_steps=10,
            max_tokens=2000,
            context_window=6000,
            use_native_tools=True,
            compact_service=SmallWindowCompactService(),
        ),
        tools=tools,
        state=DefaultState(max_steps=10),
        hooks=[observer],
    )
    with preset:
        if asynchronous:
            async for _ in preset.run_async(
                backend, task={"goal": "Read six evidence pages and finish"}
            ):
                pass
            result = RunResult.from_state(preset.state)
        else:
            result = preset.run_result(backend, task={"goal": "Read six evidence pages and finish"})
    assert result.completed, result.diagnostics()
    assert result.steps[0].tool_result.data == original
    assert len(result.steps) == len(replies)
    assert all(step.tool_result.error is None for step in result.steps)
    assert "post_compact" in [event.lower() for event in observer.events]


@pytest.mark.parametrize("native_blocks", [False, True])
def test_provider_tool_results_preserve_retrieval_refs_and_warnings(native_blocks):
    import json
    from types import SimpleNamespace

    from looplet.backends import _conversation_messages

    snapshot = {
        "prompt": "Task",
        "conversation": {
            "messages": [
                {
                    "role": "assistant",
                    "metadata": {"origin": "model"},
                    "content": [
                        {
                            "kind": "tool_use",
                            "data": {
                                "id": "call-1",
                                "name": "source",
                                "input": {},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_result": {
                        "tool": "source",
                        "call_id": "call-1",
                        "data": "bounded preview",
                        "result_key": "sha256:retained",
                        "warnings": ["incomplete preview"],
                    },
                },
            ]
        },
    }
    messages = _conversation_messages(
        SimpleNamespace(_conversation_input=snapshot), "Task", native_blocks=native_blocks
    )
    if native_blocks:
        block = messages[-1]["content"][0]
        assert block["tool_use_id"] == "call-1"
        payload = json.loads(block["content"])
    else:
        tool_message = next(message for message in messages if message["role"] == "tool")
        assert tool_message["tool_call_id"] == "call-1"
        payload = json.loads(tool_message["content"])
    assert payload["data"] == "bounded preview"
    assert payload["result_key"] == "sha256:retained"
    assert payload["warnings"] == ["incomplete preview"]


def retained_evidence_source() -> str:
    return "ordinary evidence\n" * 4000 + "needle=405\n"


def retained_evidence_view() -> dict:
    return {"content": "view-data" * 200, "next_offset": 1800}


@pytest.mark.parametrize("native_blocks", [False, True])
@pytest.mark.parametrize("refused", [False, True])
def test_conversation_preserves_complete_result_envelope(native_blocks, refused):
    import json
    from types import SimpleNamespace

    from looplet.backends import _conversation_messages
    from looplet.conversation import ContentBlock, Conversation, Message, MessageRole
    from looplet.result_projection import project_tool_result, result_payload

    original = ToolResult(
        tool="read_tool_result",
        args_summary="",
        call_id="observed",
        data={"content": "x" * (8000 if refused else 10), "offset": 13, "next_offset": 8013},
        error="permission denied" if not refused else None,
        warnings=["host-observed warning"],
        result_key="sha256:" + "a" * 64,
        metadata={"result_is_view": True},
    )
    result = project_tool_result(original, max_chars=900)
    conversation = Conversation(
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ContentBlock(
                        kind="tool_use",
                        data={
                            "id": "observed",
                            "name": "read_tool_result",
                            "input": {},
                        },
                    )
                ],
                metadata={"origin": "model"},
            ),
            Message(role=MessageRole.TOOL, content="", tool_result=result),
        ]
    )
    restored = Conversation.deserialize(conversation.serialize())
    assert restored.messages[-1].tool_result.warnings == result.warnings
    assert json.loads(
        restored.continuation_context().split("[TOOL RESULT read_tool_result] ")[1]
    ) == result_payload(result)
    messages = _conversation_messages(
        SimpleNamespace(
            _conversation_input={"prompt": "Task", "conversation": restored.serialize()}
        ),
        "Task",
        native_blocks=native_blocks,
    )
    raw = (
        next(
            message["content"][0]["content"]
            for message in messages
            if isinstance(message.get("content"), list)
            and message["content"][0].get("type") == "tool_result"
        )
        if native_blocks
        else next(message["content"] for message in messages if message["role"] == "tool")
    )
    assert json.loads(raw) == result_payload(result)


@pytest.mark.parametrize("execution_error", [None, "permission denied"])
def test_delivery_refusal_preserves_execution_status(execution_error):
    import json

    from looplet.result_projection import payload_text, project_tool_result

    original = ToolResult(
        tool="read_tool_result",
        args_summary="offset=11, limit=8000",
        data={"offset": 11, "content": "x" * 8000, "next_offset": 8011},
        error=execution_error,
        result_key="sha256:" + "a" * 64,
        metadata={"result_is_view": True},
    )
    projected = project_tool_result(original, max_chars=900)
    payload = json.loads(payload_text(projected))
    assert projected.error == execution_error
    assert projected.data is None
    assert len(payload_text(projected)) <= 900
    assert payload["delivery"]["status"] == "refused"
    assert payload["delivery"]["reason"] == "result_view_too_large"
    assert payload["delivery"]["max_chars"] == 900
    assert payload["delivery"]["offset"] == 11
    assert 0 < payload["delivery"]["suggested_limit"] < 8000
    assert "next_offset" not in payload["delivery"]
    assert original.data["next_offset"] == 8011
    assert original.error == execution_error


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_backend_observes_original_result_before_projection(asynchronous, wrapped):
    from looplet import LoopConfig, async_composable_loop, composable_loop, tools_from
    from looplet.backend_contract import BackendWrapper

    observed = []

    def source():
        return {"body": "original evidence" * 4000}

    class Backend:
        def generate(self, prompt, **kwargs):
            return (
                '{"tool":"source","args":{}}'
                if not observed
                else '{"tool":"done","args":{"summary":"checked"}}'
            )

        def record_tool_result(self, call_id, data, error=None):
            raise AssertionError("Legacy recorder belongs to the host hook")

        def record_tool_receipt(self, call, result):
            observed.append((call, result))

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    backend = AsyncBackend() if asynchronous else Backend()
    if wrapped:
        backend = BackendWrapper(backend)
    arguments = dict(
        llm=backend,
        tools=tools_from([source], include_done=True),
        config=LoopConfig(max_steps=2, use_native_tools=False),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))
    assert observed[0][0].tool == "source"
    assert observed[0][1].data == source()
    assert "model_projection" not in observed[0][1].metadata
    assert len(observed) == 2
    assert observed[-1][0].tool == "done"
    assert all(result.error is None for _, result in observed)


@pytest.mark.parametrize("name", ["read_tool_result", "search_tool_result"])
def test_query_tools_document_original_result_pointer(name):
    from looplet.builtin_tools.tool_results import SPECS

    spec = next(spec for spec in SPECS if spec.name == name)
    pointer = spec.to_api_schema()["input_schema"]["properties"]["json_pointer"]
    assert pointer["default"] == ""
    assert "original tool return value" in pointer["description"]
    assert "not its preview wrapper" in pointer["description"]


def test_delivery_retry_hint_accounts_for_json_escaping():
    from looplet.builtin_tools.tool_results import read_result
    from looplet.result_projection import project_tool_result

    source = '\\"\r\n' * 4000
    key = "sha256:" + "a" * 64
    page = read_result(source, result_key=key, json_pointer="", offset=13, limit=8000)
    refused = project_tool_result(
        ToolResult(
            tool="read_tool_result",
            args_summary="",
            data=page,
            metadata={"result_is_view": True},
        ),
        max_chars=900,
    )
    delivery = refused.metadata["model_projection"]["delivery"]
    retry = read_result(
        source,
        result_key=key,
        json_pointer="",
        offset=13,
        limit=delivery["suggested_limit"],
    )
    delivered = project_tool_result(
        ToolResult(
            tool="read_tool_result",
            args_summary="",
            data=retry,
            metadata={"result_is_view": True},
        ),
        max_chars=900,
    )
    assert not delivered.metadata["model_projection"].get("refused")
    assert delivered.data["content"] == source[13 : 13 + delivery["suggested_limit"]]
    assert delivery["result_key"] == key
    assert "next_offset" not in delivery


@pytest.mark.parametrize("as_view", [False, True])
@pytest.mark.parametrize("max_chars", [1000, 2000, 5000])
def test_batch_projection_can_be_verified_from_original_receipts(as_view, max_chars):
    from looplet.result_projection import bound_result_batch, project_tool_result, result_payload
    from looplet.scaffolding import ContextOverflowError

    originals = [
        ToolResult(
            tool="source",
            args_summary="",
            data={"offset": 0, "content": '\\"\r\n' * 3000, "next_offset": 12000},
            result_key="sha256:" + str(index) * 64,
            metadata={"result_is_view": as_view},
        )
        for index in range(3)
    ]
    try:
        selected = bound_result_batch(
            [project_tool_result(result) for result in originals],
            max_chars=max_chars,
        )
    except ContextOverflowError:
        assert max_chars == 1000
        return
    for original, result in zip(originals, selected):
        reproduced = project_tool_result(
            original,
            max_chars=result.metadata["model_projection"]["max_chars"],
        )
        assert result_payload(result) == result_payload(reproduced)


def test_source_projection_honors_legacy_limits_without_clipping_views(monkeypatch):
    from looplet.result_projection import payload_text, project_tool_result

    monkeypatch.setattr(context_budget, "TOOL_RESULT_MAX_CHARS", 500)
    monkeypatch.setattr(context_budget, "TOOL_RESULT_MAX_ROWS", 2)
    source = ToolResult(tool="source", args_summary="", data={"body": "x" * 4000})
    assert len(payload_text(project_tool_result(source))) <= 500
    rows = ToolResult(tool="rows", args_summary="", data=[1, 2, 3, 4])
    assert project_tool_result(rows).data["rows"] == [1, 2]
    view = ToolResult(
        tool="view", args_summary="", data="x" * 1000, metadata={"result_is_view": True}
    )
    assert project_tool_result(view).data == view.data
    source.metadata["render"] = {"max_chars": 2000}
    assert 1500 < len(payload_text(project_tool_result(source))) <= 2000


def test_result_view_contract_and_render_policy_roundtrip(tmp_path):
    from looplet import (
        AgentPreset,
        LoopConfig,
        cartridge_to_preset,
        preset_to_cartridge,
        tool,
        tools_from,
    )
    from looplet.result_projection import project_tool_result

    view = tool(retained_evidence_view, result_is_view=True)
    view.render = {"max_chars": 1000}
    preset = AgentPreset(
        config=LoopConfig(max_steps=2),
        hooks=[],
        tools=tools_from([view], include_done=True),
        state=DefaultState(max_steps=2),
    )
    bundle = preset_to_cartridge(preset, tmp_path / "agent.cartridge", strict=True)
    with cartridge_to_preset(bundle.path) as restored:
        result = restored.tools.dispatch(ToolCall(tool="retained_evidence_view"))
        assert result.error is None
        assert result.metadata["result_is_view"] is True
        assert result.metadata["render"] == {"max_chars": 1000}
        assert restored.tools.list_results()["total_results"] == 0
        projected = project_tool_result(result)
        assert projected.error is None
        assert (
            projected.metadata["model_projection"]["delivery"]["reason"] == "result_view_too_large"
        )
        assert projected.data is None
        assert result.data == retained_evidence_view()
    assert "tool_render_hints:" in (bundle.path / "runtime.yaml").read_text()
    assert "render:" not in (bundle.path / "tools/retained_evidence_view/tool.yaml").read_text()


def test_file_result_store_resource_roundtrips_with_its_quota(tmp_path):
    from looplet import (
        AgentPreset,
        FileToolResultStore,
        LoopConfig,
        cartridge_to_preset,
        preset_to_cartridge,
        tools_from,
    )

    registry = tools_from([retained_evidence_source], include_done=True, include_result_tools=True)
    preset = AgentPreset(
        config=LoopConfig(max_steps=3),
        hooks=[],
        tools=registry,
        state=DefaultState(max_steps=3),
        resources={
            "tool_result_store": FileToolResultStore(str(tmp_path / "evidence"), max_bytes=100_000)
        },
    )
    bundle = preset_to_cartridge(preset, tmp_path / "agent.cartridge", strict=True)
    with cartridge_to_preset(bundle.path) as restored:
        store = restored.tools.resources["tool_result_store"]
        assert isinstance(store, FileToolResultStore)
        assert store.max_bytes == 100_000
        result = restored.tools.dispatch(ToolCall(tool="retained_evidence_source"))
        assert result.error is None
        assert store.get_result(result.result_key) == retained_evidence_source()
        page = restored.tools.dispatch(
            ToolCall(
                tool="read_tool_result",
                args={"result_key": result.result_key, "offset": 72_000, "limit": 100},
            )
        )
        assert page.error is None
        assert page.data["content"] == "needle=405\n"
        assert store.list_results()["total_results"] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cartridge", [False, True])
async def test_full_result_retrieval_across_loop_forms(tmp_path, asynchronous, native, cartridge):
    import json
    import re

    from looplet import AgentPreset, LoopConfig, ProvenanceSink, RunResult, tools_from
    from looplet.cartridge import cartridge_to_preset, preset_to_cartridge
    from looplet.checkpoint import FileCheckpointStore

    preset = AgentPreset(
        config=LoopConfig(
            max_steps=4, use_native_tools=native, checkpoint_dir=str(tmp_path / "checkpoints")
        ),
        hooks=[],
        tools=tools_from([retained_evidence_source], include_done=True, include_result_tools=True),
        state=DefaultState(max_steps=4),
    )
    if cartridge:
        bundle = preset_to_cartridge(preset, tmp_path / "agent.cartridge", strict=True)
        preset.close()
        preset = cartridge_to_preset(bundle.path)

    class Backend:
        conversation_mode = "messages"

        def __init__(self):
            self.calls = 0
            self.snapshot = None
            self.result_key = None

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

        def response(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "retained_evidence_source", {}
            material = json.dumps(self.snapshot) if native else prompt
            if self.calls == 2:
                assert "needle=405" not in material
                self.result_key = re.search(r"sha256:[a-f0-9]{64}", material).group()
                return "search_tool_result", {"result_key": self.result_key, "query": "needle="}
            assert "needle=405" in material
            if self.calls == 3:
                return "read_tool_result", {
                    "result_key": self.result_key,
                    "offset": 72_000,
                    "limit": 100,
                }
            return "done", {"summary": "405"}

        def generate(self, prompt, **kwargs):
            assert not native
            name, args = self.response(prompt)
            return json.dumps({"tool": name, "args": args})

        def generate_with_tools(self, prompt, **kwargs):
            assert native
            assert {"read_tool_result", "search_tool_result", "list_tool_results"} <= {
                tool["name"] for tool in kwargs["tools"]
            }
            name, args = self.response(prompt)
            return [{"type": "tool_use", "id": f"call-{self.calls}", "name": name, "input": args}]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    backend = AsyncBackend() if asynchronous else Backend()
    sink = ProvenanceSink(dir=tmp_path / "trace")
    preset.hooks.append(sink.trajectory_hook())
    recorded_backend = sink.wrap_llm(backend)
    with preset:
        if asynchronous:
            async for _ in preset.run_async(recorded_backend, task={"goal": "Find needle's value"}):
                pass
            result = RunResult.from_state(preset.state)
        else:
            result = preset.run_result(recorded_backend, task={"goal": "Find needle's value"})
        sink.flush()
        assert result.completed, result.diagnostics()
        assert backend.calls == 4
        assert result.steps[0].tool_result.data == retained_evidence_source()
        assert result.steps[2].tool_result.data["content"] == "needle=405\n"
        saved = json.loads((tmp_path / "trace/trajectory.json").read_text())
        assert saved["steps"][0]["tool_result"]["data"] == retained_evidence_source()
        assert len(saved["llm_calls"]) == 4
        if native:
            presented = saved["llm_calls"][1]["capture"]["request"]["conversation_input"]
            assert backend.result_key in json.dumps(presented)
            assert "needle=405" not in json.dumps(presented)

    checkpoint = FileCheckpointStore(str(tmp_path / "checkpoints")).load_latest()
    restored = tools_from([], include_result_tools=True)
    restored.restore_results(checkpoint.tool_results_store)
    page = restored.dispatch(
        ToolCall(
            tool="read_tool_result",
            args={"result_key": backend.result_key, "offset": 72_000, "limit": 100},
        )
    )
    assert page.error is None, page.error
    assert page.data["content"] == "needle=405\n"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("inline_chars", [3000, 12_000])
async def test_result_view_delivery_is_complete_or_explicitly_refused(
    asynchronous, native, inline_chars
):
    import json
    import re

    from looplet import LoopConfig, RunResult, async_composable_loop, composable_loop, tools_from
    from looplet.backends import _conversation_messages

    original = "prefix" * 2000 + "p" * 7900 + "__PAGE_END__" + "tail" * 3000
    expected_page = original[12_000:20_000]

    def source() -> str:
        return original

    class Backend:
        conversation_mode = "messages"

        def __init__(self):
            self.calls = 0
            self._conversation_input = None

        def set_conversation_input(self, snapshot):
            self._conversation_input = snapshot

        def response(self, prompt):
            self.calls += 1
            material = (
                json.dumps(_conversation_messages(self, prompt, native_blocks=False))
                if native
                else prompt
            )
            if self.calls == 1:
                return "source", {}
            if self.calls == 2:
                result_key = re.search(r"sha256:[a-f0-9]{64}", material).group()
                return "read_tool_result", {
                    "result_key": result_key,
                    "offset": 12_000,
                    "limit": 8000,
                }
            assert expected_page in material or "result_view_too_large" in material
            if expected_page not in material:
                assert '"next_offset": 20000' not in material
            return "done", {"summary": "checked"}

        def generate(self, prompt, **kwargs):
            name, args = self.response(prompt)
            return json.dumps({"tool": name, "args": args})

        def generate_with_tools(self, prompt, **kwargs):
            name, args = self.response(prompt)
            return [{"type": "tool_use", "id": f"call-{self.calls}", "name": name, "input": args}]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    state = DefaultState(max_steps=3)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([source], include_done=True, include_result_tools=True),
        state=state,
        config=LoopConfig(
            max_steps=3,
            use_native_tools=native,
            context_inline_per_step_chars=inline_chars,
        ),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))
    result = RunResult.from_state(state)
    assert result.completed, result.diagnostics()
    assert arguments["llm"].calls == 3
    assert state.steps[1].tool_result.data["content"] == expected_page
    assert state.steps[1].tool_result.data["next_offset"] == 20_000


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native_blocks", [False, True])
async def test_native_result_batches_are_bounded_and_stable(asynchronous, native_blocks):
    import json

    from looplet import LoopConfig, RunResult, async_composable_loop, composable_loop, tools_from
    from looplet.backends import _conversation_messages

    def source(*, batch: int) -> dict:
        return {"batch": batch, "nested": {"archive": str(batch) * 20_000}}

    class Backend:
        conversation_mode = "messages"

        def __init__(self):
            self.requests = []
            self._conversation_input = None

        def set_conversation_input(self, snapshot):
            self._conversation_input = snapshot

        def generate(self, prompt, **kwargs):
            raise AssertionError("Native batch fixture must not use a text fallback")

        def generate_with_tools(self, prompt, **kwargs):
            self.requests.append(_conversation_messages(self, prompt, native_blocks=native_blocks))
            if len(self.requests) == 1:
                return [
                    {
                        "type": "tool_use",
                        "id": f"source-{batch}",
                        "name": "source",
                        "input": {"batch": batch},
                    }
                    for batch in range(4)
                ]
            if len(self.requests) == 2:
                return [
                    {"type": "tool_use", "id": "source-4", "name": "source", "input": {"batch": 4}}
                ]
            return [
                {
                    "type": "tool_use",
                    "id": "finish",
                    "name": "done",
                    "input": {"summary": "checked"},
                }
            ]

    class AsyncBackend(Backend):
        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    backend = AsyncBackend() if asynchronous else Backend()
    state = DefaultState(max_steps=6)
    arguments = dict(
        llm=backend,
        tools=tools_from([source], include_done=True),
        state=state,
        config=LoopConfig(max_steps=6, context_window_total_chars=5000),
    )
    if asynchronous:
        async for _ in async_composable_loop(**arguments):
            pass
    else:
        list(composable_loop(**arguments))
    result = RunResult.from_state(state)
    assert result.completed, result.diagnostics()
    assert len(backend.requests) == 3

    def tool_payloads(messages):
        if native_blocks:
            return [
                block["content"]
                for message in messages
                if isinstance(message.get("content"), list)
                for block in message["content"]
                if block.get("type") == "tool_result"
            ]
        return [message["content"] for message in messages if message["role"] == "tool"]

    initial = tool_payloads(backend.requests[1])
    continued = tool_payloads(backend.requests[2])
    assert len(initial) == 4
    assert sum(map(len, initial)) <= 5000
    assert continued[:4] == initial
    assert all(json.loads(payload)["result_key"] for payload in initial)
    assert [step.tool_result.data for step in result.steps[:5]] == [
        source(batch=batch) for batch in range(5)
    ]


def _make_step(number: int, tool: str, data: Any) -> Step:
    return Step(
        number=number,
        tool_call=ToolCall(tool=tool, args={}),
        tool_result=ToolResult(tool=tool, args_summary="", data=data),
    )


@pytest.mark.parametrize("limit", [128, 300, 1000])
@pytest.mark.parametrize("as_view", [False, True])
def test_result_projection_bounds_escaped_json_and_warning_envelopes(limit, as_view):
    from looplet.result_projection import payload_text, project_tool_result
    from looplet.scaffolding import ContextOverflowError

    data = {"nested": [{"text": '\\"\r\n\U0001f600' * 3000}], "next_offset": 12_000}
    result = ToolResult(
        tool="source",
        args_summary="",
        data=data,
        result_key="sha256:" + "a" * 64,
        warnings=["long warning" * 1000],
        metadata={"result_is_view": as_view},
    )
    try:
        projected = project_tool_result(result, max_chars=limit)
    except ContextOverflowError:
        assert limit < 300
    else:
        assert len(payload_text(projected)) <= limit
        if as_view:
            assert projected.data is None
            assert projected.error is None
            delivery = projected.metadata["model_projection"]["delivery"]
            assert delivery["reason"] == "result_view_too_large"
    assert result.data is data
    assert result.warnings == ["long warning" * 1000]


def test_text_context_honors_a_tool_specific_view_budget():
    from looplet.result_projection import payload_text, project_tool_result

    result = ToolResult(
        tool="custom_page",
        args_summary="",
        data={"content": "p" * 4000, "next_offset": 4000},
        metadata={"result_is_view": True, "render": {"max_chars": 5000}},
    )
    state = DefaultState()
    state.steps.append(Step(number=1, tool_call=ToolCall(tool="custom_page"), tool_result=result))
    assert payload_text(project_tool_result(result)) in state.context_summary()


# ── Layer 2: context_summary inlines actual data ────────────────


def test_context_summary_inlines_full_data() -> None:
    """The model must SEE the data, not a digest. Regression for the
    fabrication bug where ``commits: (5)`` left the LLM no choice but
    to invent commit shas on the next turn."""
    state = DefaultState()
    commits = [
        {"sha": "7321837", "message": "fix: third-pass audit"},
        {"sha": "7851476", "message": "fix: 10 second-pass audit findings"},
    ]
    state.steps.append(_make_step(1, "fetch_commits", {"commits": commits}))
    summary = state.context_summary()
    # The actual sha values must appear in the LLM-facing string.
    assert "7321837" in summary
    assert "7851476" in summary
    # Tool name and step number too.
    assert "fetch_commits" in summary
    assert "S1" in summary


def test_context_summary_per_step_cap_truncates_with_marker() -> None:
    """A single huge step gets truncated with a clear marker."""
    state = DefaultState()
    big = "x" * 10_000  # bigger than default CONTEXT_INLINE_PER_STEP_CHARS=3000
    state.steps.append(_make_step(1, "fetch", {"blob": big}))
    summary = state.context_summary()
    assert "[truncated; full result" in summary
    # Per-step cap is 3000 chars by default; resulting block ~3000+marker.
    assert len(summary) < 4000


def test_context_summary_window_steps_limits_recent() -> None:
    """Only the last ``CONTEXT_WINDOW_STEPS`` steps appear; older are
    out of scope (Layer 3 / compact handles them)."""
    state = DefaultState()
    for i in range(1, 11):  # 10 steps, names 'tool_a'..'tool_j'
        state.steps.append(_make_step(i, f"tool_{chr(96 + i)}", {"v": i}))
    summary = state.context_summary()
    # Default window = 5; steps 6..10 (tool_f..tool_j) appear, 1..5 (tool_a..tool_e) do not.
    assert "tool_j" in summary  # step 10
    assert "tool_f" in summary  # step 6
    assert "tool_e" not in summary  # step 5 is out of window
    assert "tool_a" not in summary  # step 1 is out of window


@pytest.mark.parametrize("window", [0, -1])
def test_context_summary_nonpositive_window_is_empty(window: int) -> None:
    state = DefaultState()
    state.steps.append(_make_step(1, "tool_a", {"v": 1}))
    token = context_budget._CONTEXT_WINDOW_STEPS_OVERRIDE.set(window)
    try:
        assert state.context_summary() == ""
    finally:
        context_budget._CONTEXT_WINDOW_STEPS_OVERRIDE.reset(token)


def test_context_summary_aggregate_cap_shrinks_largest() -> None:
    """When aggregate exceeds budget, the largest contributor shrinks."""
    state = DefaultState()
    # Five steps; one is far larger than the others.
    for i in range(1, 5):
        state.steps.append(_make_step(i, f"small_{i}", {"v": i}))
    state.steps.append(_make_step(5, "huge_tool", {"blob": "Y" * 20_000}))

    # Tighten the aggregate cap so the test exercises shrinking.
    with patch.object(context_budget, "CONTEXT_WINDOW_TOTAL_CHARS", 5000):
        summary = state.context_summary()
    assert len(summary) <= 6000  # cap + small overhead
    # Most-recent step is the huge one; it must remain visible (truncated).
    assert "huge_tool" in summary
    # Small steps are visible, full.
    assert "small_4" in summary


def test_context_summary_error_block_format() -> None:
    state = DefaultState()
    state.steps.append(
        Step(
            number=1,
            tool_call=ToolCall(tool="x", args={}),
            tool_result=ToolResult(tool="x", args_summary="", data=None, error="boom"),
        )
    )
    summary = state.context_summary()
    assert "✗" in summary
    assert "boom" in summary


def test_context_summary_empty_state() -> None:
    state = DefaultState()
    assert state.context_summary() == ""


# ── env override ────────────────────────────────────────────────


def test_env_override_tool_result_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LOOPLET_TOOL_RESULT_MAX_CHARS`` overrides the default."""
    monkeypatch.setenv("LOOPLET_TOOL_RESULT_MAX_CHARS", "12345")
    # Re-import to pick up new env value.
    importlib.reload(context_budget)
    try:
        assert context_budget.TOOL_RESULT_MAX_CHARS == 12345
    finally:
        monkeypatch.delenv("LOOPLET_TOOL_RESULT_MAX_CHARS", raising=False)
        importlib.reload(context_budget)


def test_env_override_unparseable_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A garbage env value must NOT crash the process; we fall back to
    the hardcoded default."""
    monkeypatch.setenv("LOOPLET_TOOL_RESULT_MAX_CHARS", "not-a-number")
    importlib.reload(context_budget)
    try:
        assert context_budget.TOOL_RESULT_MAX_CHARS == 6000  # the default
    finally:
        monkeypatch.delenv("LOOPLET_TOOL_RESULT_MAX_CHARS", raising=False)
        importlib.reload(context_budget)


def test_env_override_compact_trigger_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPLET_COMPACT_TRIGGER_FRACTION", "0.85")
    importlib.reload(context_budget)
    try:
        assert context_budget.COMPACT_TRIGGER_FRACTION == 0.85
    finally:
        monkeypatch.delenv("LOOPLET_COMPACT_TRIGGER_FRACTION", raising=False)
        importlib.reload(context_budget)


# ── Layer 1: persist-and-preview is wired through LoopConfig ────


def test_loopconfig_has_tool_result_persist_dir() -> None:
    """``LoopConfig`` exposes a knob for the persist directory."""
    from looplet.loop import LoopConfig

    cfg = LoopConfig()
    assert cfg.tool_result_persist_dir is None
    cfg2 = LoopConfig(tool_result_persist_dir="/tmp/looplet-tool-results")
    assert cfg2.tool_result_persist_dir == "/tmp/looplet-tool-results"


def test_tool_result_persistence_reports_utf8_bytes_and_characters(tmp_path: Path) -> None:
    from looplet.scaffolding import truncate_tool_result

    original = "\U0001f600" * 1000
    projected = truncate_tool_result(
        original, max_chars=100, persist_dir=str(tmp_path), persist_threshold=500
    )
    saved = Path(projected["persisted_output_path"])
    assert saved.read_bytes() == original.encode("utf-8")
    assert projected["persisted_output_size"] == len(original.encode("utf-8"))
    assert projected["persisted_output_chars"] == len(original)


def test_truncate_tool_result_persists_when_threshold_exceeded(tmp_path: Path) -> None:
    """When the result is huge AND persist_dir is set, full output goes
    to disk and the model sees a preview + the path."""
    from looplet.scaffolding import truncate_tool_result

    big = "Z" * 60_000
    result = truncate_tool_result(
        {"big_string": big},
        persist_dir=str(tmp_path),
        persist_threshold=50_000,
    )
    assert isinstance(result, dict)
    assert "persisted_output_path" in result
    assert "truncated_output" in result
    persisted = Path(result["persisted_output_path"])
    assert persisted.exists()
    # Full content is on disk.
    assert big in persisted.read_text()


def test_truncate_tool_result_no_persist_when_dir_unset() -> None:
    """Without persist_dir, only inline truncation applies."""
    from looplet.scaffolding import truncate_tool_result

    big = {"k": "Y" * 60_000}
    result = truncate_tool_result(big)
    # No persistence keys.
    assert "persisted_output_path" not in (result or {})
