"""Tests for native tool calling wiring in backends + scaffolding + loop.

Covers:
 - OpenAI / Anthropic backends implement ``generate_with_tools`` that returns
   normalised Anthropic-style content blocks.
 - ``llm_call_with_retry`` requires native tool support when enabled, and
     uses ``generate`` only for explicitly selected text mode.
 - The composable loop passes tool schemas through when
         ``LoopConfig.use_native_tools`` is enabled (the default), rejects
         text-only native responses, and parses structured ``tool_use`` blocks.
"""

from __future__ import annotations

from contextlib import aclosing, closing
from types import SimpleNamespace
from typing import Any

import pytest

from looplet import (
    DefaultState,
    LoopConfig,
    async_composable_loop,
    composable_loop,
    tool,
    tools_from,
)
from looplet.backends import (
    AnthropicBackend,
    OpenAIBackend,
    _anthropic_response_to_blocks,
    _openai_message_to_blocks,
    _to_openai_tools,
)
from looplet.hook_decision import HookDecision
from looplet.native_tools import NativeToolPolicy, NativeToolUnsupportedError
from looplet.scaffolding import llm_call_with_retry
from looplet.testing import AsyncMockLLMBackend, MockLLMBackend
from looplet.types import CancelToken, NativeToolBackend


@pytest.mark.parametrize(
    "response, reason, expected",
    [
        (
            "Compare the two sources before acting.",
            "stop",
            "Compare the two sources before acting.",
        ),
        ([{"type": "text", "text": "Check the evidence."}], "end_turn", "Check the evidence."),
        ([{"type": "text", "text": "Legacy response"}], None, "Legacy response"),
        ([{"type": "thinking", "thinking": "Opaque only"}], "stop", None),
        ([{"type": "text", "text": "Partial"}], "max_tokens", None),
        ([{"type": "text", "text": "No"}, {"type": "refusal"}], "stop", None),
        ([{"type": "tool_use", "name": ""}, {"type": "text", "text": "Malformed"}], "stop", None),
        ('{"tool":', "stop", None),
        ('{"answer": "Not a tool call"}', "stop", None),
        ("   ", "stop", None),
    ],
)
def test_assistant_text_classification_is_separate_from_completion(response, reason, expected):
    policy = NativeToolPolicy()
    assert policy.assistant_text(response, stop_reason=reason) == expected
    assert policy.parse_response(response, stop_reason=reason) == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("max_steps", [1, 2, 4])
@pytest.mark.parametrize("rejection", ["none", "hook", "schema", "hook_schema"])
async def test_shared_turn_policy_preserves_world_state_and_terminal_metadata(
    tmp_path, asynchronous, native, max_steps, rejection
):
    import json

    from looplet import Block, run_result, run_result_async
    from looplet.validation import FieldSpec, OutputSchema

    writes = []
    reports = []

    def write(label: str):
        writes.append(label)
        return {"written": label}

    def report(summary: str, answer: Any = None):
        reports.append(summary)
        return {"summary": summary, "answer": answer}

    class Gate:
        rejected = False

        def check_done(self, state, session_log, context, step_num):
            if rejection in {"hook", "hook_schema"} and not self.rejected:
                self.rejected = True
                return Block("First report rejected")
            return None

    first_args = {
        "summary": "accepted" if rejection == "none" else "rejected",
        "answer": "invalid" if rejection in {"schema", "hook_schema"} else 7,
    }
    responses = [
        json.dumps(
            {
                "tools": [
                    {"tool": "write", "args": {"label": "first"}},
                    {"tool": "report", "args": first_args},
                    {"tool": "write", "args": {"label": "ignored"}},
                    {"tool": "done", "args": {"summary": "ignored"}},
                ]
            }
        ),
        '{"tool":"report","args":{"summary":"accepted","answer":7}}',
    ]
    backend = (
        AsyncMockLLMBackend(responses, cycle=False)
        if asynchronous
        else MockLLMBackend(responses, cycle=False)
    )
    arguments = {
        "tools": tools_from([write, report], include_done=True),
        "hooks": [Gate()],
        "config": LoopConfig(
            max_steps=max_steps,
            use_native_tools=native,
            done_tools=["report"],
            done_tool_schemas={
                "report": OutputSchema(fields={"answer": FieldSpec("answer", "int")})
            },
            harness_profile="minimal",
            run_dir=str(tmp_path),
        ),
    }
    result = (
        await run_result_async(backend, **arguments)
        if asynchronous
        else run_result(backend, **arguments)
    )
    accepted = max_steps >= 2 and (rejection == "none" or max_steps == 4)
    assert writes == ["first"]
    assert reports == (["accepted"] if accepted else [])
    assert result.completed is accepted
    assert result.output == ({"summary": "accepted", "answer": 7} if accepted else None)
    assert backend.calls == (2 if rejection != "none" and max_steps == 4 else 1)
    assert [step.number for step in result.steps] == list(range(1, len(result.steps) + 1))
    if max_steps < 4:
        assert len(result.steps[0].metadata["turn"]["budget_skipped"]) == 4 - max_steps
    if max_steps >= 2:
        initial_terminal = result.steps[1]
        assert initial_terminal.tool_call.tool == "report"
        assert initial_terminal.metadata["terminal_accepted"] is (rejection == "none")
        if rejection in {"hook", "hook_schema"}:
            assert initial_terminal.tool_result.error == "First report rejected"
        elif rejection == "schema":
            assert initial_terminal.tool_result.error.startswith("Output schema validation failed:")
        if max_steps == 4:
            assert [
                call["tool"] for call in initial_terminal.metadata["turn"]["ignored_after_terminal"]
            ] == [
                "write",
                "done",
            ]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_model_call_admission_charges_started_calls_once(asynchronous):
    from looplet.assistant_turns import _AssistantTurns

    effects = []

    class Backend:
        def generate(self, prompt):
            effects.append(prompt)
            return "Recorded"

    class AsyncBackend:
        async def generate(self, prompt):
            effects.append(prompt)
            return "Recorded"

    turns = _AssistantTurns(DefaultState(), LoopConfig(max_model_calls=1))
    turns.begin_turn()
    backend = turns.wrap(AsyncBackend() if asynchronous else Backend())
    first = backend.generate("first")
    assert (await first if asynchronous else first) == "Recorded"
    with pytest.raises(RuntimeError, match="model_call_budget"):
        second = backend.generate("denied")
        if asynchronous:
            await second
    assert effects == ["first"]
    assert turns.data["model_calls"] == 1
    assert turns.data["turn_count"] == 1


async def test_hybrid_backend_keeps_pending_admission_until_response():
    from looplet.assistant_turns import _AssistantTurns

    class Backend:
        last_usage = {"input": 2, "output": 3}

        def generate(self, prompt):
            async def response():
                assert "pending_model_call" in turns.data
                return "Acknowledged"

            return response()

    turns = _AssistantTurns(DefaultState(), LoopConfig(max_model_calls=1))
    turns.begin_turn()
    pending = turns.wrap(Backend()).generate("hello")
    assert "pending_model_call" in turns.data
    assert await pending == "Acknowledged"
    assert "pending_model_call" not in turns.data
    assert turns.metadata["usage_total"] == {"input": 2, "output": 3}


@pytest.mark.parametrize("limit", [None, 0, -1, True])
def test_assistant_policy_requires_bounded_configuration(limit):
    from looplet.assistant_turns import validate_assistant_config

    with pytest.raises(ValueError, match="max_model_calls"):
        validate_assistant_config(
            LoopConfig(assistant_turn_policy=lambda context: None, max_model_calls=limit)
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema", "unknown"),
        ("policy_required", "true"),
        ("turn_count", -1),
        ("model_calls", True),
        ("assistant_turns", 1),
        ("consecutive_assistant_turns", 1),
        ("max_model_calls", True),
        ("max_assistant_turns", 0),
        ("decisions", [{}]),
        ("requests", [{}]),
    ],
)
def test_malformed_saved_assistant_state_rejects_without_mutating_input(field, value):
    import copy

    from looplet.assistant_turns import TURN_STATE_KEY, _AssistantTurns

    state = DefaultState()
    _AssistantTurns(state, LoopConfig(max_model_calls=5))
    state.metadata[TURN_STATE_KEY][field] = value
    before = copy.deepcopy(state.metadata)
    with pytest.raises(ValueError):
        _AssistantTurns(state, LoopConfig(max_model_calls=3))
    assert state.metadata == before


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["raises", "invalid", "feedback"])
async def test_invalid_host_policy_stops_without_effects_or_private_error_details(
    asynchronous, failure
):
    from looplet import AssistantTurnDecision, RunResult

    requests = []

    class Backend:
        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            return "Candidate"

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    def policy(context):
        if failure == "raises":
            raise RuntimeError("PRIVATE_ERROR_DETAIL")
        if failure == "feedback":
            return AssistantTurnDecision("continue", "X" * 8193)
        return {"action": "continue", "PRIVATE_POLICY_VALUE": True}

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([], include_done=True),
        state=state,
        config=LoopConfig(
            max_steps=2,
            max_model_calls=3,
            use_native_tools=False,
            assistant_turn_policy=policy,
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    result = RunResult.from_state(state)
    assert steps == []
    assert len(requests) == 1
    assert result.status.value == "failed"
    assert result.termination_reason == "assistant_policy_error"
    assert result.output is None
    assert "PRIVATE_" not in str(state.metadata)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["messages", "stateful"])
@pytest.mark.parametrize("keyword", [False, True])
async def test_helper_inference_never_reuses_parent_native_task(asynchronous, mode, keyword):
    from looplet.assistant_turns import _AssistantTurns
    from looplet.conversation import Conversation

    requests = []

    class Backend:
        conversation_mode = mode

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

        def generate(self, prompt, **kwargs):
            requests.append((prompt, self.snapshot))
            return "Returned"

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    turns = _AssistantTurns(DefaultState(), LoopConfig(max_model_calls=3))
    turns.begin_turn()
    backend = turns.wrap(AsyncBackend() if asynchronous else Backend())
    backend.prepare_conversation(
        Conversation(), "Parent task", request_prompt="Parent task with history"
    )
    helper_args = () if keyword else ("Independent helper task",)
    helper_kwargs = {"prompt": "Independent helper task"} if keyword else {}
    if mode == "stateful":
        with pytest.raises(RuntimeError, match="unsupported_context"):
            response = backend.generate(*helper_args, **helper_kwargs)
            if asynchronous:
                await response
        assert requests == []
    else:
        response = backend.generate(*helper_args, **helper_kwargs)
        assert (await response if asynchronous else response) == "Returned"
        assert requests == [("Independent helper task", None)]
    response = backend.generate("Parent task with history")
    assert (await response if asynchronous else response) == "Returned"
    assert requests[-1][1]["prompt"] == "Parent task"
    assert turns.data["model_calls"] == len(requests)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_denied_request_preserves_backend_snapshot_owner(asynchronous):
    from looplet.assistant_turns import _AssistantTurns

    preparation = []

    class Backend:
        def __init__(self, name):
            self.name = name

        def set_conversation_input(self, snapshot):
            preparation.append(self.name)

        def checkpoint_state(self):
            return {"owner": self.name}

        def generate(self, prompt):
            return "Recorded"

    class AsyncBackend(Backend):
        async def generate(self, prompt):
            return super().generate(prompt)

    backend_type = AsyncBackend if asynchronous else Backend
    turns = _AssistantTurns(DefaultState(), LoopConfig(max_model_calls=1))
    turns.begin_turn()
    first = turns.wrap(backend_type("primary")).generate(prompt="first")
    assert (await first if asynchronous else first) == "Recorded"
    with pytest.raises(RuntimeError, match="model_call_budget"):
        denied = turns.wrap(backend_type("unadmitted")).generate(prompt="denied")
        if asynchronous:
            await denied
    assert turns.wrap_checkpoint(None)(None)["llm"] == {"owner": "primary"}
    assert preparation == ["primary"]
    assert turns.data["requests"][0]["prompt_chars"] == len("first")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit", [1, 2])
async def test_post_run_inference_cannot_bypass_cap_or_erase_completion(asynchronous, limit):
    from looplet import AssistantTurnDecision, RunResult

    requests = []

    class Backend:
        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            return '{"tool":"done","args":{"summary":"accepted"}}'

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    def finish(state, session_log, context, llm):
        llm.generate("Optional post-run inference")
        return 999

    async def finish_async(state, session_log, context, llm):
        await llm.generate("Optional post-run inference")
        return 999

    hook = SimpleNamespace(on_loop_end=finish_async if asynchronous else finish)
    state = DefaultState(max_steps=1)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([], include_done=True),
        state=state,
        hooks=[hook],
        config=LoopConfig(
            max_steps=1,
            max_model_calls=limit,
            use_native_tools=False,
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert len(steps) == 1
    assert RunResult.from_state(state).completed
    assert len(requests) == limit
    assert state._looplet_run_stats["llm_calls"] == limit
    assert state.metadata.get("post_run_model_denials", []) == (
        ["model_call_budget"] if limit == 1 else []
    )


@pytest.mark.parametrize("limit", [1, 2])
def test_trace_builder_inference_uses_same_admission(limit):
    from looplet import AssistantTurnDecision, RunResult

    requests = []

    class Backend:
        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            return '{"tool":"done","args":{"summary":"accepted"}}'

    def build_trace(**kwargs):
        kwargs["llm"].generate("Optional trace inference")
        return {"checked": True}

    state = DefaultState(max_steps=1)
    list(
        composable_loop(
            llm=Backend(),
            tools=tools_from([], include_done=True),
            state=state,
            config=LoopConfig(
                max_steps=1,
                max_model_calls=limit,
                use_native_tools=False,
                build_trace=build_trace,
                assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
            ),
        )
    )
    assert RunResult.from_state(state).completed
    assert len(requests) == limit
    assert state._looplet_run_stats["llm_calls"] == limit
    assert state.metadata.get("post_run_model_denials", []) == (
        ["model_call_budget"] if limit == 1 else []
    )


async def test_overlapping_model_admission_fails_before_second_invocation():
    import asyncio

    from looplet.assistant_turns import _AssistantTurns

    started = asyncio.Event()
    finish = asyncio.Event()
    requests = []

    class Backend:
        async def generate(self, prompt):
            requests.append(prompt)
            started.set()
            await finish.wait()
            return "Recorded"

    turns = _AssistantTurns(DefaultState(), LoopConfig(max_model_calls=3))
    turns.begin_turn()
    backend = turns.wrap(Backend())
    first = asyncio.create_task(backend.generate("first"))
    await started.wait()
    with pytest.raises(RuntimeError, match="unsupported_context"):
        await backend.generate("second")
    assert "pending_model_call" in turns.data
    finish.set()
    assert await first == "Recorded"
    assert requests == ["first"]
    assert turns.data["model_calls"] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit", [1, 2])
async def test_resilient_attempts_are_individually_admitted_and_captured(asynchronous, limit):
    from looplet import RunResult
    from looplet.provenance import RecordingLLMBackend
    from looplet.resilient import ResilientBackend

    requests = []

    class Backend:
        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            if len(requests) == 1:
                raise OSError("Transient failure")
            return '{"tool":"done","args":{"summary":"accepted"}}'

    backend = RecordingLLMBackend(
        ResilientBackend(
            Backend(),
            retries=5,
            sleep=lambda seconds: pytest.fail("wrapper must not own capped retries"),
        )
    )
    state = DefaultState(max_steps=1)
    arguments = dict(
        llm=backend,
        tools=tools_from([], include_done=True),
        state=state,
        config=LoopConfig(max_steps=1, max_model_calls=limit, use_native_tools=False),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert len(requests) == limit
    assert len(backend.calls) == limit
    assert [call.capture["execution"]["model_call_id"] for call in backend.calls] == [
        f"run:turn-1:call-{number}" for number in range(1, limit + 1)
    ]
    assert state._looplet_run_stats["llm_calls"] == limit
    assert RunResult.from_state(state).completed is (limit == 2)
    assert len(steps) == (1 if limit == 2 else 0)
    if limit == 1:
        assert state.termination_reason == "model_call_budget"


def test_unaware_retry_owner_is_rejected_before_capped_inference():
    class Backend:
        _looplet_manages_retries = True

        def generate(self, prompt, **kwargs):
            pytest.fail("unsupported retry owner must not be invoked")

    state = DefaultState(max_steps=1)
    assert (
        list(
            composable_loop(
                llm=Backend(),
                tools=tools_from([], include_done=True),
                state=state,
                config=LoopConfig(max_steps=1, max_model_calls=2, use_native_tools=False),
            )
        )
        == []
    )
    assert state.termination_reason == "unsupported_context"
    assert state._looplet_run_stats["llm_calls"] == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_assistant_turns_continue_without_fake_steps(asynchronous, native):
    import json

    from looplet.assistant_turns import TURN_STATE_KEY, AssistantTurnDecision
    from looplet.conversation import Conversation

    prompts = []
    decisions = []
    effects = []
    planned = []
    selected = []

    def planner(*, model_turn_id, model_turn_number, **kwargs):
        planned.append((model_turn_id, model_turn_number, kwargs["step_number"]))

    def selector(*, model_turn_id, model_turn_number, **kwargs):
        selected.append((model_turn_id, model_turn_number, kwargs["step_num"]))

    def done(answer: str):
        effects.append(answer)
        return {"answer": answer}

    def policy(context):
        decisions.append(context)
        return AssistantTurnDecision("continue", feedback=f"Inspect source {len(decisions)}.")

    class Backend:
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            self.last_stop_reason = "stop"
            return [
                "Compare source A.",
                "Compare source B.",
                json.dumps({"tool": "done", "args": {"answer": "Checked"}}),
            ][len(prompts) - 1]

        def generate_with_tools(self, prompt, **kwargs):
            text = self.generate(prompt, **kwargs)
            if len(prompts) == 3:
                self.last_stop_reason = "tool_use"
                return [
                    {
                        "type": "tool_use",
                        "id": "final",
                        "name": "done",
                        "input": {"answer": "Checked"},
                    }
                ]
            return [{"type": "text", "text": text}]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return Backend.generate(self, prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            text = Backend.generate(self, prompt, **kwargs)
            if len(prompts) == 3:
                self.last_stop_reason = "tool_use"
                return [
                    {
                        "type": "tool_use",
                        "id": "final",
                        "name": "done",
                        "input": {"answer": "Checked"},
                    }
                ]
            return [{"type": "text", "text": text}]

    state = DefaultState(max_steps=1)
    conversation = Conversation()
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([done]),
        state=state,
        conversation=conversation,
        config=LoopConfig(
            max_steps=1,
            max_model_calls=3,
            assistant_turn_policy=policy,
            use_native_tools=native,
            context_planner=planner,
            tool_view_selector=selector,
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert effects == ["Checked"]
    assert len(steps) == 1
    assert steps[0].tool_call.tool == "done"
    assert state._stop_reason == "done"
    assert state.metadata[TURN_STATE_KEY]["model_calls"] == 3
    assert [context.model_turn_number for context in decisions] == [1, 2]
    assert [cursor[1:] for cursor in planned] == [(1, 1), (2, 1), (3, 1)]
    assert selected == planned
    assert prompts[2].count("Compare source A.") == 1
    assert prompts[2].count("Compare source B.") == 1
    assert prompts[2].count("Inspect source 1.") == 1
    assert all(not message.text.startswith("TASK:") for message in conversation.messages)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_text_adapter_receives_each_tool_observation_once(asynchronous):
    from looplet import AssistantTurnDecision, RunResult

    prompts = []

    class Backend:
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return [
                '{"tool":"lookup","args":{}}',
                "Consider the observed result.",
                '{"tool":"done","args":{"answer":"Accepted"}}',
            ][len(prompts) - 1]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    def lookup():
        return {"observed": "UNIQUE_TOOL_OBSERVATION"}

    def done(answer: str):
        return {"answer": answer}

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([lookup, done]),
        state=state,
        config=LoopConfig(
            max_steps=2,
            max_model_calls=3,
            use_native_tools=False,
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert [step.tool_call.tool for step in steps] == ["lookup", "done"]
    assert RunResult.from_state(state).completed
    assert prompts[1].count("UNIQUE_TOOL_OBSERVATION") == 1
    assert prompts[2].count("UNIQUE_TOOL_OBSERVATION") == 1
    assert prompts[2].count("Consider the observed result.") == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_assistant_only_turns_exhaust_calls_without_tool_steps(asynchronous):
    from looplet.assistant_turns import TURN_STATE_KEY, AssistantTurnDecision

    class Backend:
        last_stop_reason = "stop"

        def generate(self, prompt, **kwargs):
            return "Compare again."

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([], include_done=True),
        state=state,
        config=LoopConfig(
            max_steps=2,
            use_native_tools=False,
            max_model_calls=3,
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert steps == []
    assert state._stop_reason == "model_call_budget"
    assert state.metadata[TURN_STATE_KEY]["model_calls"] == 3
    assert state.metadata[TURN_STATE_KEY]["turn_count"] == 3
    assert state.run_status == "stopped"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "action, allowed, reject, reason",
    [
        ("stop", False, False, "assistant_stop"),
        ("complete", False, False, "assistant_policy_error"),
        ("complete", True, False, "done"),
        ("complete", True, True, "model_call_budget"),
    ],
)
async def test_assistant_policy_completion_and_stop_contract(
    asynchronous, native, action, allowed, reject, reason
):
    from looplet.assistant_turns import AssistantTurnDecision
    from looplet.types import RunResult

    effects = []

    def done(answer: str):
        effects.append(answer)
        return {"answer": answer}

    class Backend:
        last_stop_reason = "stop"

        def generate(self, prompt, **kwargs):
            return "Candidate"

        def generate_with_tools(self, prompt, **kwargs):
            return [{"type": "text", "text": "Candidate"}]

    backend = Backend()
    if asynchronous:
        original_generate = backend.generate
        original_native = backend.generate_with_tools

        async def generate(prompt, **kwargs):
            return original_generate(prompt, **kwargs)

        async def native_generate(prompt, **kwargs):
            return original_native(prompt, **kwargs)

        backend.generate = generate
        backend.generate_with_tools = native_generate

    class Gate:
        def check_done(self, *args, **kwargs):
            return HookDecision(block="Missing evidence") if reject else None

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=backend,
        tools=tools_from([done]),
        state=state,
        hooks=[Gate()],
        config=LoopConfig(
            max_steps=2,
            max_model_calls=1,
            use_native_tools=native,
            allow_text_completion=allowed,
            assistant_turn_policy=lambda context: AssistantTurnDecision(action),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    result = RunResult.from_state(state)
    assert effects == []
    assert result.termination_reason == reason
    assert result.completed is (reason == "done")
    if reason in {"assistant_stop", "assistant_policy_error"}:
        assert steps == []
        assert result.output is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_hook", [False, True])
async def test_assistant_progress_is_correlated_without_content_or_fake_steps(
    asynchronous, with_hook
):
    import json

    from looplet import AssistantTurnDecision
    from looplet.streaming import CallbackEmitter, StreamingHook

    events = []
    wire = []
    effects = []
    emitter = CallbackEmitter(events.append)
    responses = iter(
        [
            "PRIVATE_INTERMEDIATE_A",
            "PRIVATE_INTERMEDIATE_B",
            '{"tool":"lookup","args":{"key":"evidence"}}',
            '{"tool":"done","args":{"answer":"accepted"}}',
        ]
    )

    class Backend:
        def generate(self, prompt, **kwargs):
            return next(responses)

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    class Observer:
        def on_event(self, payload):
            wire.append(payload.to_jsonable())

    def lookup(key: str):
        effects.append(key)
        return {"observed": True}

    def done(answer: str):
        return {"answer": answer}

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([lookup, done]),
        state=state,
        stream=emitter,
        hooks=[Observer(), *([StreamingHook(emitter)] if with_hook else [])],
        config=LoopConfig(
            max_steps=2,
            max_model_calls=4,
            use_native_tools=False,
            assistant_turn_policy=lambda context: AssistantTurnDecision(
                "continue", "PRIVATE_HOST_FEEDBACK"
            ),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert [step.tool_call.tool for step in steps] == ["lookup", "done"]
    assert effects == ["evidence"]
    starts = [event for event in events if event.event_type == "LLMCallStartEvent"]
    ends = [event for event in events if event.event_type == "LLMCallEndEvent"]
    assert [event.model_turn_id for event in starts] == [
        f"run:turn-{number}" for number in range(1, 5)
    ]
    assert [event.model_turn_id for event in ends] == [event.model_turn_id for event in starts]
    assert len([event for event in events if event.event_type == "AssistantTurnDecisionEvent"]) == 2
    assert [event.step_num for event in events if event.event_type == "StepStartEvent"] == [1, 2]
    assert [event.total_llm_calls for event in events if event.event_type == "LoopEndEvent"] == [4]
    assert "PRIVATE_" not in json.dumps(wire)
    assert "PRIVATE_" not in str(events)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_consecutive_assistant_limit_is_independent_of_tool_budget(asynchronous):
    from looplet import AssistantTurnDecision

    state = DefaultState(max_steps=2)
    backend = (AsyncMockLLMBackend if asynchronous else MockLLMBackend)(["Continue."] * 5)
    arguments = dict(
        llm=backend,
        tools=tools_from([], include_done=True),
        state=state,
        config=LoopConfig(
            max_steps=2,
            max_model_calls=5,
            max_assistant_turns=2,
            use_native_tools=False,
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert steps == []
    assert state.budget_remaining == 2
    assert state._stop_reason == "assistant_turn_limit"
    assert state.metadata["looplet_assistant_turns"]["model_calls"] == 2


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_rejected_text_completion_retains_logical_host_feedback(asynchronous):
    from copy import deepcopy

    from looplet import AssistantTurnDecision, RunResult

    snapshots = []

    class Backend:
        conversation_mode = "stateful"

        def set_conversation_input(self, snapshot):
            snapshots.append(deepcopy(snapshot))

        def generate(self, prompt, **kwargs):
            return "Candidate answer"

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    class Gate:
        def check_done(self, *args, **kwargs):
            return HookDecision(block="Missing evidence") if len(snapshots) == 1 else None

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=tools_from([], include_done=True),
        state=state,
        hooks=[Gate()],
        config=LoopConfig(
            max_steps=2,
            max_model_calls=2,
            use_native_tools=False,
            allow_text_completion=True,
            assistant_turn_policy=lambda context: AssistantTurnDecision("complete"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert len(steps) == 2
    assert RunResult.from_state(state).completed
    feedback = [
        message
        for message in snapshots[1]["conversation"]["messages"]
        if message["metadata"].get("origin") == "host"
    ]
    assert len(feedback) == 1
    assert feedback[0]["role"] == "user"
    assert feedback[0]["content"] == "Missing evidence"
    assert "Missing evidence" not in snapshots[1]["prompt"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_rejected_text_completion_does_not_reset_consecutive_limit(asynchronous):
    from looplet import AssistantTurnDecision, RunResult

    class Gate:
        def check_done(self, *args, **kwargs):
            return HookDecision(block="Missing evidence")

    state = DefaultState(max_steps=4)
    backend = (AsyncMockLLMBackend if asynchronous else MockLLMBackend)(["Candidate"] * 5)
    arguments = dict(
        llm=backend,
        tools=tools_from([], include_done=True),
        state=state,
        hooks=[Gate()],
        config=LoopConfig(
            max_steps=4,
            max_model_calls=5,
            max_assistant_turns=1,
            use_native_tools=False,
            allow_text_completion=True,
            assistant_turn_policy=lambda context: AssistantTurnDecision("complete"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    result = RunResult.from_state(state)
    assert len(steps) == 1
    assert not result.completed
    assert result.output is None
    assert result.termination_reason == "assistant_turn_limit"
    assert state.metadata["looplet_assistant_turns"]["model_calls"] == 1
    assert state.metadata["looplet_assistant_turns"]["consecutive_assistant_turns"] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("when", ["pre_llm_call", "post_llm_response"])
async def test_assistant_mode_honors_model_hook_stop_before_effects(asynchronous, when):
    from looplet.assistant_turns import AssistantTurnDecision

    calls = []
    effects = []

    def mutate(value: int):
        effects.append(value)

    class Backend:
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            return "Unused"

        def generate_with_tools(self, prompt, **kwargs):
            calls.append(prompt)
            return [
                {"type": "tool_use", "id": "mutation", "name": "mutate", "input": {"value": 99}}
            ]

    backend = Backend()
    if asynchronous:
        original = backend.generate_with_tools

        async def generate_with_tools(prompt, **kwargs):
            return original(prompt, **kwargs)

        backend.generate_with_tools = generate_with_tools

    class StopHook:
        def on_event(self, payload):
            if payload.event.value == when:
                return HookDecision(stop="host_stop")

    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=backend,
        tools=tools_from([mutate], include_done=True),
        state=state,
        hooks=[StopHook()],
        config=LoopConfig(
            max_steps=2,
            max_model_calls=3,
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert steps == []
    assert effects == []
    assert len(calls) == int(when == "post_llm_response")
    assert state._stop_reason == "host_stop"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_recovery_cannot_exceed_admitted_calls(tmp_path, asynchronous):
    from looplet.assistant_turns import TURN_STATE_KEY, AssistantTurnDecision
    from looplet.provenance import ProvenanceSink

    effects = []

    def mutate(value: int):
        effects.append(value)

    class Backend:
        last_stop_reason = "max_tokens"
        last_usage = {"input": 5, "output": 2}

        def generate(self, prompt, **kwargs):
            raise AssertionError("No text fallback")

        def generate_with_tools(self, prompt, **kwargs):
            return [{"type": "tool_use", "id": "partial", "name": "mutate", "input": {"value": 99}}]

    backend = Backend()
    if asynchronous:
        original = backend.generate_with_tools

        async def generate_with_tools(prompt, **kwargs):
            return original(prompt, **kwargs)

        backend.generate_with_tools = generate_with_tools
    sink = ProvenanceSink(tmp_path)
    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=sink.wrap_llm(backend),
        tools=tools_from([mutate], include_done=True),
        state=state,
        hooks=[sink.trajectory_hook()],
        config=LoopConfig(
            max_steps=2,
            max_model_calls=2,
            max_tokens=10,
            max_turn_continuations=3,
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert steps == []
    assert effects == []
    assert state._stop_reason == "model_call_budget"
    assert state.metadata[TURN_STATE_KEY]["model_calls"] == 2
    assert state._looplet_run_stats["llm_calls"] == 2
    assert state.metadata["usage_total"] == {"input": 10, "output": 4}
    document = __import__("json").loads((sink.flush() / "trajectory.json").read_text())
    assert len(document["llm_calls"]) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("invalid_schema", [False, True])
async def test_text_completion_is_opt_in_gated_and_recorded(
    tmp_path, asynchronous, native, reject, invalid_schema
):
    from looplet.provenance import ProvenanceSink, async_replay_loop, replay_loop
    from looplet.validation import FieldSpec, OutputSchema

    effects = []
    gate_calls = []

    def done(answer: str) -> dict:
        effects.append(answer)
        return {"answer": answer}

    class Gate:
        def check_done(self, state, session_log, context, step_num, tool_call=None):
            gate_calls.append(dict(tool_call.metadata))
            return HookDecision(block="Synthetic completion gate") if reject else None

    class Backend:
        last_stop_reason = "stop"

        def generate(self, prompt, **kwargs):
            return "Source-backed final answer"

        def generate_with_tools(self, prompt, **kwargs):
            return [{"type": "text", "text": "Source-backed final answer"}]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    sink = ProvenanceSink(tmp_path)
    state = DefaultState(max_steps=1)
    arguments = dict(
        llm=sink.wrap_llm(AsyncBackend() if asynchronous else Backend()),
        tools=tools_from([done]),
        state=state,
        config=LoopConfig(
            max_steps=1,
            use_native_tools=native,
            allow_text_completion=True,
            output_schema=(
                OutputSchema(fields={"answer": FieldSpec("answer", "int")})
                if invalid_schema
                else None
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
        hooks=[Gate(), sink.trajectory_hook()],
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert effects == []
    assert gate_calls == [{"completion_source": "assistant_message"}]
    assert len(steps) == 1
    assert steps[0].tool_call.call_id == ""
    assert steps[0].tool_call.metadata["completion_source"] == "assistant_message"
    rejected = reject or invalid_schema
    assert state._stop_reason == ("budget_exhausted" if rejected else "done")
    assert bool(steps[0].tool_result.error) is rejected
    assert not list((tmp_path / "checkpoints").rglob("*pending*"))
    document = __import__("json").loads((sink.flush() / "trajectory.json").read_text())
    assert document["steps"][0]["llm_call_indices"] == [0]
    assert document["steps"][0]["tool_call"]["metadata"]["completion_source"] == "assistant_message"
    if not rejected:
        replay_state = DefaultState(max_steps=1)
        replay_arguments = dict(trace_dir=tmp_path, tools=tools_from([done]), state=replay_state)
        replayed = (
            [step async for step in async_replay_loop(**replay_arguments)]
            if asynchronous
            else list(replay_loop(**replay_arguments))
        )
        assert replay_state._stop_reason == "done"
        assert replayed[-1].tool_result.data == steps[-1].tool_result.data
        assert effects == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
async def test_native_recovery_preserves_attempts_and_dispatches_only_complete_turns(
    tmp_path, asynchronous, exhausted, overflow
):
    from looplet.cost import CostHook, CostTracker
    from looplet.provenance import ProvenanceSink, async_replay_loop, replay_loop

    effects = []

    def append(value: int) -> dict:
        effects.append(value)
        return {"value": value}

    def done(answer: str) -> dict:
        return {"answer": answer}

    class Backend:
        calls = 0

        def generate(self, prompt, **kwargs):
            raise AssertionError("No text fallback")

        def generate_with_tools(self, prompt, **kwargs):
            self.calls += 1
            if overflow and self.calls == 1:
                raise RuntimeError("context_length_exceeded")
            native_turn = self.calls - int(overflow)
            truncated = exhausted or native_turn == 1
            self.last_stop_reason = "max_tokens" if truncated else "tool_use"
            self.last_usage = {"input": 10, "output": 5}
            return [
                {"type": "thinking", "thinking": f"Attempt {self.calls}", "signature": "opaque"},
                {
                    "type": "tool_use",
                    "id": f"call-{self.calls}",
                    "name": "append" if native_turn < 3 else "done",
                    "input": {"value": 99 if truncated else 7}
                    if native_turn < 3
                    else {"answer": "Saved"},
                },
            ]

    backend = Backend()
    if asynchronous:
        original = backend.generate_with_tools

        async def generate_with_tools(prompt, **kwargs):
            return original(prompt, **kwargs)

        backend.generate_with_tools = generate_with_tools
    sink = ProvenanceSink(tmp_path)
    wrapped = sink.wrap_llm(backend)
    tracker = CostTracker()
    state = DefaultState(max_steps=2)
    arguments = dict(
        llm=wrapped,
        tools=tools_from([append, done]),
        state=state,
        config=LoopConfig(
            max_steps=2, max_tokens=40, max_turn_continuations=1, reactive_recovery=True
        ),
        hooks=[CostHook(tracker, backend=wrapped), sink.trajectory_hook()],
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    count = (2 if exhausted else 3) + int(overflow)
    assert state._stop_reason == ("output_truncated" if exhausted else "done")
    assert effects == ([] if exhausted else [7])
    assert tracker.calls == count
    assert tracker.input_tokens == 10 * (count - int(overflow))
    assert state._looplet_run_stats["llm_calls"] == count
    document = __import__("json").loads((sink.flush() / "trajectory.json").read_text())
    assert len(document["llm_calls"]) == count
    assert document["steps"][0]["llm_call_indices"] == list(range(2 + int(overflow)))
    assert document["llm_calls"][int(overflow)]["capture"]["response"][0]["thinking"] == (
        f"Attempt {1 + int(overflow)}"
    )
    replay_state = DefaultState(max_steps=len(steps))
    replay_arguments = dict(
        trace_dir=tmp_path, tools=tools_from([append, done]), state=replay_state
    )
    replayed = (
        [step async for step in async_replay_loop(**replay_arguments)]
        if asynchronous
        else list(replay_loop(**replay_arguments))
    )
    assert replay_state._stop_reason == state._stop_reason
    assert effects == ([] if exhausted else [7, 7])
    assert replayed[-1].tool_result.data == steps[-1].tool_result.data


@pytest.mark.parametrize(
    "response, reason",
    [
        ([], "stop"),
        ([{"type": "thinking", "thinking": "Only reasoning"}], "stop"),
        ([{"type": "text", "text": "Partial answer"}], "max_tokens"),
        ([{"type": "text", "text": "Refused"}, {"type": "refusal"}], "stop"),
        ([{"type": "text", "text": "Malformed"}, {"type": "tool_use", "name": ""}], "stop"),
        ('{"tool":', "stop"),
        ([{"type": "text", "text": '{"tool":'}], "stop"),
        ([{"type": "text", "text": '{"answer": "Not a tool call"}'}], "stop"),
    ],
)
def test_text_completion_does_not_promote_invalid_responses(response, reason):
    assert (
        NativeToolPolicy().parse_response(response, allow_text_completion=True, stop_reason=reason)
        == []
    )


# ── Helpers ──────────────────────────────────────────────────────


def _weather_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


class _FakeOpenAIClient:
    """Minimal fake OpenAI client capturing the last create() kwargs."""

    def __init__(self, message: Any) -> None:
        self._message = message
        self.last_kwargs: dict[str, Any] | None = None

    @property
    def chat(self) -> Any:
        client = self

        class _Completions:
            def create(_, **kwargs):
                client.last_kwargs = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(message=client._message)])

        return SimpleNamespace(completions=_Completions())


class _FakeAnthropicClient:
    def __init__(self, content: list[Any]) -> None:
        self._content = content
        self.last_kwargs: dict[str, Any] | None = None

        client = self

        class _Messages:
            def create(_, **kwargs):
                client.last_kwargs = kwargs
                return SimpleNamespace(content=client._content)

        self.messages = _Messages()


# ── _to_openai_tools ──────────────────────────────────────────────


class TestToOpenAITools:
    def test_wraps_schema_in_function_envelope(self):
        out = _to_openai_tools(_weather_schema())
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

    def test_missing_input_schema_yields_empty_object(self):
        out = _to_openai_tools([{"name": "noop", "description": "no-op"}])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# ── Message normalisers ───────────────────────────────────────────


class TestOpenAIMessageNormaliser:
    def test_tool_calls_and_content(self):
        msg = SimpleNamespace(
            content="Let me check.",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
                )
            ],
        )
        blocks = _openai_message_to_blocks(msg)
        assert blocks == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
        ]

    def test_malformed_arguments_kept_as_raw(self):
        msg = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_2",
                    function=SimpleNamespace(name="x", arguments="not-json"),
                )
            ],
        )
        blocks = _openai_message_to_blocks(msg)
        assert blocks == [
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "x",
                "input": {"_raw_arguments": "not-json"},
            }
        ]

    def test_no_tool_calls(self):
        msg = SimpleNamespace(content="Hello", tool_calls=None)
        assert _openai_message_to_blocks(msg) == [{"type": "text", "text": "Hello"}]

    def test_preserves_all_native_tool_calls_in_order(self):
        msg = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="first", arguments='{"value":1}'),
                ),
                SimpleNamespace(
                    id="call_2",
                    function=SimpleNamespace(name="second", arguments='{"value":2}'),
                ),
            ],
        )

        assert _openai_message_to_blocks(msg) == [
            {"type": "tool_use", "id": "call_1", "name": "first", "input": {"value": 1}},
            {"type": "tool_use", "id": "call_2", "name": "second", "input": {"value": 2}},
        ]


class TestAnthropicResponseNormaliser:
    def test_mixed_text_and_tool_use(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me check."),
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="get_weather", input={"city": "Paris"}
                ),
            ]
        )
        assert _anthropic_response_to_blocks(response) == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
        ]

    def test_preserves_all_native_tool_calls_in_order(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", id="tu_1", name="first", input={"value": 1}),
                SimpleNamespace(type="tool_use", id="tu_2", name="second", input={"value": 2}),
            ]
        )

        assert _anthropic_response_to_blocks(response) == [
            {"type": "tool_use", "id": "tu_1", "name": "first", "input": {"value": 1}},
            {"type": "tool_use", "id": "tu_2", "name": "second", "input": {"value": 2}},
        ]


# ── Backends implement NativeToolBackend ──────────────────────────


class TestBackendImplementsProtocol:
    def test_openai_backend_has_generate_with_tools(self):
        backend = OpenAIBackend(_FakeOpenAIClient(SimpleNamespace(content="", tool_calls=[])))
        assert isinstance(backend, NativeToolBackend)

    def test_anthropic_backend_has_generate_with_tools(self):
        backend = AnthropicBackend(_FakeAnthropicClient([]))
        assert isinstance(backend, NativeToolBackend)


class TestGenerateWithTools:
    def test_openai_passes_tools_and_returns_blocks(self):
        client = _FakeOpenAIClient(
            SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="c1",
                        function=SimpleNamespace(name="get_weather", arguments='{"city":"Tokyo"}'),
                    )
                ],
            )
        )
        backend = OpenAIBackend(client, model="gpt-x")
        blocks = backend.generate_with_tools("Weather?", tools=_weather_schema())
        assert blocks == [
            {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Tokyo"}}
        ]
        assert client.last_kwargs["tools"][0]["type"] == "function"
        assert client.last_kwargs["tool_choice"] == "auto"

    def test_anthropic_passes_tools_as_is(self):
        client = _FakeAnthropicClient(
            [
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="get_weather", input={"city": "Tokyo"}
                ),
            ]
        )
        backend = AnthropicBackend(client, model="claude-x")
        schemas = _weather_schema()
        blocks = backend.generate_with_tools("Weather?", tools=schemas, system_prompt="be brief")
        assert blocks == [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Tokyo"}}
        ]
        assert client.last_kwargs["tools"] == schemas
        assert client.last_kwargs["system"] == "be brief"


# ── llm_call_with_retry gating ───────────────────────────────────


class _NoToolBackend:
    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        return '{"tool": "done", "args": {}}'


class _NativeBackend:
    def __init__(self, blocks: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.blocks = (
            blocks
            if blocks is not None
            else [{"type": "tool_use", "id": "n1", "name": "done", "input": {}}]
        )

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        raise AssertionError("generate() should not be called in native mode")

    def generate_with_tools(
        self,
        prompt,
        *,
        tools,
        max_tokens=2000,
        system_prompt="",
        temperature=0.2,
        cancel_token=None,
    ):
        self.calls.append({"prompt": prompt, "tools": tools})
        return self.blocks


class _AsyncNativeBackend(_NativeBackend):
    async def generate_with_tools(self, prompt, **kwargs):
        return super().generate_with_tools(prompt, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_serial_native_batch_stops_at_consumed_step(is_async):
    effects = []

    @tool(description="Record one value.")
    def record(value: int) -> dict:
        effects.append(value)
        return {"value": value}

    blocks = [
        {"type": "tool_use", "id": "first", "name": "record", "input": {"value": 1}},
        {"type": "tool_use", "id": "second", "name": "record", "input": {"value": 2}},
        {"type": "tool_use", "id": "finish", "name": "done", "input": {}},
    ]
    backend = _AsyncNativeBackend(blocks) if is_async else _NativeBackend(blocks)
    state = DefaultState(max_steps=3)
    kwargs = {
        "llm": backend,
        "tools": tools_from([record], include_done=True),
        "state": state,
        "config": LoopConfig(max_steps=3, concurrent_dispatch=False),
        "task": {},
    }

    if is_async:
        async with aclosing(async_composable_loop(**kwargs)) as steps:
            first = await anext(steps)
            assert effects == [1]
    else:
        with closing(composable_loop(**kwargs)) as steps:
            first = next(steps)
            assert effects == [1]

    assert first.tool_call.args == {"value": 1}
    assert len(state.steps) == 1
    assert state.queries_used == 1
    assert state.termination_reason == "stream_closed"
    assert len(backend.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_serial_native_batch_intercepts_after_prior_step(is_async):
    effects = []
    observations = []

    @tool(description="Record one value.")
    def record(value: int) -> dict:
        effects.append(value)
        return {"value": value}

    class Observe:
        def pre_dispatch(self, state, session_log, tool_call, step_num):
            if tool_call.tool == "record":
                observations.append((tool_call.args["value"], len(state.steps), list(effects)))

    blocks = [
        {"type": "tool_use", "id": "first", "name": "record", "input": {"value": 1}},
        {"type": "tool_use", "id": "second", "name": "record", "input": {"value": 2}},
        {
            "type": "tool_use",
            "id": "finish",
            "name": "done",
            "input": {"summary": "Recorded both values."},
        },
    ]
    state = DefaultState(max_steps=3)
    kwargs = {
        "llm": _AsyncNativeBackend(blocks) if is_async else _NativeBackend(blocks),
        "tools": tools_from([record], include_done=True),
        "hooks": [Observe()],
        "state": state,
        "config": LoopConfig(max_steps=3, concurrent_dispatch=False),
        "task": {},
    }
    if is_async:
        steps = [step async for step in async_composable_loop(**kwargs)]
    else:
        steps = list(composable_loop(**kwargs))

    assert observations == [(1, 0, []), (2, 1, [1])]
    assert effects == [1, 2]
    assert [step.tool_call.tool for step in steps] == ["record", "record", "done"]
    assert steps[-1].tool_result.error is None
    assert state.termination_reason == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stop_kind", ["cancel", "post_dispatch", "should_stop"])
async def test_serial_native_batch_respects_stop_before_next_call(is_async, stop_kind):
    effects = []
    token = CancelToken()

    @tool(description="Record one value.")
    def record(value: int) -> dict:
        effects.append(value)
        return {"value": value}

    class StopAfterFirst:
        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            if stop_kind == "post_dispatch":
                return HookDecision(stop="first_finished")

        def should_stop(self, state, step_num, new_entities):
            if stop_kind == "should_stop":
                return HookDecision(stop="first_finished")

    blocks = [
        {"type": "tool_use", "id": "first", "name": "record", "input": {"value": 1}},
        {"type": "tool_use", "id": "second", "name": "record", "input": {"value": 2}},
        {"type": "tool_use", "id": "finish", "name": "done", "input": {}},
    ]
    state = DefaultState(max_steps=3)
    kwargs = {
        "llm": _AsyncNativeBackend(blocks) if is_async else _NativeBackend(blocks),
        "tools": tools_from([record], include_done=True),
        "hooks": [StopAfterFirst()],
        "state": state,
        "config": LoopConfig(max_steps=3, concurrent_dispatch=False, cancel_token=token),
        "task": {},
    }
    if is_async:
        async with aclosing(async_composable_loop(**kwargs)) as steps:
            first = await anext(steps)
            if stop_kind == "cancel":
                token.cancel()
            remaining = [step async for step in steps]
    else:
        with closing(composable_loop(**kwargs)) as steps:
            first = next(steps)
            if stop_kind == "cancel":
                token.cancel()
            remaining = list(steps)

    assert first.tool_call.args == {"value": 1}
    assert effects == [1]
    assert remaining == []
    assert len(state.steps) == 1
    assert state.termination_reason == ("cancelled" if stop_kind == "cancel" else "first_finished")


class _NativeFailureBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_with_tools(
        self, prompt, *, tools, max_tokens=2000, system_prompt="", temperature=0.2
    ):
        self.calls.append("native")
        from looplet.native_tools import NativeToolUnsupportedError

        raise NativeToolUnsupportedError("tools endpoint unsupported")

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.calls.append("regular")
        return '{"tool": "done", "args": {}}'


class _StickyNativeFailureBackend:
    def __init__(self) -> None:
        self.native_calls = 0
        self.regular_calls = 0

    def generate_with_tools(
        self, prompt, *, tools, max_tokens=2000, system_prompt="", temperature=0.2
    ):
        del prompt, tools, max_tokens, system_prompt, temperature
        self.native_calls += 1
        from looplet.native_tools import NativeToolUnsupportedError

        raise NativeToolUnsupportedError("tools endpoint unsupported")

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        del prompt, max_tokens, system_prompt, temperature
        self.regular_calls += 1
        if self.regular_calls == 1:
            return '{"tool": "echo", "args": {"value": "x"}}'
        return '{"tool": "done", "args": {"answer": "done"}}'


@tool(description="Echo a value.")
def _echo(*, value: str) -> dict[str, str]:
    return {"value": value}


class TestLLMCallWithRetryNative:
    def test_scripted_mock_returns_structured_native_calls(self):
        backend = MockLLMBackend(responses=['{"tool":"done","args":{},"call_id":"mock-1"}'])
        result = llm_call_with_retry(backend, "finish", tools=_weather_schema())

        assert result.ok
        assert result.text == [{"type": "tool_use", "name": "done", "input": {}, "id": "mock-1"}]
        assert NativeToolPolicy().parse_response(result.text)[0].tool == "done"
        assert backend.calls == 1

    @pytest.mark.asyncio
    async def test_async_scripted_mock_returns_structured_native_calls(self):
        from looplet.async_loop import async_llm_call

        backend = AsyncMockLLMBackend(responses=['{"tool":"done","args":{}}'])
        result = await async_llm_call(backend, "finish", tools=_weather_schema())

        assert result.ok
        assert len(result.text) == 1
        assert result.text[0]["type"] == "tool_use"
        assert result.text[0]["name"] == "done"
        assert result.text[0]["input"] == {}
        assert isinstance(result.text[0]["id"], str)
        assert NativeToolPolicy().parse_response(result.text)[0].tool == "done"
        assert backend.calls == 1

    def test_native_stats_are_inactive_before_any_result(self):
        from looplet.scaffolding import NativeToolStats

        assert NativeToolStats().has_activity is False

    def test_native_stats_record_success_and_unsupported(self):
        from looplet.scaffolding import NativeToolStats

        stats = NativeToolStats()
        success = llm_call_with_retry(_NativeBackend(), "hi", tools=_weather_schema())
        stats.record(success)
        unsupported = llm_call_with_retry(
            _NativeFailureBackend(), "hi", tools=_weather_schema(), max_retries=0
        )
        stats.record(unsupported)

        assert stats.to_dict() == {
            "requested": 2,
            "attempted": 2,
            "succeeded": 1,
            "fallbacks": 0,
            "last_fallback_reason": None,
        }
        assert isinstance(unsupported.error, NativeToolUnsupportedError)

    def test_native_policy_defaults_on_and_parses_after_demotion(self):
        policy = NativeToolPolicy()
        backend = _NativeBackend()
        assert policy.enabled is True
        assert policy.should_use(backend, _weather_schema())
        assert (
            policy.parse_response([{"type": "tool_use", "id": "n1", "name": "done", "input": {}}])[
                0
            ].tool
            == "done"
        )

        policy.demote()
        assert not policy.should_use(backend, _weather_schema())
        assert policy.parse_response('{"tool": "done", "args": {}}')[0].tool == "done"

    def test_native_policy_rejects_json_text_without_explicit_text_mode(self):
        policy = NativeToolPolicy()
        json_text = '{"tool": "done", "args": {}}'

        assert policy.parse_response([{"type": "text", "text": json_text}]) == []
        assert policy.parse_response(json_text) == []

        policy.enabled = False
        assert policy.parse_response(json_text)[0].tool == "done"

    def test_no_tools_uses_generate(self):
        backend = _NoToolBackend()
        result = llm_call_with_retry(backend, "hi")
        assert result.ok
        assert isinstance(result.text, str)

    def test_tools_and_native_backend_uses_generate_with_tools(self):
        backend = _NativeBackend()
        result = llm_call_with_retry(backend, "hi", tools=_weather_schema())
        assert result.ok
        assert isinstance(result.text, list)
        assert result.text[0]["type"] == "tool_use"
        assert backend.calls[0]["tools"] == _weather_schema()

    def test_tools_but_non_native_backend_requires_explicit_text_mode(self):
        backend = _NoToolBackend()  # no generate_with_tools
        result = llm_call_with_retry(backend, "hi", tools=_weather_schema())
        assert not result.ok
        assert isinstance(result.error, NativeToolUnsupportedError)
        assert "use_native_tools=False" in str(result.error)

        result = llm_call_with_retry(
            backend, "hi", tools=_weather_schema(), native_policy=NativeToolPolicy(enabled=False)
        )
        assert result.ok
        assert isinstance(result.text, str)

    def test_loop_missing_native_backend_reports_unsupported(self):
        steps = list(
            composable_loop(
                llm=_NoToolBackend(),
                task={"goal": "finish"},
                tools=tools_from([_echo], include_done=True),
                state=DefaultState(max_steps=2),
                config=LoopConfig(max_steps=2),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
        assert "use_native_tools=False" in steps[0].tool_result.error

    def test_native_failure_stays_error_without_regular_generation(self):
        backend = _NativeFailureBackend()
        result = llm_call_with_retry(
            backend,
            "hi",
            tools=_weather_schema(),
            max_retries=0,
        )
        assert not result.ok
        assert isinstance(result.error, NativeToolUnsupportedError)
        assert backend.calls == ["native"]
        assert result.native_fallback is False

    def test_provider_failure_does_not_fallback_to_text(self):
        class ProviderFailureBackend:
            def generate_with_tools(self, prompt, *, tools, **kwargs):
                del prompt, tools, kwargs
                raise RuntimeError("authentication failed")

            def generate(self, prompt, **kwargs):
                del prompt, kwargs
                raise AssertionError("ordinary provider failures must not demote")

        result = llm_call_with_retry(
            ProviderFailureBackend(),
            "hi",
            tools=_weather_schema(),
            max_retries=0,
        )

        assert not result.ok
        assert isinstance(result.error, RuntimeError)
        assert "authentication failed" in str(result.error)
        assert result.native_fallback is False

    def test_text_only_native_response_retries_without_json_recovery(self):
        class TextThenNativeBackend:
            def __init__(self):
                self.native_calls = 0

            def generate(self, prompt, **kwargs):
                raise AssertionError("native mode must not call JSON-text recovery")

            def generate_with_tools(self, prompt, *, tools, **kwargs):
                self.native_calls += 1
                if self.native_calls == 1:
                    return [{"type": "text", "text": '{"tool":"done","args":{}}'}]
                return [{"type": "tool_use", "id": "n2", "name": "done", "input": {}}]

        backend = TextThenNativeBackend()
        tools = tools_from([_echo], include_done=True)
        steps = list(
            composable_loop(
                llm=backend,
                tools=tools,
                task={"goal": "finish"},
                state=DefaultState(max_steps=2),
                config=LoopConfig(max_steps=2),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["__parse_error__", "done"]
        assert "expected structured tool_use" in steps[0].tool_result.error
        assert backend.native_calls == 2

    @pytest.mark.asyncio
    async def test_async_text_only_native_response_retries_without_json_recovery(self):
        class TextThenNativeBackend:
            def __init__(self):
                self.native_calls = 0

            async def generate(self, prompt, **kwargs):
                raise AssertionError("native mode must not call JSON-text recovery")

            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                self.native_calls += 1
                if self.native_calls == 1:
                    return [{"type": "text", "text": '{"tool":"done","args":{}}'}]
                return [{"type": "tool_use", "id": "n2", "name": "done", "input": {}}]

        backend = TextThenNativeBackend()
        tools = tools_from([_echo], include_done=True)
        steps = []
        async for step in async_composable_loop(
            llm=backend,
            tools=tools,
            task={"goal": "finish"},
            state=DefaultState(max_steps=2),
            config=LoopConfig(max_steps=2),
        ):
            steps.append(step)

        assert [step.tool_call.tool for step in steps] == ["__parse_error__", "done"]
        assert "expected structured tool_use" in steps[0].tool_result.error
        assert backend.native_calls == 2

    def test_native_failure_stops_the_loop_without_demoting(self):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})

        steps = list(
            composable_loop(
                llm=backend,
                task={"goal": "echo then finish"},
                tools=tools,
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
            )
        )

        assert backend.native_calls == 1
        assert backend.regular_calls == 0
        assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
        assert "tools endpoint unsupported" in steps[0].tool_result.error

    @pytest.mark.asyncio
    async def test_async_native_failure_stops_the_loop_without_demoting(self):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})

        steps = []
        async for step in async_composable_loop(
            llm=backend,
            task={"goal": "echo then finish"},
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
        ):
            steps.append(step)

        assert backend.native_calls == 1
        assert backend.regular_calls == 0
        assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
        assert "tools endpoint unsupported" in steps[0].tool_result.error

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.asyncio
    async def test_explicit_text_mode_completes_without_native_calls(self, async_mode):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})
        config = LoopConfig(max_steps=3, use_native_tools=False)
        if async_mode:
            steps = []
            async for step in async_composable_loop(
                llm=backend,
                task={"goal": "echo then finish"},
                tools=tools,
                state=DefaultState(max_steps=3),
                config=config,
            ):
                steps.append(step)
        else:
            steps = list(
                composable_loop(
                    llm=backend,
                    task={"goal": "echo then finish"},
                    tools=tools,
                    state=DefaultState(max_steps=3),
                    config=config,
                )
            )

        assert [step.tool_call.tool for step in steps] == ["echo", "done"]
        assert backend.native_calls == 0
        assert backend.regular_calls == 2
