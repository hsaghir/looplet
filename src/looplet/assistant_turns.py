"""Host decisions and admission state for bounded assistant-only turns."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import threading
import time
from dataclasses import asdict, dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Literal

from looplet._backend_contract import TerminalLLMError, backend_attribute, single_model_request

if TYPE_CHECKING:
    from looplet.checkpoint import Checkpoint
    from looplet.loop import LoopConfig

TURN_STATE_KEY = "looplet_assistant_turns"
TURN_STATE_SCHEMA = "looplet.assistant-turns.v1"


@dataclass(frozen=True)
class AssistantTurnContext:
    """Detached information supplied to the host's assistant-turn policy."""

    text: str
    model_turn_id: str
    model_turn_number: int
    model_calls: int
    max_model_calls: int
    consecutive_assistant_turns: int
    step_num: int


@dataclass(frozen=True)
class AssistantTurnDecision:
    """Continue, propose gated text completion, or stop without success."""

    action: Literal["continue", "complete", "stop"]
    feedback: str = ""

    def __post_init__(self) -> None:
        if self.action not in {"continue", "complete", "stop"}:
            raise ValueError("assistant turn action must be continue, complete, or stop")
        if not isinstance(self.feedback, str) or len(self.feedback) > 8192:
            raise ValueError("assistant turn feedback must be text of at most 8192 characters")


def validate_assistant_config(config: Any, *, require_binding: bool = True) -> None:
    """Reject unbounded or ambiguous policy wiring before a model call."""
    for name in ("max_model_calls", "max_assistant_turns"):
        value = getattr(config, name, None)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"config.{name} must be a positive integer or None")
    policy = getattr(config, "assistant_turn_policy", None)
    required = getattr(config, "requires_assistant_turn_policy", False)
    if type(required) is not bool:
        raise ValueError("config.requires_assistant_turn_policy must be boolean")
    if required and policy is None and require_binding:
        raise ValueError("an explicit host assistant_turn_policy binding is required")
    if config.max_assistant_turns is not None and config.max_model_calls is None:
        raise ValueError("max_assistant_turns requires an explicit max_model_calls limit")
    if policy is not None:
        if not callable(policy):
            raise ValueError("config.assistant_turn_policy must be callable")
        if config.max_model_calls is None:
            raise ValueError("assistant_turn_policy requires an explicit max_model_calls limit")


class _AssistantTurns:
    """Share turn decisions and backend admission across both loop variants."""

    def __init__(self, state: Any, config: Any) -> None:
        validate_assistant_config(config)
        self.config = config
        self.enabled = config.assistant_turn_policy is not None
        metadata = getattr(state, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("model-call admission requires mutable state.metadata")
        self.metadata = metadata
        self.state = state
        self.backend: Any = None
        self.admission_lock = threading.RLock()
        saved = metadata.get(TURN_STATE_KEY)
        if saved is None:
            saved = {
                "schema": TURN_STATE_SCHEMA,
                "turn_count": 0,
                "model_calls": 0,
                "assistant_turns": 0,
                "consecutive_assistant_turns": 0,
                "policy_required": self.enabled,
                "max_model_calls": config.max_model_calls,
                "max_assistant_turns": config.max_assistant_turns,
                "decisions": [],
            }
        if not isinstance(saved, dict) or saved.get("schema") != TURN_STATE_SCHEMA:
            raise ValueError("unsupported assistant-turn state schema")
        saved = copy.deepcopy(saved)
        if type(saved.get("policy_required")) is not bool:
            raise ValueError("invalid assistant-turn policy declaration")
        if saved.get("terminal_status"):
            raise ValueError("terminal assistant runs require an explicit new run identity")
        if saved.get("policy_required") and not self.enabled:
            raise ValueError("assistant-turn resume requires an explicit host policy binding")
        for name in ("turn_count", "model_calls", "assistant_turns", "consecutive_assistant_turns"):
            if type(saved.get(name)) is not int or saved[name] < 0:
                raise ValueError(f"invalid assistant-turn counter: {name}")
        for name in ("max_model_calls", "max_assistant_turns"):
            previous = saved.get(name)
            if previous is not None and (type(previous) is not int or previous <= 0):
                raise ValueError(f"invalid saved assistant-turn ceiling: {name}")
            current = getattr(config, name)
            if previous is not None and (current is None or current > previous):
                raise ValueError(f"assistant-turn resume cannot increase {name}")
            saved[name] = current
        decisions = saved.get("decisions")
        requests = saved.get("requests", [])
        if not isinstance(decisions, list) or len(decisions) > saved["assistant_turns"]:
            raise ValueError("invalid saved assistant-turn decisions")
        if not isinstance(requests, list) or len(requests) > saved["model_calls"]:
            raise ValueError("invalid saved assistant-turn requests")
        if (
            saved["consecutive_assistant_turns"] > saved["assistant_turns"]
            or saved["assistant_turns"] > saved["turn_count"]
        ):
            raise ValueError("inconsistent saved assistant-turn counters")
        for decision in decisions:
            if (
                not isinstance(decision, dict)
                or type(decision.get("action")) is not str
                or decision["action"] not in {"continue", "complete", "stop"}
            ):
                raise ValueError("invalid saved assistant-turn decision")
            digest = decision.get("feedback_hash")
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("invalid saved assistant-turn feedback digest")
            if type(decision.get("model_turn_id")) is not str or not decision["model_turn_id"]:
                raise ValueError("invalid saved assistant-turn decision identity")
        for request in requests:
            if not isinstance(request, dict) or not isinstance(request.get("usage"), dict):
                raise ValueError("invalid saved assistant-turn request")
            if any(
                type(request.get(name)) is not str or not request[name]
                for name in ("model_turn_id", "model_call_id")
            ):
                raise ValueError("invalid saved assistant-turn request identity")
            if any(
                type(request.get(name)) is not int or request[name] < 0
                for name in ("prompt_chars", "response_chars")
            ):
                raise ValueError("invalid saved assistant-turn request size")
        self.data = copy.deepcopy(saved)
        self.data.setdefault("requests", [])
        metadata[TURN_STATE_KEY] = self.data
        self.checkpoint: Callable[[], None] | None = None

    def wrap_checkpoint(self, domain_checkpoint: Any) -> Callable[[Any], dict[str, Any]]:
        def snapshot(loop_ctx: Any) -> dict[str, Any]:
            domain = copy.deepcopy(domain_checkpoint(loop_ctx) if domain_checkpoint else {})
            method = getattr(getattr(self, "backend", None), "checkpoint_state", None)
            if callable(method):
                backend_state = method()
                if inspect.isawaitable(backend_state):
                    if inspect.iscoroutine(backend_state):
                        backend_state.close()
                    raise TypeError("backend checkpoint_state must be synchronous")
                domain["llm"] = copy.deepcopy(backend_state)
            return domain

        return snapshot

    def save(self, store: Any, loop_ctx: Any, domain_checkpoint: Any) -> None:
        if store is None:
            return
        from looplet.checkpoint import Checkpoint

        if loop_ctx.phase.value == "terminal":
            self.data["terminal_status"] = loop_ctx.status.value
        domain = copy.deepcopy(domain_checkpoint(loop_ctx) if domain_checkpoint else {})
        checkpoint = Checkpoint(
            step_number=max((step.number for step in self.state.steps), default=0),
            session_log_data={
                "entries": loop_ctx.session_log.to_list(),
                "current_theory": loop_ctx.session_log.current_theory,
            },
            conversation_data=loop_ctx.conversation.serialize(),
            config_snapshot={
                name: getattr(self.config, name)
                for name in (
                    "max_steps",
                    "max_tokens",
                    "max_turn_continuations",
                    "max_model_calls",
                    "max_assistant_turns",
                    "allow_text_completion",
                    "temperature",
                    "done_tool",
                )
            }
            | {
                "budget_remaining": self.state.budget_remaining,
                "queries_used": getattr(self.state, "queries_used", 0),
            },
            tool_results_store=loop_ctx.tools.snapshot_results(),
            domain_state=domain,
            run_envelope=loop_ctx.run_envelope.to_dict() if loop_ctx.run_envelope else None,
            metadata=assistant_checkpoint_metadata(self.state),
            run_status=loop_ctx.status.value,
            run_phase=loop_ctx.phase.value,
            termination_reason=loop_ctx.termination_reason,
        )
        store.save(checkpoint, "model_turn")

    @property
    def turn_id(self) -> str:
        envelope = self.config.run_envelope
        run_id = envelope.run_id if envelope is not None else "run"
        return f"{run_id}:turn-{self.data['turn_count']}"

    @property
    def stop_reason(self) -> str | None:
        if self.data["model_calls"] >= self.config.max_model_calls:
            return "model_call_budget"
        limit = self.config.max_assistant_turns
        if limit is not None and self.data["consecutive_assistant_turns"] >= limit:
            return "assistant_turn_limit"
        return None

    def begin_turn(self) -> None:
        self.data["turn_count"] += 1

    def context(self, text: str, step_num: int) -> AssistantTurnContext:
        self.data["assistant_turns"] += 1
        self.data["consecutive_assistant_turns"] += 1
        return AssistantTurnContext(
            text=text,
            model_turn_id=self.turn_id,
            model_turn_number=self.data["turn_count"],
            model_calls=self.data["model_calls"],
            max_model_calls=self.config.max_model_calls,
            consecutive_assistant_turns=self.data["consecutive_assistant_turns"],
            step_num=step_num,
        )

    def record_decision(self, decision: Any) -> AssistantTurnDecision:
        if not isinstance(decision, AssistantTurnDecision):
            raise TypeError("assistant_turn_policy must return AssistantTurnDecision")
        if decision.action == "complete" and not self.config.allow_text_completion:
            raise ValueError("assistant completion requires allow_text_completion=True")
        self.data["decisions"].append(
            {
                "model_turn_id": self.turn_id,
                "action": decision.action,
                "feedback_hash": hashlib.sha256(decision.feedback.encode("utf-8")).hexdigest(),
            }
        )
        return decision

    def propose(self, text: str, step_num: int) -> Any:
        policy = self.config.assistant_turn_policy
        if policy is None:
            raise ValueError("assistant-turn policy binding is missing")
        return policy(self.context(text, step_num))

    def interruption_reason(self, decisions: Any = ()) -> str | None:
        envelope = self.config.run_envelope
        if envelope is not None and envelope.deadline_at is not None:
            if time.time() >= envelope.deadline_at:
                return "deadline_exceeded"
        token = self.config.cancel_token
        if token is not None and token.is_cancelled:
            return "cancelled"
        return next((decision.stop for decision in decisions if decision.stop is not None), None)

    def check_admission(self) -> None:
        if "pending_model_call" in self.data:
            raise _AdmissionDenied("unsupported_context")
        token = self.config.cancel_token
        if token is not None and token.is_cancelled:
            raise _AdmissionDenied("cancelled")
        envelope = self.config.run_envelope
        if envelope is not None and envelope.deadline_at is not None:
            if time.time() >= envelope.deadline_at:
                raise _AdmissionDenied("deadline_exceeded")
        if self.data["model_calls"] >= self.config.max_model_calls:
            raise _AdmissionDenied("model_call_budget")

    def admit(self, backend: Any = None) -> None:
        self.check_admission()
        if backend is not None:
            self.backend = backend
        self.data["model_calls"] += 1
        self.data["pending_model_call"] = {
            "model_turn_id": self.turn_id,
            "model_call_number": self.data["model_calls"],
        }
        if self.checkpoint is not None:
            self.checkpoint()

    def acknowledge(self) -> None:
        self.data.pop("pending_model_call", None)

    def wrap(self, backend: Any) -> Any:
        return _AdmissionBackend(backend, self)

    def surface_usage(self) -> None:
        totals: dict[str, Any] = {}
        for record in self.data["requests"]:
            for name, value in record.get("usage", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[name] = totals.get(name, 0) + value
        if totals:
            self.metadata["usage_total"] = totals
            self.metadata["last_usage"] = copy.deepcopy(self.data["requests"][-1]["usage"])


def assistant_checkpoint_metadata(state: Any) -> dict[str, Any]:
    metadata = getattr(state, "metadata", {})
    if not isinstance(metadata, dict) or TURN_STATE_KEY not in metadata:
        return {}
    return copy.deepcopy(
        {
            TURN_STATE_KEY: metadata[TURN_STATE_KEY],
            "assistant_steps": [asdict(step) for step in state.steps],
            "accepted_terminal_tool": getattr(state, "_accepted_terminal_tool", None),
            "assistant_usage": {
                name: metadata[name] for name in ("last_usage", "usage_total") if name in metadata
            },
            "assistant_stats": {
                **getattr(state, "_looplet_run_stats", {}),
                "llm_calls": metadata[TURN_STATE_KEY]["model_calls"],
            },
        }
    )


def restore_assistant_checkpoint(
    state: Any, config: LoopConfig, checkpoint: Checkpoint
) -> LoopConfig:
    saved = checkpoint.metadata.get(TURN_STATE_KEY)
    if saved is None:
        return config
    if not isinstance(saved, dict) or saved.get("schema") != TURN_STATE_SCHEMA:
        raise ValueError("unsupported assistant-turn checkpoint schema")
    from looplet.types import ErrorKind, Step, ToolCall, ToolError, ToolResult

    records = copy.deepcopy(checkpoint.metadata.get("assistant_steps", []))
    restored = []
    for record in records:
        result = record["tool_result"]
        detail = result.get("error_detail")
        if detail is not None:
            result["error_detail"] = ToolError(**{**detail, "kind": ErrorKind(detail["kind"])})
        restored.append(
            Step(
                number=record["number"],
                tool_call=ToolCall(**record["tool_call"]),
                tool_result=ToolResult(**result),
                metadata=record.get("metadata", {}),
            )
        )
    state.steps[:] = restored
    state.metadata[TURN_STATE_KEY] = copy.deepcopy(saved)
    state.metadata.update(copy.deepcopy(checkpoint.metadata.get("assistant_usage", {})))
    setattr(
        state, "_looplet_run_stats", copy.deepcopy(checkpoint.metadata.get("assistant_stats", {}))
    )
    setattr(state, "_accepted_terminal_tool", checkpoint.metadata.get("accepted_terminal_tool"))
    previous = checkpoint.config_snapshot
    remaining = previous.get("budget_remaining", config.max_steps)
    state.budget_remaining = max(
        0, remaining + config.max_steps - previous.get("max_steps", config.max_steps)
    )
    return replace(
        config,
        **{
            name: saved.get(name)
            for name in ("max_model_calls", "max_assistant_turns")
            if getattr(config, name) is None
        },
    )


class _AdmissionDenied(TerminalLLMError):
    request_started = False

    def __init__(self, reason: str) -> None:
        super().__init__(f"Model request not admitted: {reason}", termination_reason=reason)


class _AdmissionBackend:
    def __init__(self, backend: Any, turns: _AssistantTurns) -> None:
        self.backend = backend
        self.turns = turns
        self.request_records: list[dict[str, Any]] = []
        self.conversation_input: dict[str, Any] | None = None
        self.request_prompt: str | None = None
        self.context_error = False

    def prepare_conversation(
        self, conversation: Any, prompt: str, *, request_prompt: str | None = None
    ) -> None:
        self.request_prompt = request_prompt if request_prompt is not None else prompt
        mode = getattr(self.backend, "conversation_mode", "text")
        if mode not in {"text", "messages", "stateful"}:
            self.context_error = True
            return
        if mode != "text":
            if not callable(getattr(self.backend, "set_conversation_input", None)):
                self.context_error = True
                return
            self.conversation_input = {"conversation": conversation.serialize(), "prompt": prompt}
        local_chars = (
            len(json.dumps(self.conversation_input)) if self.conversation_input else len(prompt)
        )
        self.turns.data["context"] = {
            "local_chars": local_chars,
            "local_estimated_tokens": (local_chars + 3) // 4,
            "effective_context_tokens": None if mode == "stateful" else (local_chars + 3) // 4,
            "provider_held": mode == "stateful",
        }

    def _begin(self, args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        with self.turns.admission_lock:
            return self._begin_locked(args, kwargs)

    def _begin_locked(self, args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.turns.check_admission()
        if self.context_error:
            raise _AdmissionDenied("unsupported_context")
        if getattr(self.backend, "_looplet_manages_retries", False) and not getattr(
            self.backend, "_supports_single_model_request", False
        ):
            raise _AdmissionDenied("unsupported_context")
        if "pending_model_call" in self.turns.data:
            raise _AdmissionDenied("unsupported_context")
        prompt = args[0] if args else kwargs.get("prompt")
        is_helper = self.conversation_input is not None and prompt != self.request_prompt
        mode = getattr(self.backend, "conversation_mode", "text")
        if mode == "stateful" and self.turns.enabled and self.conversation_input is None:
            raise _AdmissionDenied("unsupported_context")
        if is_helper and mode == "stateful":
            raise _AdmissionDenied("unsupported_context")
        if not is_helper and self.turns.data.get("context_reset_required"):
            if getattr(self.backend, "conversation_mode", "text") == "stateful":
                reset = getattr(self.backend, "apply_context_compaction", None)
                if not callable(reset) or self.conversation_input is None:
                    raise _AdmissionDenied("unsupported_context")
                try:
                    acknowledged = reset(copy.deepcopy(self.conversation_input))
                except TerminalLLMError as error:
                    raise _AdmissionDenied(error.termination_reason) from error
                if inspect.isawaitable(acknowledged):
                    if inspect.iscoroutine(acknowledged):
                        acknowledged.close()
                    raise _AdmissionDenied("unsupported_context")
                if acknowledged is not True:
                    raise _AdmissionDenied("unsupported_context")
                self.turns.data.setdefault("context_resets", []).append(
                    {
                        "model_turn_id": self.turns.turn_id,
                        "acknowledged": True,
                    }
                )
            self.turns.data.pop("context_reset_required", None)
        setter = getattr(self.backend, "set_conversation_input", None)
        if callable(setter):
            try:
                setter(None if is_helper else copy.deepcopy(self.conversation_input))
            except TerminalLLMError as error:
                raise _AdmissionDenied(error.termination_reason) from error
        context = self.turns.data.get("context", {})
        if (
            not is_helper
            and self.conversation_input is not None
            and context.get("local_estimated_tokens", 0) > self.turns.config.context_window - 3000
        ):
            raise _AdmissionDenied("context_limit")
        correlation = getattr(self.backend, "set_model_turn_context", None)
        if callable(correlation):
            correlation(
                {
                    "required_semantics": TURN_STATE_SCHEMA,
                    "model_turn_id": self.turns.turn_id,
                    "model_call_id": f"{self.turns.turn_id}:call-{self.turns.data['model_calls'] + 1}",
                    "assistant_policy_required": self.turns.enabled,
                }
            )
        self.turns.admit(backend=self.backend)
        record = {
            "model_turn_id": self.turns.turn_id,
            "model_call_id": f"{self.turns.turn_id}:call-{self.turns.data['model_calls']}",
            "prompt_chars": len(prompt) if isinstance(prompt, str) else 0,
            "response_chars": 0,
            "usage": {},
        }
        self.request_records.append(record)
        self.turns.data["requests"].append(record)
        return record

    def _finish(self, record: dict[str, Any], response: Any) -> None:
        record["response_chars"] = (
            len(response) if isinstance(response, str) else len(json.dumps(response, default=str))
        )
        usage = getattr(self.backend, "last_usage", None)
        if isinstance(usage, dict):
            record["usage"] = copy.deepcopy(usage)
        self.turns.surface_usage()

    def __getattr__(self, name: str) -> Any:
        if name == "_looplet_manages_retries" and getattr(
            self.backend, "_supports_single_model_request", False
        ):
            return False
        if name not in {"generate", "generate_with_tools"}:
            return backend_attribute(self.backend, name)
        method = getattr(self.backend, name)
        if inspect.iscoroutinefunction(method):

            @wraps(method)
            async def invoke_async(*args: Any, **kwargs: Any) -> Any:
                record = self._begin(args, kwargs)
                try:
                    with single_model_request():
                        response = await method(*args, **kwargs)
                    self._finish(record, response)
                    return response
                finally:
                    self.turns.acknowledge()

            return invoke_async

        @wraps(method)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            record = self._begin(args, kwargs)
            response: Any = None
            try:
                with single_model_request():
                    response = method(*args, **kwargs)
                if inspect.isawaitable(response):

                    async def finish_response() -> Any:
                        try:
                            with single_model_request():
                                resolved = await response
                            self._finish(record, resolved)
                            return resolved
                        finally:
                            self.turns.acknowledge()

                    return finish_response()
                self._finish(record, response)
                return response
            finally:
                if not inspect.isawaitable(response):
                    self.turns.acknowledge()

        return invoke
