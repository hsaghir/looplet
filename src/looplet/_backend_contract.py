"""Internal composition rules shared by transparent backend wrappers."""

from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Callable, Iterator

_SINGLE_MODEL_REQUEST: ContextVar[bool] = ContextVar("looplet_single_model_request", default=False)


@contextmanager
def single_model_request() -> Iterator[None]:
    token = _SINGLE_MODEL_REQUEST.set(True)
    try:
        yield
    finally:
        _SINGLE_MODEL_REQUEST.reset(token)


def single_model_request_active() -> bool:
    return _SINGLE_MODEL_REQUEST.get()


FORWARDED_CAPABILITIES = frozenset(
    {
        "last_usage",
        "last_model",
        "last_request",
        "last_response",
        "last_stop_reason",
        "checkpoint_state",
        "restore_checkpoint_state",
        "reset_conversation",
        "conversation_mode",
        "set_conversation_input",
        "apply_context_compaction",
        "set_model_turn_context",
        "record_tool_result",
        "set_run_envelope",
        "close",
        "_client",
        "_state_path",
        "_restore_local_state",
        "_looplet_manages_retries",
        "_supports_single_model_request",
    }
)
MUTABLE_CAPABILITIES = frozenset({"last_usage", "_client", "_state_path"})


def json_snapshot(value: Any) -> Any:
    """Copy JSON-compatible values and provider models without retaining live references."""

    def serialize(item: Any) -> Any:
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            return dump(mode="json")
        if isinstance(item, SimpleNamespace):
            return vars(item)
        return str(item)

    return json.loads(json.dumps(value, default=serialize))


class TerminalLLMError(RuntimeError):
    """A backend failure that cannot recover by repeating the same call."""

    def __init__(self, message: str, *, termination_reason: str) -> None:
        super().__init__(message)
        self.termination_reason = termination_reason


def backend_attribute(backend: Any, name: str) -> Any:
    """Forward only supported run capabilities, never generation itself."""
    if name not in FORWARDED_CAPABILITIES:
        raise AttributeError(name)
    return getattr(backend, name)


def invoke_backend(method: Callable[..., Any], prompt: str, options: dict[str, Any]) -> Any:
    """Pass supported options through wrappers and tolerate legacy signatures."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(prompt, **options)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return method(prompt, **options)
    return method(prompt, **{name: value for name, value in options.items() if name in parameters})


def attempt_limit(backend: Any, max_retries: int) -> int:
    """Do not retry a call whose resilience wrapper already owns attempts."""
    return 1 if getattr(backend, "_looplet_manages_retries", False) is True else max_retries + 1
