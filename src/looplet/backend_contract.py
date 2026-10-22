"""Optional backend contracts and transparent composition helpers.

Only generation is required. Capability protocols are independent and do not
require inheritance. Wrappers should not advertise capabilities they cannot
preserve. The single-request scope lets an adapter disable hidden SDK retries
when the host owns request admission; it is not itself a retry or budget policy.
Use callable attribute discovery for dynamic wrappers: Python 3.12+ protocol
instance checks use static inspection and do not follow ``__getattr__``.
"""

from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

from looplet.types import (
    CheckpointableLLMBackend,
    LLMBackend,
    NativeToolBackend,
    ToolCall,
    ToolResult,
)

__all__ = [
    "BackendRequestPolicy",
    "BackendWrapper",
    "CheckpointableLLMBackend",
    "CompactionAwareBackend",
    "ConversationInputBackend",
    "IsolatedBackend",
    "LLMBackend",
    "NativeToolBackend",
    "RequestAttemptBackend",
    "TerminalLLMError",
    "ToolResultBackend",
    "attempt_limit",
    "backend_attribute",
    "backend_request_policy",
    "invoke_backend",
    "json_snapshot",
    "single_model_request",
    "single_model_request_active",
]


@dataclass(frozen=True)
class BackendRequestPolicy:
    """Declare retry ownership and support for host-admitted individual attempts."""

    manages_retries: bool = False
    supports_single_model_request: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.manages_retries) is not bool
            or type(self.supports_single_model_request) is not bool
        ):
            raise ValueError("backend request policy fields must be booleans")


def backend_request_policy(backend: Any) -> BackendRequestPolicy:
    """Read public policy while retaining conservative legacy retry declarations."""
    policy = getattr(backend, "request_policy", None)
    legacy_owner = getattr(backend, "_looplet_manages_retries", False) is True
    legacy_support = getattr(backend, "_supports_single_model_request", None)
    if policy is not None:
        if not isinstance(policy, BackendRequestPolicy):
            if inspect.getattr_static(backend, "request_policy", None) is not None:
                raise ValueError("request_policy must be a BackendRequestPolicy")
        else:
            return BackendRequestPolicy(
                manages_retries=policy.manages_retries or legacy_owner,
                supports_single_model_request=(
                    policy.supports_single_model_request
                    and (legacy_support is None or legacy_support is True)
                ),
            )
    return BackendRequestPolicy(
        manages_retries=legacy_owner,
        supports_single_model_request=legacy_support is True,
    )


@runtime_checkable
class ConversationInputBackend(Protocol):
    """Accept a detached selected conversation instead of reconstructing history."""

    conversation_mode: str

    def set_conversation_input(self, snapshot: dict[str, Any] | None) -> None: ...


@runtime_checkable
class CompactionAwareBackend(Protocol):
    """Acknowledge an explicitly selected compacted conversation."""

    def apply_context_compaction(self, snapshot: dict[str, Any]) -> bool: ...


@runtime_checkable
class ToolResultBackend(Protocol):
    """Observe actual dispatch results, not just the model's proposed calls."""

    def record_tool_result(self, call: ToolCall, result: ToolResult) -> None: ...


@runtime_checkable
class RequestAttemptBackend(Protocol):
    """Notify host accounting before each provider request, including recovery."""

    request_attempt_callback: Callable[[], None] | None

    def set_request_attempt_callback(self, callback: Callable[[], None] | None) -> None: ...


@runtime_checkable
class IsolatedBackend(Protocol):
    """Create independent provider conversation state for a helper run."""

    def new_run_backend(self) -> Any: ...


_SINGLE_MODEL_REQUEST: ContextVar[bool] = ContextVar("looplet_single_model_request", default=False)


@contextmanager
def single_model_request() -> Iterator[None]:
    """Require one admitted provider attempt, with no hidden SDK retry."""
    token = _SINGLE_MODEL_REQUEST.set(True)
    try:
        yield
    finally:
        _SINGLE_MODEL_REQUEST.reset(token)


def single_model_request_active() -> bool:
    """Whether the caller owns admission of each individual provider attempt."""
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
        "validate_checkpoint_state",
        "reset_conversation",
        "conversation_mode",
        "set_conversation_input",
        "apply_context_compaction",
        "set_model_turn_context",
        "record_tool_result",
        "set_run_envelope",
        "new_run_backend",
        "request_budgets",
        "request_policy",
        "request_attempt_callback",
        "set_request_attempt_callback",
        "configure_request_options",
        "supported_request_options",
        "request_configuration",
        "close",
        "_client",
        "_state_path",
        "_restore_local_state",
        "_looplet_manages_retries",
        "_supports_single_model_request",
    }
)
MUTABLE_CAPABILITIES = frozenset(
    {"last_usage", "request_attempt_callback", "_client", "_state_path"}
)


class BackendWrapper:
    """Optional base for wrappers that preserve independent backend capabilities.

    A plain instance delegates generation without changing its signature or
    sync/async nature. An observing wrapper can override generation and use
    :func:`invoke_backend` to preserve legacy signatures. A delegated factory
    returns the backend's fresh instance; hosts choose its child-run wrappers.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __getattr__(self, name: str) -> Any:
        backend = object.__getattribute__(self, "backend")
        if name in {"generate", "generate_with_tools"}:
            return getattr(backend, name)
        return backend_attribute(backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in MUTABLE_CAPABILITIES:
            setattr(object.__getattribute__(self, "backend"), name, value)
        else:
            object.__setattr__(self, name, value)


def json_snapshot(value: Any) -> Any:
    """Copy JSON-compatible values and provider models without live references."""

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
    return 1 if backend_request_policy(backend).manages_retries else max_retries + 1
