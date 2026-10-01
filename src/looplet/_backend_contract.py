"""Internal composition rules shared by transparent backend wrappers."""

from __future__ import annotations

import inspect
from typing import Any, Callable

FORWARDED_CAPABILITIES = frozenset(
    {
        "last_usage",
        "last_model",
        "checkpoint_state",
        "restore_checkpoint_state",
        "reset_conversation",
        "record_tool_result",
        "set_run_envelope",
        "close",
        "_client",
        "_state_path",
        "_restore_local_state",
        "_looplet_manages_retries",
    }
)
MUTABLE_CAPABILITIES = frozenset({"last_usage", "_client", "_state_path"})


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
