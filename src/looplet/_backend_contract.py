"""Compatibility aliases for the now-public :mod:`looplet.backend_contract`."""

from looplet.backend_contract import (
    FORWARDED_CAPABILITIES,
    MUTABLE_CAPABILITIES,
    TerminalLLMError,
    attempt_limit,
    backend_attribute,
    invoke_backend,
    json_snapshot,
    single_model_request,
    single_model_request_active,
)

__all__ = [
    "FORWARDED_CAPABILITIES",
    "MUTABLE_CAPABILITIES",
    "TerminalLLMError",
    "attempt_limit",
    "backend_attribute",
    "invoke_backend",
    "json_snapshot",
    "single_model_request",
    "single_model_request_active",
]
