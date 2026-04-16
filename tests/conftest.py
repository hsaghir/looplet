"""Common test fixtures for the cadence test suite."""

from __future__ import annotations

from typing import Any

import pytest


class MockLLMBackend:
    """Scripted LLM backend for tests — zero cadence imports, plain Python only.

    Accepts a list of responses at construction; each call to ``generate``
    returns the next response in the list, cycling if exhausted.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = responses or ["mock response"]
        self._index: int = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response

    def reset(self) -> None:
        self._index = 0


@pytest.fixture
def mock_llm() -> MockLLMBackend:
    """Return a MockLLMBackend with a single default response."""
    return MockLLMBackend()


@pytest.fixture
def mock_llm_scripted():
    """Factory fixture: call with a list of responses to get a scripted backend."""

    def _factory(responses: list[str]) -> MockLLMBackend:
        return MockLLMBackend(responses=responses)

    return _factory


@pytest.fixture
def mock_registry() -> Any:
    """Return a BaseToolRegistry instance (lazy import — only resolves after task 2.2)."""
    from openharness.tools import BaseToolRegistry  # noqa: PLC0415  # lazy import intentional

    return BaseToolRegistry()


@pytest.fixture
def sample_task() -> dict[str, Any]:
    """Return a minimal task dict suitable for pipeline tests."""
    return {
        "id": "test-task-001",
        "description": "A sample task for unit testing the cadence pipeline.",
    }
