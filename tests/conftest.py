"""Common test fixtures for the openharness test suite.

The ``MockLLMBackend`` implementation lives in ``openharness.testing`` so
downstream packages can reuse it; this conftest only exposes pytest
fixtures around it.
"""

from __future__ import annotations

from typing import Any

import pytest

from openharness.testing import MockLLMBackend


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
