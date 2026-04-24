"""Smoke tests for the looplet package scaffold."""

import pytest

pytestmark = pytest.mark.smoke


def test_version_is_string() -> None:
    import looplet

    assert isinstance(looplet.__version__, str)


def test_version_value() -> None:
    import looplet

    assert looplet.__version__ == "0.1.8"


def test_package_docstring() -> None:
    import looplet

    assert looplet.__doc__ is not None
    assert "looplet" in looplet.__doc__.lower() or "looplet" in looplet.__doc__.lower()


def test_mock_llm_backend_no_external_imports(mock_llm) -> None:
    """MockLLMBackend must work without any external package imports."""
    response = mock_llm.generate("hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_mock_llm_backend_scripted(mock_llm_scripted) -> None:
    backend = mock_llm_scripted(["first", "second", "third"])
    assert backend.generate("q1") == "first"
    assert backend.generate("q2") == "second"
    assert backend.generate("q3") == "third"
    assert backend.generate("q4") == "first"  # cycles


def test_mock_llm_backend_reset(mock_llm_scripted) -> None:
    backend = mock_llm_scripted(["a", "b"])
    backend.generate("x")
    backend.reset()
    assert backend.generate("y") == "a"


def test_sample_task_fixture(sample_task) -> None:
    assert "id" in sample_task
    assert "description" in sample_task
    assert isinstance(sample_task["id"], str)
    assert isinstance(sample_task["description"], str)
