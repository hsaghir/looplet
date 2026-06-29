"""make_backend resolves on any host: copilot/proxy, opt-in keyless mock, or raises."""
from __future__ import annotations

import pytest

from looplet.backends import OpenAIBackend, make_backend
from looplet.testing import MockLLMBackend

_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "LOOPLET_PROVIDER",
         "COPILOT_PROXY_URL", "COPILOT_PROXY_KEY", "LOOPLET_TAX_LLM_BASE_URL",
         "LOOPLET_TAX_LLM_API_KEY", "LOOPLET_ALLOW_MOCK"]


def _clear(mp: pytest.MonkeyPatch) -> None:
    for k in _KEYS:
        mp.delenv(k, raising=False)


def test_keyless_mock_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("LOOPLET_ALLOW_MOCK", "1")
    assert isinstance(make_backend(), MockLLMBackend)


def test_copilot_proxy_resolves_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("COPILOT_PROXY_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("COPILOT_PROXY_KEY", "x")
    assert isinstance(make_backend(), OpenAIBackend)


def test_no_creds_no_mock_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(ValueError, match="LOOPLET_ALLOW_MOCK"):
        make_backend()
