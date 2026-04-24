"""default_max_tokens=None lets API decide."""

from __future__ import annotations

import pytest

from looplet.backends import OpenAIBackend, _resolve_max_tokens

pytestmark = pytest.mark.smoke


class TestResolveMaxTokens:
    def test_per_call_wins(self):
        assert _resolve_max_tokens(500, 2000) == 500

    def test_default_wins_when_per_call_zero(self):
        assert _resolve_max_tokens(0, 2000) == 2000

    def test_none_default_returns_per_call(self):
        assert _resolve_max_tokens(500, None) == 500

    def test_both_none_returns_none(self):
        assert _resolve_max_tokens(0, None) is None

    def test_both_set_per_call_wins(self):
        assert _resolve_max_tokens(300, 1000) == 300


class TestDefaultMaxTokensNone:
    def test_default_is_none(self):
        llm = OpenAIBackend(base_url="http://localhost:9999/v1", api_key="x")
        assert llm._default_max_tokens is None

    def test_explicit_default(self):
        llm = OpenAIBackend(
            base_url="http://localhost:9999/v1",
            api_key="x",
            default_max_tokens=4096,
        )
        assert llm._default_max_tokens == 4096
