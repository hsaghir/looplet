"""OpenAIBackend.tool_choice parameter for native tool calling."""

from __future__ import annotations

import pytest

from looplet.backends import OpenAIBackend

pytestmark = pytest.mark.smoke


class TestToolChoice:
    def test_default_tool_choice_is_auto(self):
        llm = OpenAIBackend(base_url="http://localhost:9999/v1", api_key="x")
        assert llm._tool_choice == "auto"

    def test_custom_tool_choice(self):
        llm = OpenAIBackend(
            base_url="http://localhost:9999/v1",
            api_key="x",
            tool_choice="required",
        )
        assert llm._tool_choice == "required"

    def test_tool_choice_with_explicit_client(self):
        from unittest.mock import MagicMock

        client = MagicMock()
        llm = OpenAIBackend(client, model="gpt-4o", tool_choice="none")
        assert llm._tool_choice == "none"
