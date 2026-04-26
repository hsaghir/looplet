"""Tests for reusable native tool protocol probing."""

from __future__ import annotations

import pytest

from looplet import NativeToolProbeResult, probe_native_tool_support, supports_native_tools

pytestmark = pytest.mark.smoke


class TestNativeToolProbe:
    def test_accepts_backend_that_returns_tool_use_block(self) -> None:
        class Backend:
            def generate_with_tools(self, *args, **kwargs):
                return [{"type": "tool_use", "name": "test_probe", "input": {}}]

        result = probe_native_tool_support(Backend())

        assert isinstance(result, NativeToolProbeResult)
        assert result.supported is True
        assert supports_native_tools(Backend()) is True

    def test_rejects_backend_without_generate_with_tools(self) -> None:
        class Backend:
            def generate(self, *args, **kwargs):
                return "{}"

        result = probe_native_tool_support(Backend())

        assert result.supported is False
        assert "no generate_with_tools" in result.reason

    def test_rejects_proxy_that_ignores_tools_and_returns_text(self) -> None:
        class Backend:
            def generate_with_tools(self, *args, **kwargs):
                return [{"type": "text", "text": '{"tool": "test_probe"}'}]

        result = probe_native_tool_support(Backend())

        assert result.supported is False
        assert "no matching tool_use" in result.reason
        assert supports_native_tools(Backend()) is False

    def test_rejects_backend_when_probe_raises(self) -> None:
        class Backend:
            def generate_with_tools(self, *args, **kwargs):
                raise RuntimeError("tools unsupported")

        result = probe_native_tool_support(Backend())

        assert result.supported is False
        assert "RuntimeError" in result.reason

    def test_custom_probe_tool_name(self) -> None:
        class Backend:
            def generate_with_tools(self, *args, **kwargs):
                return [{"type": "tool_use", "name": "custom_probe", "input": {}}]

        assert supports_native_tools(Backend(), tool_name="custom_probe") is True
