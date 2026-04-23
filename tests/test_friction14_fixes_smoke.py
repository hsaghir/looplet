"""Round-14 friction fix: PerToolLimitHook default_limit."""

from __future__ import annotations

import pytest

from looplet.limits import PerToolLimitHook
from looplet.types import ToolCall, ToolResult

pytestmark = pytest.mark.smoke


class TestPerToolLimitDefaultLimit:
    def test_default_limit_caps_all_tools(self):
        hook = PerToolLimitHook(default_limit=2)
        tc = ToolCall(tool="search", args={"q": "x"}, reasoning="r")
        # First 2 calls pass
        assert hook.pre_dispatch(None, None, tc, 1) is None
        assert hook.pre_dispatch(None, None, tc, 2) is None
        # 3rd call blocked
        result = hook.pre_dispatch(None, None, tc, 3)
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert "limit" in result.error.lower()

    def test_limits_override_default(self):
        hook = PerToolLimitHook(default_limit=10, limits={"search": 1})
        tc_search = ToolCall(tool="search", args={}, reasoning="r")
        tc_other = ToolCall(tool="other", args={}, reasoning="r")
        # search: limit 1
        assert hook.pre_dispatch(None, None, tc_search, 1) is None
        result = hook.pre_dispatch(None, None, tc_search, 2)
        assert result is not None and result.error
        # other: limit 10 (default)
        for i in range(10):
            assert hook.pre_dispatch(None, None, tc_other, i) is None
        result = hook.pre_dispatch(None, None, tc_other, 11)
        assert result is not None and result.error

    def test_requires_limits_or_default(self):
        with pytest.raises(TypeError, match="at least one"):
            PerToolLimitHook()

    def test_limits_only_no_default(self):
        hook = PerToolLimitHook(limits={"bash": 3})
        tc_bash = ToolCall(tool="bash", args={}, reasoning="r")
        tc_other = ToolCall(tool="unlisted", args={}, reasoning="r")
        # bash limited
        for _ in range(3):
            assert hook.pre_dispatch(None, None, tc_bash, 1) is None
        assert hook.pre_dispatch(None, None, tc_bash, 4) is not None
        # unlisted unlimited (no default_limit set)
        for i in range(100):
            assert hook.pre_dispatch(None, None, tc_other, i) is None

    def test_negative_default_limit_raises(self):
        with pytest.raises(ValueError, match="default_limit"):
            PerToolLimitHook(default_limit=-1)
