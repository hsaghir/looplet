"""Dogfood tests for capability-scoped hook views (``looplet.hook_view``).

A view is the *only* state an out-of-process hook is allowed to read.
These tests pin two contract guarantees:

* ``ViewSpec`` rejects unknown fields and invalid fidelity at
  construction time (fail-loud, not silent drop);
* ``extract_view`` projects **only** the declared fields and produces a
  JSON-safe payload (so it can cross a process boundary intact).
"""

from __future__ import annotations

import json

import pytest

from looplet.hook_view import (
    KNOWN_VIEW_FIELDS,
    ViewSpec,
    extract_view,
)
from looplet.types import ToolCall, ToolResult


class TestViewSpec:
    def test_defaults(self):
        spec = ViewSpec()
        assert spec.fidelity == "digest"
        assert spec.fields == frozenset()

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            ViewSpec(fields=frozenset({"not_a_field"}))

    def test_invalid_fidelity_rejected(self):
        with pytest.raises(ValueError, match="fidelity"):
            ViewSpec(fields=frozenset({"tool"}), fidelity="telepathic")

    def test_from_dict_accepts_none_list_dict(self):
        assert ViewSpec.from_dict(None).fields == frozenset()
        assert ViewSpec.from_dict(["tool", "args"]).fields == frozenset({"tool", "args"})
        spec = ViewSpec.from_dict({"fields": ["tool"], "fidelity": "full"})
        assert spec.fields == frozenset({"tool"})
        assert spec.fidelity == "full"

    def test_to_dict_roundtrip(self):
        spec = ViewSpec(fields=frozenset({"tool", "args"}), fidelity="full")
        again = ViewSpec.from_dict(spec.to_dict())
        assert again == spec

    def test_known_fields_are_all_extractable(self):
        # Every advertised field must be a real key extract_view can emit.
        assert "tool" in KNOWN_VIEW_FIELDS
        assert "transcript" in KNOWN_VIEW_FIELDS


class TestExtractView:
    def test_projects_only_declared_fields(self):
        spec = ViewSpec(fields=frozenset({"tool", "args"}))
        call = ToolCall(tool="rm", args={"path": "/x"}, reasoning="cleanup")
        view = extract_view(spec, tool_call=call, step=3)
        assert set(view) == {"tool", "args"}
        assert view["tool"] == "rm"
        assert view["args"] == {"path": "/x"}
        # ``reasoning`` and ``step`` were not declared → not present.
        assert "reasoning" not in view
        assert "step" not in view

    def test_step_field(self):
        spec = ViewSpec(fields=frozenset({"step"}))
        view = extract_view(spec, step=7)
        assert view == {"step": 7}

    def test_result_view(self):
        spec = ViewSpec(fields=frozenset({"tool_result"}))
        result = ToolResult(tool="ls", args_summary="ls", data=["a", "b"])
        view = extract_view(spec, tool_result=result)
        assert "tool_result" in view
        # JSON-safe - must serialise without error.
        json.dumps(view)

    def test_payload_is_json_safe(self):
        spec = ViewSpec(fields=frozenset({"tool", "args"}))
        call = ToolCall(tool="x", args={"n": 1, "nested": {"k": [1, 2, 3]}})
        view = extract_view(spec, tool_call=call)
        # round-trips through JSON unchanged
        assert json.loads(json.dumps(view)) == view

    def test_empty_spec_emits_nothing(self):
        view = extract_view(ViewSpec(), tool_call=ToolCall(tool="x", args={}), step=1)
        assert view == {}
