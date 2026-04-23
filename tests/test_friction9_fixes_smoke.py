"""Round-9 friction fixes: tool-name collision + validation batch args."""

from __future__ import annotations

import logging

import pytest

from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import ToolCall
from looplet.validation import FieldSpec, OutputSchema, ValidatingToolRegistry

pytestmark = pytest.mark.smoke


class TestToolNameCollisionWarning:
    def test_duplicate_name_logs_warning(self, caplog):
        reg = BaseToolRegistry()
        reg.register(ToolSpec(name="bash", description="v1", parameters={}, execute=lambda: None))
        with caplog.at_level(logging.WARNING, logger="looplet.tools"):
            reg.register(
                ToolSpec(
                    name="bash",
                    description="v2",
                    parameters={},
                    execute=lambda: None,
                )
            )
        assert any("already registered" in rec.message for rec in caplog.records)

    def test_no_warning_on_first_register(self, caplog):
        reg = BaseToolRegistry()
        with caplog.at_level(logging.WARNING, logger="looplet.tools"):
            reg.register(
                ToolSpec(name="bash", description="v1", parameters={}, execute=lambda: None)
            )
        assert not any("already registered" in rec.message for rec in caplog.records)


class TestValidatingBatchArgsSummary:
    def test_batch_error_uses_kv_formatting(self):
        reg = ValidatingToolRegistry()
        schema = OutputSchema(fields={"n": FieldSpec(name="n", field_type="int", required=True)})
        reg.register_with_schema(
            ToolSpec(
                name="square",
                description="square",
                parameters={"n": "int"},
                execute=lambda *, n: {"sq": n * n},
            ),
            schema,
        )
        call = ToolCall(tool="square", args={"n": "oops"}, reasoning="r")
        result = reg.dispatch_batch([call])[0]
        assert result.error is not None
        # Previously: "{'n': 'oops'}"  — dict repr, truncated at 100 chars.
        # Now: consistent k=v formatting like the single-dispatch path.
        assert result.args_summary == "n=oops"
