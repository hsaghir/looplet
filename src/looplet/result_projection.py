"""Bounded model views, independent of complete tool evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from looplet.context_budget import (
    get_context_inline_per_step_chars,
    get_context_window_total_chars,
)
from looplet.types import ToolResult


def model_result_payload(
    data: Any,
    *,
    error: str | None = None,
    result_key: str | None = None,
    warnings: list[str] | None = None,
    delivery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the same envelope for text and provider-native results."""
    payload: dict[str, Any] = {}
    if result_key:
        payload["result_key"] = result_key
    payload.update(data=data, error=error)
    if warnings:
        payload["warnings"] = list(warnings)
    if delivery:
        payload["delivery"] = dict(delivery)
    return payload


def result_payload(result: ToolResult) -> dict[str, Any]:
    return model_result_payload(
        result.data,
        error=result.error,
        result_key=result.result_key,
        warnings=result.warnings,
        delivery=result.metadata.get("model_projection", {}).get("delivery"),
    )


def payload_text(result: ToolResult) -> str:
    return json.dumps(result_payload(result), ensure_ascii=False, default=str)


def _refuse(result: ToolResult, error: str, limit: int) -> ToolResult:
    from looplet.scaffolding import ContextOverflowError

    prior = result.metadata.get("model_projection", {}).get("delivery", {})
    delivery = {**prior, "status": "refused", "reason": error.partition(":")[0], "max_chars": limit}
    data = result.data
    if isinstance(data, dict) and isinstance(data.get("content"), str):
        offset = data.get("offset")
        if type(offset) is int:
            delivery["offset"] = offset
            content = data["content"]
            lower, upper = 0, len(content)
            while lower < upper:
                middle = (lower + upper + 1) // 2
                page = {**data, "content": content[:middle], "next_offset": offset + middle}
                candidate = replace(
                    result,
                    data=page,
                    metadata={**result.metadata, "model_projection": {"max_chars": limit}},
                )
                if len(payload_text(candidate)) <= limit:
                    lower = middle
                else:
                    upper = middle - 1
            delivery["suggested_limit"] = lower or None
        if data.get("result_key"):
            delivery["result_key"] = data["result_key"]
    refused = replace(
        result,
        data=None,
        warnings=[],
        metadata={
            **result.metadata,
            "model_projection": {"max_chars": limit, "refused": True, "delivery": delivery},
        },
    )
    if len(payload_text(refused)) > limit:
        if not result.error:
            raise ContextOverflowError("Tool result control envelope exceeds max_chars")
        delivery["error_truncated"] = True
        lower, upper = 0, len(result.error)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            candidate = replace(refused, error=result.error[:middle])
            if len(payload_text(candidate)) <= limit:
                lower = middle
            else:
                upper = middle - 1
        if not lower:
            raise ContextOverflowError("Tool result control envelope exceeds max_chars")
        refused = replace(refused, error=result.error[:lower])
    return refused


def project_tool_result(
    result: ToolResult,
    *,
    max_chars: int | None = None,
    persist_dir: str | None = None,
    persist_threshold: int = 0,
) -> ToolResult:
    """Preserve a bounded view intact or return an explicit delivery refusal."""
    from looplet import context_budget
    from looplet.scaffolding import truncate_tool_result

    hints = result.metadata.get("render", {})
    default_limit = get_context_inline_per_step_chars()
    if not result.metadata.get("result_is_view"):
        default_limit = min(default_limit, context_budget.TOOL_RESULT_MAX_CHARS)
    limit = hints.get("max_chars", default_limit) if max_chars is None else max_chars
    if type(limit) is not int or limit < 1:
        raise ValueError("Tool result max_chars must be a positive integer")
    prior = result.metadata.get("model_projection", {})
    if prior.get("refused"):
        return _refuse(result, prior["delivery"]["reason"], limit)
    metadata = {**result.metadata, "model_projection": {"max_chars": limit}}
    candidate = replace(result, metadata=metadata)
    if result.metadata.get("result_is_view"):
        if len(payload_text(candidate)) <= limit:
            return candidate
        return _refuse(
            candidate,
            "result_view_too_large: request a smaller limit; no page was delivered.",
            limit,
        )
    if not result.error:
        candidate = replace(
            candidate,
            data=truncate_tool_result(
                result.data,
                max_chars=limit,
                max_rows=hints.get("preview", context_budget.TOOL_RESULT_MAX_ROWS),
                persist_dir=persist_dir,
                persist_threshold=persist_threshold,
            ),
        )
    if len(payload_text(candidate)) <= limit:
        return candidate

    preview_key = "preview"
    prior = result.metadata.get("model_projection")
    if prior and isinstance(result.data, dict) and result.data.get("truncated"):
        preview_key = "truncated_output" if "truncated_output" in result.data else "preview"
        raw = result.data.get(preview_key, "")
        data = {key: value for key, value in result.data.items() if key != preview_key}
    else:
        raw = (
            result.data
            if isinstance(result.data, str)
            else json.dumps(result.data, ensure_ascii=False, default=str)
        )
        data = {
            "truncated": True,
            "total_chars": len(raw),
            "note": f"[truncated; full result {len(raw)} chars]",
        }
        if isinstance(candidate.data, dict) and "persisted_output_path" in candidate.data:
            data.update(
                {key: value for key, value in candidate.data.items() if key != "truncated_output"}
            )
            preview_key = "truncated_output"
    candidate = replace(candidate, data=data)
    if len(payload_text(replace(candidate, data={**data, preview_key: ""}))) > limit:
        return _refuse(
            candidate,
            "result_metadata_too_large: result remains in run evidence.",
            limit,
        )
    lower, upper = 0, min(len(raw), limit)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        preview = replace(candidate, data={**data, preview_key: raw[:middle]})
        if len(payload_text(preview)) <= limit:
            lower = middle
        else:
            upper = middle - 1
    return replace(candidate, data={**data, preview_key: raw[:lower]})


def bound_result_batch(results: list[ToolResult], *, max_chars: int) -> list[ToolResult]:
    """Shrink the largest projections without cutting a derived view's cursor."""
    from looplet.scaffolding import ContextOverflowError

    projected = list(results)
    sizes = [len(payload_text(result)) for result in projected]
    eligible = set(range(len(results)))
    while sum(sizes) > max_chars:
        if not eligible:
            raise ContextOverflowError("Tool result envelopes cannot fit the aggregate budget")
        largest = max(sorted(eligible), key=sizes.__getitem__)
        try:
            replacement = project_tool_result(
                projected[largest], max_chars=max(1, sizes[largest] // 2)
            )
        except ContextOverflowError:
            try:
                replacement = project_tool_result(
                    projected[largest], max_chars=max(1, sizes[largest] - 1)
                )
            except ContextOverflowError:
                eligible.remove(largest)
                continue
        size = len(payload_text(replacement))
        if size >= sizes[largest]:
            eligible.remove(largest)
            continue
        projected[largest] = replacement
        sizes[largest] = size
    return projected


def project_result_batches(conversation: Any) -> None:
    """Freeze bounded model views before the next request, leaving raw steps alone."""
    batch = []

    def finish() -> None:
        results = bound_result_batch(
            [message.tool_result for message in batch],
            max_chars=get_context_window_total_chars(),
        )
        for message, result in zip(batch, results):
            message.tool_result = result
        batch.clear()

    for message in getattr(conversation, "messages", []):
        if getattr(message.role, "value", message.role) == "assistant" and batch:
            finish()
        if message.tool_result is not None:
            batch.append(message)
    if batch:
        finish()
