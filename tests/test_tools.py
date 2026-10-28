"""Tests for looplet.tools - ToolSpec, BaseToolRegistry, register_think_tool."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_original_results_are_detached_and_retention_is_host_controlled():
    from looplet import ToolCall, tools_from
    from looplet.tools import BaseToolRegistry

    data = {"rows": [{"value": "original"}]}
    registry = tools_from([lambda: data])
    result = registry.dispatch(ToolCall(tool="<lambda>", args={}))
    data["rows"][0]["value"] = "changed"
    assert registry.get_result(result.result_key) == {"rows": [{"value": "original"}]}
    recalled = registry.get_result(result.result_key)
    recalled["rows"].clear()
    assert registry.get_result(result.result_key)["rows"]
    unretained = BaseToolRegistry(retain_results=False)
    assert unretained._store_result(ToolCall(tool="source"), data) is None
    assert unretained.snapshot_results() == {}


def test_result_tools_read_search_and_discover_full_retained_evidence():
    from looplet import ToolCall, tools_from

    text = "unrelated line\n" * 1000 + "answer=405\n"
    original = {"stdout": text, "a/b": {"~key": [1, 2, 3]}}
    registry = tools_from([lambda: original], include_result_tools=True)
    source = registry.dispatch(ToolCall(tool="<lambda>"))
    key = source.result_key
    assert key is not None

    def retrieve(name, **args):
        result = registry.dispatch(ToolCall(tool=name, args=args))
        assert result.error is None, result.error
        return result.data

    listed = retrieve("list_tool_results")
    assert key in [item["result_key"] for item in listed["results"]]
    page = retrieve(
        "read_tool_result", result_key=key, json_pointer="/stdout", offset=15_000, limit=100
    )
    assert page["content"] == "answer=405\n"
    assert page["next_offset"] is None
    assert page["offset_unit"] == "characters"
    found = retrieve(
        "search_tool_result", result_key=key, json_pointer="/stdout", query="answer=405"
    )
    assert found["matches"][0]["offset"] == 15_000
    assert "answer=405" in found["matches"][0]["excerpt"]
    escaped = retrieve("read_tool_result", result_key=key, json_pointer="/a~1b/~0key/2")
    assert escaped["content"] == "3"
    assert registry.get_result(key) == original


def test_result_tools_are_an_explicit_host_capability():
    from looplet import tools_from

    assert not {"list_tool_results", "read_tool_result", "search_tool_result"}.intersection(
        tools_from([]).tool_names
    )


def test_result_catalog_does_not_enumerate_its_own_views():
    from looplet import ToolCall, tools_from

    def source(*, value: str) -> str:
        return value

    registry = tools_from([source], include_result_tools=True)
    originals = [
        registry.dispatch(ToolCall(tool="source", args={"value": value}))
        for value in ("first evidence", "second evidence")
    ]
    for offset, original in enumerate(originals):
        listed = registry.dispatch(
            ToolCall(tool="list_tool_results", args={"offset": offset, "limit": 1})
        )
        assert listed.error is None
        assert listed.data["total_results"] == len(originals)
        assert listed.data["results"][0]["result_key"] == original.result_key
        expected_next = offset + 1 if offset + 1 < len(originals) else None
        assert listed.data["next_offset"] == expected_next

    for name in ("read_tool_result", "search_tool_result"):
        args = {"result_key": originals[0].result_key}
        if name == "search_tool_result":
            args["query"] = "evidence"
        retrieved = registry.dispatch(ToolCall(tool=name, args=args))
        assert retrieved.error is None
        assert "evidence" in str(retrieved.data)
    assert registry.list_results()["total_results"] == len(originals)


def test_result_tools_preserve_unicode_and_page_continuations():
    from looplet import ToolCall, tools_from

    original = "row-\U0001f600\n" * 400
    registry = tools_from([lambda: original], include_result_tools=True)
    key = registry.dispatch(ToolCall(tool="<lambda>")).result_key
    parts = []
    offset = 0
    while True:
        page = registry.read_result(key, offset=offset, limit=37)
        parts.append(page["content"])
        assert page["total_chars"] == len(original)
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert "".join(parts) == original


def test_result_tools_are_scoped_to_the_current_or_host_selected_store():
    from looplet import ToolCall, tools_from

    first = tools_from([lambda: "first run only"], include_result_tools=True)
    key = first.dispatch(ToolCall(tool="<lambda>")).result_key
    second = tools_from([], include_result_tools=True)
    denied = second.dispatch(ToolCall(tool="read_tool_result", args={"result_key": key}))
    assert denied.error
    second.set_resources({"tool_result_store": first})
    allowed = second.dispatch(ToolCall(tool="read_tool_result", args={"result_key": key}))
    assert allowed.error is None
    assert allowed.data["content"] == "first run only"


@pytest.mark.parametrize("name", ["read_tool_result", "search_tool_result"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("offset", -1),
        ("offset", True),
        ("limit", 0),
        ("limit", 100_000),
        ("json_pointer", "not/a/pointer"),
        ("json_pointer", "/bad~2escape"),
    ],
)
def test_result_tools_reject_invalid_or_unbounded_requests(name, field, value):
    from looplet import ToolCall, tools_from

    registry = tools_from([lambda: {"stdout": "evidence"}], include_result_tools=True)
    key = registry.dispatch(ToolCall(tool="<lambda>")).result_key
    args = {"result_key": key, field: value}
    if name == "search_tool_result":
        args["query"] = "evidence"
    result = registry.dispatch(ToolCall(tool=name, args=args))
    assert result.error
    assert registry.get_result(key) == {"stdout": "evidence"}


def test_result_tools_search_is_literal_and_resumable():
    from looplet import ToolCall, tools_from

    registry = tools_from([lambda: "a.*b.*c.*"], include_result_tools=True)
    key = registry.dispatch(ToolCall(tool="<lambda>")).result_key
    first = registry.search_result(key, ".*", limit=1)
    second = registry.search_result(key, ".*", offset=first["next_offset"], limit=2)
    assert [match["offset"] for match in first["matches"] + second["matches"]] == [1, 4, 7]
    assert second["next_offset"] is None


def test_result_tools_native_schemas_preserve_required_and_optional_fields():
    from looplet import tools_from

    schemas = {
        schema["name"]: schema["input_schema"]
        for schema in tools_from([], include_result_tools=True).tool_view().schemas
    }
    assert schemas["list_tool_results"].get("required", []) == []
    assert schemas["read_tool_result"]["required"] == ["result_key"]
    assert schemas["search_tool_result"]["required"] == ["result_key", "query"]


def test_file_result_store_is_a_writable_host_resource(tmp_path, monkeypatch):
    from looplet import ToolCall, tools_from
    from looplet.builtin_tools import tool_results
    from looplet.result_store import FileToolResultStore

    original = {"nested": {"archive": "row-\U0001f600\r\n" * 4000 + "answer=405\r\n"}}
    store = FileToolResultStore(str(tmp_path / "evidence"))
    registry = tools_from([lambda: original], include_result_tools=True)
    registry.set_resources({"tool_result_store": store})
    result = registry.dispatch(ToolCall(tool="<lambda>"))
    assert result.error is None
    assert result.result_key is not None
    assert registry.get_result(result.result_key) == original
    assert store.get_result(result.result_key) == original
    assert registry.list_results()["total_results"] == 1

    serializations = []
    original_text = tool_results._text

    def observe_text(data, pointer):
        if pointer or not isinstance(data, str):
            serializations.append(pointer)
        return original_text(data, pointer)

    monkeypatch.setattr(tool_results, "_text", observe_text)
    page = registry.read_result(result.result_key, json_pointer="/nested/archive", limit=37)
    assert page["content"] == original["nested"]["archive"][:37]
    found = registry.search_result(result.result_key, "answer=", json_pointer="/nested/archive")
    assert "answer=405" in found["matches"][0]["excerpt"]
    reopened = FileToolResultStore(str(tmp_path / "evidence"))
    assert reopened.read_result(result.result_key, json_pointer="/nested/archive", limit=37) == page
    assert serializations == ["/nested/archive"]
    assert reopened.snapshot_results() == registry.snapshot_results()
    restored = FileToolResultStore(str(tmp_path / "restored"))
    restored.restore_results(registry.snapshot_results())
    assert restored.get_result(result.result_key) == original
    assert registry.list_results()["total_results"] == 1


@pytest.mark.parametrize("offset", [0, 65500, 65535, 65540, 131100])
@pytest.mark.parametrize("query", ["NEEDLE", "x" * 256, "missing", "\U0001f600"])
@pytest.mark.parametrize("limit", [1, 2, 5])
def test_cached_result_views_use_bounded_reads(tmp_path, monkeypatch, offset, query, limit):
    from pathlib import Path

    from looplet.builtin_tools.tool_results import read_result, search_result
    from looplet.result_store import FileToolResultStore

    text = "x" * 65533 + "NEEDLE" + "\r\n\U0001f600line\n" * 9000 + "NEEDLEtail"
    store = FileToolResultStore(str(tmp_path))
    key = store.store_result({"body": text})
    store.read_result(key, json_pointer="/body", limit=1)
    reopened = FileToolResultStore(str(tmp_path))
    original_open = Path.open

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size=-1):
            assert 0 <= size <= 65536
            return self.stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BoundedReader(stream) if path.suffix == ".txt" else stream

    monkeypatch.setattr(Path, "open", open_path)
    expected_page = read_result(text, result_key=key, json_pointer="", offset=offset, limit=37)
    expected_page["json_pointer"] = "/body"
    assert reopened.read_result(key, json_pointer="/body", offset=offset, limit=37) == expected_page
    expected_search = search_result(
        text, result_key=key, query=query, json_pointer="", offset=offset, limit=limit
    )
    expected_search["json_pointer"] = "/body"
    assert (
        reopened.search_result(key, query, json_pointer="/body", offset=offset, limit=limit)
        == expected_search
    )


def test_file_result_store_quota_does_not_discard_successful_tool_output(tmp_path):
    from looplet import ToolCall, tools_from
    from looplet.result_store import FileToolResultStore

    original = {"archive": "retained evidence" * 1000}
    registry = tools_from([lambda: original], include_result_tools=True)
    registry.set_resources({"tool_result_store": FileToolResultStore(str(tmp_path), max_bytes=32)})
    result = registry.dispatch(ToolCall(tool="<lambda>"))
    assert result.error is None
    assert result.data == original
    assert result.result_key is None
    assert result.warnings
    assert result.metadata["retention_error"]["type"] == "OSError"
    assert registry.list_results()["total_results"] == 0


@pytest.mark.parametrize("key", ["../outside", "sha256:" + "0" * 64])
def test_file_result_store_never_accepts_model_selected_paths(tmp_path, key):
    from looplet.result_store import FileToolResultStore

    with pytest.raises((KeyError, ValueError)):
        FileToolResultStore(str(tmp_path)).get_result(key)


@pytest.mark.parametrize("file_backed", [False, True])
def test_result_listing_can_freeze_its_catalog_boundary(tmp_path, file_backed):
    from looplet import ToolCall
    from looplet.result_store import FileToolResultStore
    from looplet.tools import BaseToolRegistry

    store = FileToolResultStore(str(tmp_path)) if file_backed else BaseToolRegistry()

    def retain(value):
        if file_backed:
            return store.store_result(value)
        return store._store_result(ToolCall(tool="source"), value)

    first_key = retain("first")
    second_key = retain("second")
    first = store.list_results(limit=1)
    assert first["results"][0]["result_key"] == first_key
    retain("arrived later")
    second = store.list_results(
        offset=first["next_offset"], end_offset=first["end_offset"], limit=1
    )
    assert second["results"][0]["result_key"] == second_key
    assert second["total_results"] == 2
    assert second["next_offset"] is None
    assert store.list_results()["total_results"] == 3


# ── ToolSpec tests ────────────────────────────────────────────────


class TestToolSpec:
    def test_creation(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="echo",
            description="Echo the input back",
            parameters={"text": "The text to echo"},
            execute=lambda text="": text,
        )
        assert spec.name == "echo"
        assert spec.description == "Echo the input back"
        assert "text" in spec.parameters
        assert spec.concurrent_safe is False
        assert spec.free is False

    def test_concurrent_safe_default(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(name="x", description="d", parameters={}, execute=lambda: None)
        assert spec.concurrent_safe is False

    def test_free_default(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(name="x", description="d", parameters={}, execute=lambda: None)
        assert spec.free is False

    def test_to_api_schema(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="search",
            description="Search for events",
            parameters={"query": "SQL query string", "limit": "Max rows"},
            execute=lambda query="", limit=10: [],
        )
        schema = spec.to_api_schema()
        assert schema["name"] == "search"
        assert schema["description"] == "Search for events"
        assert "input_schema" in schema
        props = schema["input_schema"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "limit" in props

    @pytest.mark.parametrize(
        "declaration, expected_type, optional",
        [
            ("int", "integer", False),
            ("list", "array", False),
            ("(optional) integer", "integer", True),
            ("(optional) array", "array", True),
            (["integer", "null"], ["integer", "null"], False),
        ],
    )
    def test_workspace_descriptors_normalize_native_types(
        self, declaration, expected_type, optional
    ):
        from copy import deepcopy

        from looplet.tools import ToolSpec

        parameters = {
            "answer": {"type": "string"},
            "value": {"type": declaration, "description": "Evidence value", "x-source": "kept"},
        }
        original = deepcopy(parameters)
        spec = ToolSpec("evidence", "Record evidence", parameters, lambda **kwargs: kwargs)
        schema = spec.to_api_schema()["input_schema"]
        assert schema["properties"]["value"] == {
            "type": expected_type,
            "description": "Evidence value",
            "x-source": "kept",
        }
        assert ("value" not in schema["required"]) is optional
        assert "answer" in schema["required"]
        assert parameters == original

    def test_spec_text(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="lookup",
            description="Look up a value",
            parameters={"key": "The key"},
            execute=lambda key="": key,
        )
        text = spec.spec_text()
        assert "lookup" in text
        assert "key" in text
        assert "Look up a value" in text


# ── BaseToolRegistry tests ────────────────────────────────────────


class TestBaseToolRegistry:
    def _make_registry_with_echo(self):
        from looplet.tools import BaseToolRegistry, ToolSpec

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="echo",
                description="Echo back",
                parameters={"text": "text to echo"},
                execute=lambda text="": {"echoed": text},
            )
        )
        return reg

    def test_register_and_tool_names(self):
        reg = self._make_registry_with_echo()
        assert "echo" in reg.tool_names

    def test_dispatch_success(self):
        from looplet.types import ToolCall

        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "hello"})
        result = reg.dispatch(call)
        assert result.error is None
        assert result.data == {"echoed": "hello"}
        assert result.tool == "echo"

    def test_dispatch_timing(self):
        from looplet.types import ToolCall

        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "hi"})
        result = reg.dispatch(call)
        assert result.duration_ms >= 0.0

    def test_dispatch_unknown_tool(self):
        from looplet.tools import BaseToolRegistry
        from looplet.types import ToolCall

        reg = BaseToolRegistry()
        call = ToolCall(tool="nonexistent", args={})
        result = reg.dispatch(call)
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_dispatch_error_handling(self):
        from looplet.tools import BaseToolRegistry, ToolSpec
        from looplet.types import ToolCall

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="boom",
                description="Always raises",
                parameters={},
                execute=lambda: (_ for _ in ()).throw(ValueError("kaboom")),
            )
        )
        call = ToolCall(tool="boom", args={})
        result = reg.dispatch(call)
        assert result.error is not None
        assert "ValueError" in result.error
        assert "kaboom" in result.error
        assert result.data is None

    def test_dispatch_call_id_propagated(self):
        from looplet.types import ToolCall

        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "x"}, call_id="my-id-123")
        result = reg.dispatch(call)
        assert result.call_id == "my-id-123"

    def test_dispatch_strips_dunder_args(self):
        from looplet.types import ToolCall

        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "hi", "__internal": "secret"})
        result = reg.dispatch(call)
        assert result.error is None

    def test_tool_catalog_text(self):
        reg = self._make_registry_with_echo()
        catalog = reg.tool_catalog_text()
        assert "Available tools" in catalog
        assert "echo" in catalog

    def test_tool_view_selects_both_surfaces_without_restricting_dispatch(self):
        from looplet.tools import ToolSpec
        from looplet.types import ToolCall

        reg = self._make_registry_with_echo()
        reg.register(
            ToolSpec(name="hidden", description="Hidden", parameters={}, execute=lambda: 42)
        )

        view = reg.tool_view(("echo",))
        assert view.names == ("echo",)
        assert "echo" in view.catalog_text
        assert "hidden" not in view.catalog_text
        assert [schema["name"] for schema in view.schemas] == ["echo"]
        assert reg.dispatch(ToolCall(tool="hidden", args={})).data == 42
        assert reg.tool_view().catalog_text == reg.tool_catalog_text()
        assert list(reg.tool_view().schemas) == reg.tool_schemas()

        with pytest.raises(ValueError, match="Unknown tools"):
            reg.tool_view(("missing",))
        with pytest.raises(ValueError, match="Duplicate tool names"):
            reg.tool_view(("echo", "echo"))

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.asyncio
    async def test_host_tool_view_selects_prompt_schemas_and_cache_per_turn(self, async_mode):
        from looplet import (
            Conversation,
            DefaultState,
            LoopConfig,
            async_composable_loop,
            composable_loop,
            tool,
            tools_from,
        )
        from looplet.cache import CacheControl, CachePolicy
        from looplet.tools import ToolSpec

        class RecordingBackend:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, **kwargs):
                raise AssertionError("Native view test must use generate_with_tools")

            def generate_with_tools(self, prompt, *, tools, cache_breakpoints=None, **kwargs):
                self.calls.append((prompt, tools, cache_breakpoints))
                name = "echo" if len(self.calls) == 1 else "done"
                args = {"value": "x"} if name == "echo" else {"summary": "ok"}
                return [
                    {
                        "type": "tool_use",
                        "id": f"call-{len(self.calls)}",
                        "name": name,
                        "input": args,
                    }
                ]

        class AsyncRecordingBackend(RecordingBackend):
            async def generate(self, prompt, **kwargs):
                raise AssertionError("Native view test must use generate_with_tools")

            async def generate_with_tools(self, prompt, *, tools, cache_breakpoints=None, **kwargs):
                return super().generate_with_tools(
                    prompt, tools=tools, cache_breakpoints=cache_breakpoints, **kwargs
                )

        @tool(description="Echo a value.")
        def echo(*, value: str) -> dict[str, str]:
            return {"value": value}

        tools = tools_from([echo], include_done=True)
        tools.register(
            ToolSpec(name="hidden", description="Not exposed", parameters={}, execute=lambda: 42)
        )
        selected = []

        def select_tools(*, step_num, state, tools, task):
            assert task == {"goal": "echo then finish"}
            selected.append(step_num)
            return ("echo", "done") if step_num == 1 else ("done",)

        config = LoopConfig(
            max_steps=2,
            tool_view_selector=select_tools,
            build_prompt=lambda **kwargs: kwargs["tool_catalog"],
            cache_policy=CachePolicy(tool_schemas=CacheControl()),
        )
        backend = AsyncRecordingBackend() if async_mode else RecordingBackend()
        conversation = Conversation()
        if async_mode:
            steps = []
            async for step in async_composable_loop(
                llm=backend,
                tools=tools,
                task={"goal": "echo then finish"},
                state=DefaultState(max_steps=2),
                config=config,
                conversation=conversation,
            ):
                steps.append(step)
        else:
            steps = list(
                composable_loop(
                    llm=backend,
                    tools=tools,
                    task={"goal": "echo then finish"},
                    state=DefaultState(max_steps=2),
                    config=config,
                    conversation=conversation,
                )
            )

        assert [step.tool_call.tool for step in steps] == ["echo", "done"]
        assert selected == [1, 2]
        recorded_role = "assistant" if config.uses_typed_history else "user"
        prompt_records = [
            message
            for message in conversation.serialize()["messages"]
            if message["role"] == recorded_role and "tool_view" in message["metadata"]
        ]
        for call, names, record in zip(
            backend.calls, (("echo", "done"), ("done",)), prompt_records, strict=True
        ):
            prompt, schemas, breakpoints = call
            catalog = tools.tool_view(names).catalog_text
            assert prompt.startswith(catalog)
            assert [schema["name"] for schema in schemas] == list(names)
            assert record["metadata"]["tool_view"] == {
                "names": list(names),
                "version": tools.tool_view(names).version,
            }
            assert breakpoints[0].label == "tool_schemas"
            assert breakpoints[0].content == catalog
            assert "hidden" not in prompt

    def test_to_api_schema(self):
        from looplet.tools import BaseToolRegistry

        reg = self._make_registry_with_echo()
        schemas = reg.tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"


# ── Batch dispatch tests ──────────────────────────────────────────


class TestBatchDispatch:
    def _make_registry(self):
        from looplet.tools import BaseToolRegistry, ToolSpec

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="read",
                description="Read-only concurrent-safe",
                parameters={"key": "key"},
                execute=lambda key="": f"value:{key}",
                concurrent_safe=True,
            )
        )
        reg.register(
            ToolSpec(
                name="write",
                description="Write - serial only",
                parameters={"key": "key", "val": "value"},
                execute=lambda key="", val="": f"wrote:{key}={val}",
                concurrent_safe=False,
            )
        )
        return reg

    def test_dispatch_batch_empty(self):
        from looplet.tools import BaseToolRegistry

        reg = BaseToolRegistry()
        assert reg.dispatch_batch([]) == []

    def test_dispatch_batch_returns_all_results(self):
        from looplet.types import ToolCall

        reg = self._make_registry()
        calls = [
            ToolCall(tool="read", args={"key": "a"}),
            ToolCall(tool="read", args={"key": "b"}),
            ToolCall(tool="write", args={"key": "c", "val": "1"}),
        ]
        results = reg.dispatch_batch(calls)
        assert len(results) == 3
        assert all(r.error is None for r in results)

    def test_partition_concurrent_vs_serial(self):
        from looplet.types import ToolCall

        reg = self._make_registry()
        calls = [
            ToolCall(tool="read", args={"key": "a"}),
            ToolCall(tool="read", args={"key": "b"}),
            ToolCall(tool="write", args={"key": "c", "val": "1"}),
        ]
        batches = reg._partition_calls(calls)
        # First batch: 2 concurrent reads
        assert batches[0]["concurrent"] is True
        assert len(batches[0]["calls"]) == 2
        # Second batch: 1 serial write
        assert batches[1]["concurrent"] is False
        assert len(batches[1]["calls"]) == 1

    def test_concurrent_batch_dispatches_parallel(self):
        from looplet.types import ToolCall

        reg = self._make_registry()
        calls = [ToolCall(tool="read", args={"key": str(i)}) for i in range(5)]
        results = reg._dispatch_concurrent_batch(calls)
        assert len(results) == 5
        assert all(r.error is None for r in results)

    def test_concurrent_batch_honors_worker_cap(self):
        import threading
        import time

        from looplet.tools import BaseToolRegistry, ToolSpec
        from looplet.types import ToolCall

        active = 0
        peak = 0
        lock = threading.Lock()

        def read(**kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return kwargs

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="read",
                description="read",
                parameters={"key": "key"},
                execute=read,
                concurrent_safe=True,
            )
        )

        results = reg.dispatch_batch(
            [ToolCall(tool="read", args={"key": str(i)}) for i in range(4)],
            max_workers=1,
        )

        assert len(results) == 4
        assert peak == 1


# ── Think tool tests ──────────────────────────────────────────────


class TestThinkTool:
    def test_think_tool_registered(self):
        from looplet.tools import BaseToolRegistry, register_think_tool

        reg = BaseToolRegistry()
        register_think_tool(reg)
        assert "think" in reg.tool_names

    def test_think_tool_is_free(self):
        from looplet.tools import BaseToolRegistry, register_think_tool

        reg = BaseToolRegistry()
        register_think_tool(reg)
        spec = reg._tools["think"]
        assert spec.free is True

    def test_think_tool_is_concurrent_safe(self):
        from looplet.tools import BaseToolRegistry, register_think_tool

        reg = BaseToolRegistry()
        register_think_tool(reg)
        spec = reg._tools["think"]
        assert spec.concurrent_safe is True

    def test_think_tool_dispatch(self):
        from looplet.tools import BaseToolRegistry, register_think_tool
        from looplet.types import ToolCall

        reg = BaseToolRegistry()
        register_think_tool(reg)
        call = ToolCall(tool="think", args={"analysis": "weighing pros and cons"})
        result = reg.dispatch(call)
        assert result.error is None
        assert result.data == {"acknowledged": True, "analysis": "weighing pros and cons"}

    def test_think_docstring_no_security_terms(self):
        import inspect

        from looplet.tools import register_think_tool

        doc = inspect.getdoc(register_think_tool) or ""
        lower_doc = doc.lower()
        for term in ["brute force", "vpn", "lateral movement", "security"]:
            assert term not in lower_doc, f"Security term '{term}' found in think tool docstring"

    def test_think_description_no_security_terms(self):
        from looplet.tools import BaseToolRegistry, register_think_tool

        reg = BaseToolRegistry()
        register_think_tool(reg)
        desc = reg._tools["think"].description.lower()
        for term in ["brute force", "vpn", "lateral movement"]:
            assert term not in desc, f"Security term '{term}' in think tool description"


# ── Export check ──────────────────────────────────────────────────


class TestExports:
    def test_all_tools_exported_from_looplet(self):
        import looplet as oh

        for name in ["ToolSpec", "BaseToolRegistry"]:
            assert hasattr(oh, name), f"{name} not exported from looplet"

    def test_register_think_tool_importable(self):
        from looplet.tools import register_think_tool

        assert register_think_tool is not None
