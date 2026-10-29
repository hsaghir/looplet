"""Tests for harness snapshot provenance helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from looplet import serialize_harness
from looplet.provenance import TrajectoryRecorder


def test_serialize_harness_defaults_to_schema_and_empty_extra():
    assert serialize_harness() == {"schema_version": 2, "extra": {}}


def test_serialize_harness_captures_config_and_component_names():
    class FakeConfig:
        system_prompt = "You are careful."
        max_steps = 5
        max_tokens = 1000
        temperature = 0.1
        use_native_tools = True
        concurrent_dispatch = False
        done_tool = "finish"

    class GuardHook:
        pass

    class MemorySource:
        pass

    class Backend:
        pass

    @dataclass
    class FakeSpec:
        name: str
        description: str

    class FakeTools:
        _specs = {
            "search": FakeSpec(name="search", description="Search docs."),
            "read": FakeSpec(name="read", description="Read a file."),
        }

    snap = serialize_harness(
        config=FakeConfig(),
        hooks=[GuardHook()],
        tools=FakeTools(),
        memory_sources=[MemorySource()],
        llm=Backend(),
        extra={"variant": "a"},
    )

    assert snap == {
        "schema_version": 2,
        "extra": {"variant": "a"},
        "system_prompt": "You are careful.",
        "system_prompt_sha256": sha256(b"You are careful.").hexdigest(),
        "max_steps": 5,
        "max_tokens": 1000,
        "temperature": 0.1,
        "use_native_tools": True,
        "concurrent_dispatch": False,
        "done_tool": "finish",
        "tools": [
            {"name": "search", "description": "Search docs."},
            {"name": "read", "description": "Read a file."},
        ],
        "hooks": ["GuardHook"],
        "hook_types": [f"{GuardHook.__module__}.{GuardHook.__qualname__}"],
        "memory_sources": ["MemorySource"],
        "memory_source_types": [f"{MemorySource.__module__}.{MemorySource.__qualname__}"],
        "llm_backend": "Backend",
        "llm_backend_type": f"{Backend.__module__}.{Backend.__qualname__}",
    }
    json.dumps(snap)


def test_serialize_harness_truncates_long_system_prompt():
    class FakeConfig:
        system_prompt = "x" * 5000

    snap = serialize_harness(config=FakeConfig())

    assert len(snap["system_prompt"]) < 5000
    assert "truncated" in snap["system_prompt"]


def test_trajectory_recorder_stores_harness_snapshot_on_loop_end():
    snap = {"schema_version": 2, "extra": {"variant": "baseline"}}
    recorder = TrajectoryRecorder(harness_snapshot=snap)

    recorder.on_loop_end(None, None, None, None)

    assert recorder.trajectory.metadata["harness_snapshot"] == snap
    assert recorder.trajectory.metadata["harness_snapshot"] is not snap


def test_trajectory_recorder_without_harness_snapshot_leaves_metadata_empty():
    recorder = TrajectoryRecorder()

    recorder.on_loop_end(None, None, None, None)

    assert recorder.trajectory.metadata == {}


def test_snapshot_distinguishes_effective_limits_without_exporting_secrets():
    from looplet import LoopConfig, resolve_harness

    first = LoopConfig(max_model_calls=1, context_window=8000)
    second = LoopConfig(max_model_calls=40, context_window=64000)
    second.generate_kwargs = {"api_key": "private-test-key", "extra_headers": {"token": "secret"}}
    first_snapshot = serialize_harness(config=resolve_harness(first).config)
    second_snapshot = serialize_harness(config=resolve_harness(second).config)

    assert first_snapshot != second_snapshot
    assert first_snapshot["max_model_calls"] == 1
    assert second_snapshot["max_model_calls"] == 40
    assert first_snapshot["context_window"] == 8000
    assert second_snapshot["context_window"] == 64000
    assert second_snapshot["harness_profile"] == "execution-v1"
    assert "private-test-key" not in json.dumps(second_snapshot)
    assert "secret" not in json.dumps(second_snapshot)
    assert "generate_kwargs" not in second_snapshot


def test_snapshot_hash_distinguishes_truncated_prompt_tails():
    from looplet import LoopConfig

    first = serialize_harness(config=LoopConfig(system_prompt="x" * 5000 + "first"))
    second = serialize_harness(config=LoopConfig(system_prompt="x" * 5000 + "other"))

    assert first["system_prompt"] == second["system_prompt"]
    assert first["system_prompt_sha256"] != second["system_prompt_sha256"]


def test_snapshot_detaches_tool_schemas_and_identifies_components():
    from looplet import LoopConfig, tools_from

    def source(number: int):
        return number

    tools = tools_from([source])
    config = LoopConfig(disabled_defaults=["branching"])
    snapshot = serialize_harness(config=config, tools=tools)

    assert snapshot["tools"][0]["parameters"]["properties"]["number"]["type"] == "integer"
    assert snapshot["tools"][0]["implementation"].endswith(".<locals>.source")
    config.disabled_defaults.append("checkpointing")
    assert snapshot["disabled_defaults"] == ["branching"]
