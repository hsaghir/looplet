"""Tests for harness snapshot provenance helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

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
        "memory_sources": ["MemorySource"],
        "llm_backend": "Backend",
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
