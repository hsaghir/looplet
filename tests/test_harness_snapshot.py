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


def test_snapshot_distinguishes_declared_hook_configuration():
    from looplet import LoopConfig, StagnationHook, resolve_harness

    snapshots = []
    for threshold in (3, 12):
        resolved = resolve_harness(LoopConfig(), [StagnationHook(threshold=threshold)])
        snapshots.append(
            serialize_harness(
                config=resolved.config,
                hooks=resolved.hooks,
                extra={"resolved_harness": resolved.describe()},
            )
        )
    assert snapshots[0] != snapshots[1]
    assert snapshots[0]["hook_configurations"][0]["threshold"] == 3
    assert snapshots[1]["hook_configurations"][0]["threshold"] == 12


def test_snapshot_describes_callable_policies_without_executing_them():
    from looplet import BudgetWarningHook, StagnationHook

    def policy(*args):
        raise AssertionError("snapshot must not execute a policy")

    hooks = [
        StagnationHook(fingerprint=policy, progress=policy, nudge=policy),
        BudgetWarningHook(message=policy),
    ]
    snapshot = serialize_harness(hooks=hooks)
    reference = {"callable": f"{policy.__module__}.{policy.__qualname__}"}
    assert snapshot["hook_configurations"][0]["fingerprint"] == reference
    assert snapshot["hook_configurations"][0]["progress"] == reference
    assert snapshot["hook_configurations"][0]["nudge"] == reference
    assert snapshot["hook_configurations"][1]["message"] == reference
    json.dumps(snapshot)


def test_snapshot_captures_only_explicit_provenance_configuration():
    from types import SimpleNamespace

    from looplet import LoopConfig

    class Component:
        def __init__(self):
            self.api_key = "private-test-key"
            self.limit = [3, 5]
            self.calls = 99

        def to_config(self):
            raise AssertionError("cartridge export is not provenance")

        def provenance_config(self):
            return {"limits": self.limit}

    component = Component()
    opaque = SimpleNamespace(api_key="private-other-key")
    snapshot = serialize_harness(
        config=LoopConfig(compact_service=component),
        hooks=[component, opaque],
        extra={"agent_revision": "host-revision-42"},
    )
    assert snapshot["component_configurations"]["compact_service"] == {"limits": [3, 5]}
    assert snapshot["hook_configurations"] == [{"limits": [3, 5]}, None]
    assert snapshot["extra"]["agent_revision"] == "host-revision-42"
    component.limit.append(8)
    assert snapshot["hook_configurations"][0] == {"limits": [3, 5]}
    serialized = json.dumps(snapshot)
    assert "private-test-key" not in serialized
    assert "private-other-key" not in serialized
    assert "calls" not in snapshot["component_configurations"]["compact_service"]


def test_snapshot_budget_configuration_excludes_consumed_state():
    from looplet import BudgetWarningHook, PerToolLimitHook, ToolCall, ToolCallBudgetHook

    budget = ToolCallBudgetHook(max_calls=7)
    per_tool = PerToolLimitHook(limits={"lookup": 2})
    warnings = BudgetWarningHook(thresholds=(0.8, 0.4))
    hooks = [budget, per_tool, warnings]
    before = serialize_harness(hooks=hooks)
    call = ToolCall(tool="lookup", args={})
    assert budget.pre_dispatch(None, None, call, 1) is None
    assert per_tool.pre_dispatch(None, None, call, 1) is None
    assert budget.calls == 1
    assert per_tool.counts == {"lookup": 1}
    assert serialize_harness(hooks=hooks) == before
    assert before["hook_configurations"][0] == {"max_calls": 7}
    assert before["hook_configurations"][1]["limits"] == {"lookup": 2}
    assert before["hook_configurations"][2]["thresholds"] == [0.8, 0.4]
