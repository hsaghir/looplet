"""Tests for the Composable Harness Workspace (CHW) round-trip.

Verifies:

* fresh-empty-directory write succeeds; non-empty fails without overwrite
* round-trip of a hand-built ``AgentPreset`` preserves config, system
  prompt, tools (with parameters + execute behaviour), and a hook with
  an opt-in ``to_config()`` method
* ``StaticMemorySource`` instances round-trip via ``memory/*.md``
* warnings are recorded for non-round-trippable config callables
  (``strict=False``) and raised under ``strict=True``
* layout discovery: the workspace.json metadata file is required
* ``Workspace.to_preset()`` materialises a runnable preset that the
  composable loop can execute end-to-end with a scripted MockLLM
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    Workspace,
    WorkspaceLayout,
    WorkspaceSerializationError,
    composable_loop,
    preset_to_workspace,
    workspace_to_preset,
)
from looplet.memory import StaticMemorySource
from looplet.presets import AgentPreset
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

# ── fixtures ────────────────────────────────────────────────────


def lookup_execute(*, key: str) -> dict:
    """Top-level execute so it round-trips through inspect.getsource."""
    return {"key": key, "value": {"x": 1, "y": 2}.get(key, "MISSING")}


def done_execute(*, answer: str) -> dict:
    return {"answer": answer}


class DemoCounter:
    """Hook with opt-in ``to_config()`` for round-trip kwargs."""

    def __init__(self, *, threshold: int = 3) -> None:
        self.threshold = threshold
        self.seen = 0

    def to_config(self) -> dict:
        return {"threshold": self.threshold}

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):  # noqa: D401
        self.seen += 1
        return None


def _build_demo_preset() -> AgentPreset:
    config = LoopConfig(
        max_steps=8,
        max_tokens=512,
        temperature=0.1,
        system_prompt="lookup agent",
        memory_sources=[StaticMemorySource(text="prefer x over y when both apply")],
    )
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup",
            description="Return the value for key.",
            parameters={"key": "str"},
            execute=lookup_execute,
        )
    )
    registry.register(
        ToolSpec(
            name="done",
            description="Submit final answer.",
            parameters={"answer": "str"},
            execute=done_execute,
        )
    )
    return AgentPreset(
        config=config,
        hooks=[DemoCounter(threshold=5)],
        tools=registry,
        state=DefaultState(max_steps=8),
    )


# ── basic IO ────────────────────────────────────────────────────


def test_write_to_empty_directory(tmp_path: Path) -> None:
    workspace = preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    assert (workspace.path / WorkspaceLayout.WORKSPACE_JSON).is_file()
    assert (workspace.path / WorkspaceLayout.SYSTEM_PROMPT_MD).read_text() == "lookup agent"
    assert (workspace.path / WorkspaceLayout.CONFIG_YAML).is_file()
    assert (workspace.path / WorkspaceLayout.TOOLS_DIR / "lookup" / "tool.yaml").is_file()
    assert (workspace.path / WorkspaceLayout.TOOLS_DIR / "lookup" / "execute.py").is_file()
    assert (workspace.path / WorkspaceLayout.HOOKS_DIR / "00_DemoCounter" / "hook.py").is_file()
    assert (workspace.path / WorkspaceLayout.MEMORY_DIR / "00_static.md").is_file()


def test_non_empty_directory_requires_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "ws"
    out.mkdir()
    (out / "stale").write_text("hi")
    with pytest.raises(FileExistsError):
        preset_to_workspace(_build_demo_preset(), out)
    # overwrite=True succeeds and wipes managed subdirs.
    workspace = preset_to_workspace(_build_demo_preset(), out, overwrite=True)
    assert (workspace.path / WorkspaceLayout.WORKSPACE_JSON).is_file()


def test_workspace_metadata_round_trips(tmp_path: Path) -> None:
    preset_to_workspace(
        _build_demo_preset(),
        tmp_path / "ws",
        name="demo",
        description="just a test",
    )
    loaded = Workspace.from_directory(tmp_path / "ws")
    assert loaded.name == "demo"
    assert loaded.description == "just a test"
    assert loaded.schema_version == 1


def test_missing_metadata_raises(tmp_path: Path) -> None:
    out = tmp_path / "ws"
    out.mkdir()
    (out / "config.yaml").write_text("max_steps: 5\n")
    with pytest.raises(FileNotFoundError):
        Workspace.from_directory(out)
    with pytest.raises(FileNotFoundError):
        workspace_to_preset(out)


# ── round-trip preset structure ─────────────────────────────────


def test_round_trip_preserves_config_subset(tmp_path: Path) -> None:
    preset = _build_demo_preset()
    preset_to_workspace(preset, tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    assert loaded.config.max_steps == preset.config.max_steps
    assert loaded.config.max_tokens == preset.config.max_tokens
    assert loaded.config.temperature == pytest.approx(preset.config.temperature)
    assert loaded.config.system_prompt == "lookup agent"
    assert loaded.config.done_tool == preset.config.done_tool


def test_round_trip_preserves_tools(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    names = {spec.name for spec in loaded.tools._tools.values()}  # type: ignore[attr-defined]
    assert names == {"lookup", "done"}
    lookup_spec = loaded.tools._tools["lookup"]  # type: ignore[attr-defined]
    assert lookup_spec.execute(key="x") == {"key": "x", "value": 1}
    assert lookup_spec.execute(key="y") == {"key": "y", "value": 2}
    assert lookup_spec.parameters == {"key": "str"}


def test_round_trip_preserves_hook_with_to_config(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    assert len(loaded.hooks) == 1
    hook = loaded.hooks[0]
    assert type(hook).__name__ == "DemoCounter"
    assert hook.threshold == 5


def test_round_trip_preserves_static_memory(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")
    sources = list(loaded.config.memory_sources)
    assert len(sources) == 1
    assert isinstance(sources[0], StaticMemorySource)
    assert sources[0].text == "prefer x over y when both apply"


# ── runnable round-trip ─────────────────────────────────────────


def test_round_tripped_preset_runs_end_to_end(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    loaded = workspace_to_preset(tmp_path / "ws")

    llm = MockLLMBackend(
        responses=[
            '{"tool":"lookup","args":{"key":"x"},"reasoning":"check"}',
            '{"tool":"done","args":{"answer":"x=1"},"reasoning":""}',
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=loaded.tools,
            state=loaded.state,
            config=loaded.config,
            hooks=loaded.hooks,
            task={"goal": "lookup x"},
        )
    )
    assert [s.tool_call.tool for s in steps] == ["lookup", "done"]
    # The DemoCounter's post_dispatch ran for the lookup step.
    assert loaded.hooks[0].seen >= 1


# ── warnings + strict ──────────────────────────────────────────


def test_non_serializable_config_field_warns_in_loose_mode(tmp_path: Path) -> None:
    preset = _build_demo_preset()
    # Set a callable on a known non-serializable field.
    preset.config.build_briefing = lambda **_: "x"
    workspace = preset_to_workspace(preset, tmp_path / "ws")
    assert any("build_briefing" in w for w in workspace.serialization_warnings)


def test_non_serializable_config_field_raises_in_strict_mode(tmp_path: Path) -> None:
    preset = _build_demo_preset()
    preset.config.build_briefing = lambda **_: "x"
    with pytest.raises(WorkspaceSerializationError):
        preset_to_workspace(preset, tmp_path / "ws", strict=True)


def test_warnings_are_empty_for_clean_preset(tmp_path: Path) -> None:
    workspace = preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    # The demo preset uses only round-trippable fields; no warnings expected.
    assert workspace.serialization_warnings == []


# ── layout sanity ──────────────────────────────────────────────


def test_workspace_json_is_stable(tmp_path: Path) -> None:
    preset_to_workspace(_build_demo_preset(), tmp_path / "ws", name="demo")
    payload = json.loads((tmp_path / "ws" / "workspace.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["name"] == "demo"
    assert "metadata" in payload


def test_layout_constants_match_written_paths(tmp_path: Path) -> None:
    workspace = preset_to_workspace(_build_demo_preset(), tmp_path / "ws")
    expected = {
        WorkspaceLayout.WORKSPACE_JSON,
        WorkspaceLayout.SYSTEM_PROMPT_MD,
        WorkspaceLayout.CONFIG_YAML,
    }
    for relative in expected:
        assert (workspace.path / relative).exists(), f"missing {relative}"


# ── coder-preset round-trip (real-world dogfood) ───────────────


def test_coder_preset_round_trips_with_strict_load(tmp_path: Path) -> None:
    """Regression: a real preset with built-in hooks (ThresholdCompactHook)
    that need constructor args must round-trip under strict=True after
    to_config() was added to the relevant hooks. Previously the loader
    silently dropped any hook whose config.yaml lacked kwargs the
    constructor needed."""
    from looplet import coding_agent_preset

    preset = coding_agent_preset(workspace=str(tmp_path / "ws"), max_steps=5)
    out = tmp_path / "coder.workspace"
    preset_to_workspace(preset, out, name="coder")

    # strict=True must succeed end-to-end — every hook must reload.
    reloaded = workspace_to_preset(out, strict=True)
    original_hook_names = [type(h).__name__ for h in preset.hooks]
    reloaded_hook_names = [type(h).__name__ for h in reloaded.hooks]
    assert reloaded_hook_names == original_hook_names, (
        f"hook list changed on round-trip: {original_hook_names} -> {reloaded_hook_names}"
    )


def test_strict_load_raises_on_unconstructable_hook(tmp_path: Path) -> None:
    """Regression: hooks whose config.yaml lacks required constructor
    kwargs must raise WorkspaceSerializationError under strict=True
    instead of silently dropping. Loose mode still drops + warns."""
    out = tmp_path / "broken.workspace"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    hook_dir = out / "hooks" / "00_NeedsArgs"
    hook_dir.mkdir(parents=True)
    (hook_dir / "hook.py").write_text(
        "class NeedsArgs:\n    def __init__(self, *, required_arg):\n        self.x = required_arg\n"
    )
    (hook_dir / "config.yaml").write_text("class_name: NeedsArgs\nkwargs: {}\n")

    # Loose mode: hook silently dropped (logged warning).
    loose = workspace_to_preset(out)
    assert loose.hooks == []

    # Strict mode: raises with actionable message naming to_config().
    with pytest.raises(WorkspaceSerializationError, match="to_config"):
        workspace_to_preset(out, strict=True)


# ── Shared resources + @ref + setup.py ─────────────────────────


def test_resources_dir_builds_shared_objects(tmp_path: Path) -> None:
    """resources/<name>.py with `def build()` populates the resource
    registry that ``@<name>`` references resolve against."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "resources").mkdir()
    (out / "resources" / "shared_cache.py").write_text(
        "def build():\n    return {'cache_id': 'singleton', 'items': []}\n"
    )

    # Hook that takes a `cache` kwarg via @ref.
    hook_dir = out / "hooks" / "00_TwoConsumers"
    hook_dir.mkdir(parents=True)
    (hook_dir / "hook.py").write_text(
        "class TwoConsumers:\n    def __init__(self, *, cache):\n        self.cache = cache\n"
    )
    (hook_dir / "config.yaml").write_text(
        'class_name: TwoConsumers\nkwargs:\n  cache: "@shared_cache"\n'
    )

    preset = workspace_to_preset(out, strict=True)
    assert len(preset.hooks) == 1
    assert preset.hooks[0].cache == {"cache_id": "singleton", "items": []}


def test_two_hooks_share_same_resource_object(tmp_path: Path) -> None:
    """Two hooks referencing the same @<name> get the SAME Python object,
    not two independent copies. This is the FileCacheHook + StaleFileHook
    pattern: shared mutable state must survive workspace round-trip."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "resources").mkdir()
    (out / "resources" / "cache.py").write_text("def build():\n    return {'shared': True}\n")

    for idx, name in enumerate(("Reader", "Writer")):
        d = out / "hooks" / f"{idx:02d}_{name}"
        d.mkdir(parents=True)
        (d / "hook.py").write_text(
            f"class {name}:\n    def __init__(self, *, cache):\n        self.cache = cache\n"
        )
        (d / "config.yaml").write_text(f'class_name: {name}\nkwargs:\n  cache: "@cache"\n')

    preset = workspace_to_preset(out, strict=True)
    assert preset.hooks[0].cache is preset.hooks[1].cache, (
        "Both hooks must reference the SAME object (shared state), not separate copies"
    )


def test_unresolved_ref_raises_in_strict(tmp_path: Path) -> None:
    """``"@missing"`` with no matching resource raises so the user
    sees the typo immediately."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    hook_dir = out / "hooks" / "00_NeedsRef"
    hook_dir.mkdir(parents=True)
    (hook_dir / "hook.py").write_text(
        "class NeedsRef:\n    def __init__(self, *, dep):\n        self.dep = dep\n"
    )
    (hook_dir / "config.yaml").write_text('class_name: NeedsRef\nkwargs:\n  dep: "@nonexistent"\n')

    with pytest.raises(WorkspaceSerializationError, match="unresolved resource reference"):
        workspace_to_preset(out, strict=True)


def test_setup_py_escape_hatch_runs_after_load(tmp_path: Path) -> None:
    """setup.py's `setup(preset, resources)` runs after the declarative
    load and can attach callable / opaque LoopConfig fields. Used for
    the rare case where a workspace genuinely needs load-time Python."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "config.yaml").write_text("max_steps: 7\n")
    (out / "setup.py").write_text(
        "def setup(preset, resources):\n    preset.config.max_steps = 99\n    return preset\n"
    )

    preset = workspace_to_preset(out, strict=True)
    assert preset.config.max_steps == 99, "setup.py mutation lost"


def test_setup_py_invalid_signature_raises(tmp_path: Path) -> None:
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "setup.py").write_text("# no setup function defined\n")

    with pytest.raises(WorkspaceSerializationError, match="must define"):
        workspace_to_preset(out, strict=True)


# ── examples/hello.workspace end-to-end (proof-of-concept v2 cartridge) ───


def test_hello_workspace_loads_and_runs_end_to_end() -> None:
    """examples/hello.workspace is the proof-of-concept v2 cartridge:
    fully declarative layout with shared resources + setup.py wiring.
    Loads, runs scripted, and the shared GreetingLog round-trips
    state between the greet tool and the PolitenessGate hook."""
    import json as _json
    from pathlib import Path as _P

    from looplet import composable_loop
    from looplet.testing import MockLLMBackend

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "hello.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)

    assert preset.config.max_steps == 5
    assert "polite assistant" in preset.config.system_prompt.lower()
    assert {type(h).__name__ for h in preset.hooks} == {"PolitenessGate"}
    assert sorted(preset.tools._tools.keys()) == ["done", "greet"]

    hook = preset.hooks[0]
    assert hasattr(hook, "log") and hasattr(hook.log, "entries")
    assert hook.log.entries == []

    llm = MockLLMBackend(
        responses=[
            _json.dumps({"thought": "polite", "tool": "greet", "args": {"name": "Alice"}}),
            _json.dumps({"thought": "polite", "tool": "greet", "args": {"name": "Bob"}}),
            _json.dumps({"thought": "finish", "tool": "done", "args": {"answer": "Greeted both."}}),
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"q": "greet alice and bob"},
        )
    )

    assert len(steps) == 3
    assert [s.tool_call.tool for s in steps] == ["greet", "greet", "done"]
    # Shared log captured both greetings — proves @ref + setup.py wired
    # the SAME GreetingLog instance into the tool and the hook.
    assert hook.log.names() == ["Alice", "Bob"]


# ── examples/coder.workspace end-to-end (real-world v2 cartridge) ──


def test_coder_workspace_loads_with_shared_filecache() -> None:
    """examples/coder.workspace migrates the v1 coder cartridge to the
    v2 layout. Validates that:
      * 5 built-in + custom hooks load with strict=True
      * 9 tools (bash/list_dir/read/write/edit/glob/grep/think/done) load
      * FileCacheHook and StaleFileHook share the SAME FileCache instance
        via @file_cache (proves the shared-resource registry under load)
      * setup.py wires WORKSPACE_CONFIG + FILE_CACHE module globals into
        every tool that needs them
    """
    import json as _json
    from pathlib import Path as _P

    from looplet import composable_loop
    from looplet.testing import MockLLMBackend

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "coder.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)

    hook_names = [type(h).__name__ for h in preset.hooks]
    assert hook_names == [
        "TestGuardHook",
        "FileCacheHook",
        "StaleFileHook",
        "StagnationHook",
        "ThresholdCompactHook",
    ]
    assert sorted(preset.tools._tools.keys()) == [
        "bash",
        "done",
        "edit_file",
        "glob",
        "grep",
        "list_dir",
        "read_file",
        "think",
        "write_file",
    ]

    # The shared-state proof: FileCacheHook and StaleFileHook reference
    # the SAME cache object via @file_cache, NOT two independent copies.
    fc_hook = next(h for h in preset.hooks if type(h).__name__ == "FileCacheHook")
    sf_hook = next(h for h in preset.hooks if type(h).__name__ == "StaleFileHook")
    assert fc_hook._cache is sf_hook._cache

    # End-to-end smoke: think → done with the real loop.
    llm = MockLLMBackend(
        responses=[
            _json.dumps({"thought": "plan", "tool": "think", "args": {"thought": "smoke"}}),
            _json.dumps({"thought": "finish", "tool": "done", "args": {"summary": "ok"}}),
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"q": "smoke"},
        )
    )
    assert [s.tool_call.tool for s in steps] == ["think", "done"]


def test_threat_intel_workspace_loads() -> None:
    """examples/threat_intel.workspace migrates the threat-intel
    cartridge. Loads with strict=True; tools re-import from the
    original module so their typing/closure environment stays intact."""
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "threat_intel.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)
    assert sorted(preset.tools._tools.keys()) == [
        "assess_risk",
        "done",
        "extract_iocs",
        "fetch_feed",
        "map_mitre",
        "search_cve",
        "think",
    ]
    assert [type(h).__name__ for h in preset.hooks] == [
        "StagnationHook",
        "PerToolLimitHook",
    ]


def test_dep_doctor_workspace_loads() -> None:
    """examples/dep_doctor.workspace migration."""
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "dep_doctor.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)
    # detect_dep_files / check_license_compat dirs map via tool.yaml `name`
    # to detect_files / check_license respectively (the original @tool name).
    assert sorted(preset.tools._tools.keys()) == [
        "check_license",
        "check_package",
        "detect_files",
        "done",
        "find_alternatives",
        "parse_deps",
        "think",
    ]
    assert [type(h).__name__ for h in preset.hooks] == [
        "StagnationHook",
        "PerToolLimitHook",
    ]


def test_git_detective_workspace_loads() -> None:
    """examples/git_detective.workspace migration. Uses lazy
    closure-registry via ``make_tools(REPO_CONFIG.path)`` with
    setup.py injecting the shared repo_config resource."""
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "git_detective.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)
    assert sorted(preset.tools._tools.keys()) == [
        "commit_patterns",
        "contributor_stats",
        "coupled_files",
        "directory_structure",
        "done",
        "file_age_analysis",
        "file_hotspots",
        "recent_activity",
        "repo_overview",
        "think",
    ]
    assert [type(h).__name__ for h in preset.hooks] == [
        "StagnationHook",
        "PerToolLimitHook",
    ]


# ── runtime= kwarg + ${runtime.x} substitution ────────────────


def test_runtime_substitution_in_config_yaml(tmp_path):
    """${runtime.<key>} placeholders in config.yaml get replaced with
    the host-supplied runtime value before LoopConfig is constructed."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "config.yaml").write_text("max_steps: 5\ncheckpoint_dir: ${runtime.cp_dir}\n")

    preset = workspace_to_preset(out, strict=True, runtime={"cp_dir": "/tmp/my-cp"})
    assert preset.config.max_steps == 5
    assert preset.config.checkpoint_dir == "/tmp/my-cp"


def test_runtime_substitution_unknown_key_raises(tmp_path):
    """Typos in ${runtime.<key>} fail loudly at load time, not silently."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "config.yaml").write_text("max_steps: ${runtime.nonexistent}\n")

    with pytest.raises(WorkspaceSerializationError, match="unresolved"):
        workspace_to_preset(out, strict=True, runtime={"actual_key": 5})


def test_resource_builder_receives_runtime(tmp_path):
    """resources/<name>.py builders that declare def build(runtime)
    get the host-supplied runtime dict; legacy zero-arg build()
    keeps working."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "resources").mkdir()
    (out / "resources" / "tagged.py").write_text(
        "def build(runtime=None):\n"
        "    runtime = runtime or {}\n"
        "    return {'tag': runtime.get('tag', 'default')}\n"
    )

    hook_dir = out / "hooks" / "00_TagReader"
    hook_dir.mkdir(parents=True)
    (hook_dir / "hook.py").write_text(
        "class TagReader:\n    def __init__(self, *, source):\n        self.source = source\n"
    )
    (hook_dir / "config.yaml").write_text('class_name: TagReader\nkwargs:\n  source: "@tagged"\n')

    preset = workspace_to_preset(out, strict=True, runtime={"tag": "from-runtime"})
    assert preset.hooks[0].source == {"tag": "from-runtime"}


def test_setup_py_receives_runtime_kwarg(tmp_path):
    """setup.py's setup() gets runtime= when its signature accepts it."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    (out / "setup.py").write_text(
        "def setup(preset, resources, runtime=None):\n"
        "    runtime = runtime or {}\n"
        "    preset.config.system_prompt = runtime.get('prompt', 'default')\n"
        "    return preset\n"
    )

    preset = workspace_to_preset(out, strict=True, runtime={"prompt": "from-runtime"})
    assert preset.config.system_prompt == "from-runtime"


def test_coder_workspace_runtime_kwarg_routes_files(tmp_path):
    """End-to-end regression for the runtime= kwarg in coder.workspace:
    write_file via composable_loop must land in the runtime-supplied
    workspace, NOT in the test cwd. The previous code wrote to cwd
    because there was no way to point a workspace at a runtime path."""
    import json as _json
    from pathlib import Path as _P

    from looplet import composable_loop
    from looplet.testing import MockLLMBackend

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "coder.workspace"
    target = tmp_path / "target-repo"
    target.mkdir()

    preset = workspace_to_preset(workspace_dir, strict=True, runtime={"workspace": str(target)})
    llm = MockLLMBackend(
        responses=[
            _json.dumps(
                {
                    "thought": "write",
                    "tool": "write_file",
                    "args": {"file_path": "hello.py", "content": "print('hi')\n"},
                }
            ),
            _json.dumps({"thought": "finish", "tool": "done", "args": {"summary": "ok"}}),
        ]
    )
    list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"q": "write hello.py"},
        )
    )
    assert (target / "hello.py").read_text().strip() == "print('hi')"
