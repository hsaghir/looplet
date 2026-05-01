"""Tests for the Workspace round-trip.

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


def _done_execute(*, s: str = "") -> dict:
    """Top-level done-tool callable so workspace round-trip can re-import it."""
    return {"status": "completed", "summary": s}


def _trivial_evaluator(ctx) -> "EvalResult":  # noqa: F821 - imported in test
    """Top-level evaluator so EvalHook auto-emit produces an importable resource."""
    from looplet import EvalResult  # noqa: PLC0415

    return EvalResult(name="trivial", passed=True)


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
    # Set a callable on a known non-serializable field. The writer now
    # auto-emits a ``@build_briefing`` ref + ``resources/build_briefing.py``
    # stub; for closure / lambda values the stub is the None-fallback
    # and a warning is recorded so the user knows manual editing is
    # required for cross-process round-trip.
    preset.config.build_briefing = lambda **_: "x"
    workspace = preset_to_workspace(preset, tmp_path / "ws")
    assert any("build_briefing" in w for w in workspace.serialization_warnings)
    assert (tmp_path / "ws" / "resources" / "build_briefing.py").is_file()


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


# ── examples/hello.workspace end-to-end (proof-of-concept workspace) ───


def test_hello_workspace_loads_and_runs_end_to_end() -> None:
    """examples/hello.workspace is the proof-of-concept workspace:
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


# ── examples/coder.workspace end-to-end (real-world workspace) ──


def test_coder_workspace_loads_with_shared_filecache(tmp_path) -> None:
    """examples/coder.workspace migrates the v1 coder bundle to the
    v2 layout. Validates that:
      * Declarative + setup.py-injected hooks load (8 total — TestGuard,
        FileCache, StaleFile, Stagnation, ThresholdCompact, PerToolLimit
        from YAML; LinterHook + EvalHook appended by setup.py to match
        looplet.examples coder reference feature-for-feature)
      * 9 tools (bash/list_dir/read/write/edit/glob/grep/think/done) load
      * FileCacheHook and StaleFileHook share the SAME FileCache instance
        via @file_cache (proves the shared-resource registry under load)
      * setup.py wires WORKSPACE_CONFIG + FILE_CACHE module globals into
        every tool that needs them
      * setup.py also attaches compact_service + project-context memory
    """
    import json as _json
    from pathlib import Path as _P

    from looplet import composable_loop
    from looplet.testing import MockLLMBackend

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "coder.workspace"
    # Use a tmp_path workspace so EvalHook's pytest collector doesn't
    # recurse into the looplet test suite when it runs at on_loop_end.
    target_repo = tmp_path / "target"
    target_repo.mkdir()
    preset = workspace_to_preset(
        workspace_dir,
        strict=True,
        runtime={"workspace": str(target_repo)},
    )

    hook_names = [type(h).__name__ for h in preset.hooks]
    assert hook_names == [
        "TestGuardHook",
        "FileCacheHook",
        "StaleFileHook",
        "StagnationHook",
        "ThresholdCompactHook",
        "PerToolLimitHook",
        "LinterHook",
        "EvalHook",
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

    # setup.py attaches compact_service.
    assert preset.config.compact_service is not None

    # setup.py appends the live-state CallableMemorySource for project ctx.
    assert preset.config.memory_sources, "setup.py should attach memory sources"

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
    bundle. Loads with strict=True; tools re-import from the
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
    # tool.yaml ``name`` matches the dir name so the registered tool
    # name is the same as the workspace directory.
    assert sorted(preset.tools._tools.keys()) == [
        "check_license_compat",
        "check_package",
        "detect_dep_files",
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


# ── Bidirectional + v1↔v2 parity ──────────────────────────────


def test_coder_workspace_bidirectional_round_trip(tmp_path) -> None:
    """Load coder.workspace → snapshot to a fresh dir → reload. The
    declarative hooks (with to_config()) and tools survive in-process
    round-trip; setup.py-appended hooks (EvalHook) drop on reload
    because their callable evaluators don't round-trip."""
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "coder.workspace"
    target = tmp_path / "target"
    target.mkdir()
    snap_dir = tmp_path / "snapshot"

    preset = workspace_to_preset(workspace_dir, strict=True, runtime={"workspace": str(target)})
    preset_to_workspace(preset, snap_dir, name="coder-snap")

    # The coder workspace ships co-located ``lib_*.py`` helper modules
    # that hosts and tools subclass / call into. Round-tripping a
    # preset doesn't auto-copy these — the snapshot is otherwise
    # self-contained but its hook.py / execute.py shims still
    # ``from lib_tools import ...``. Copy them alongside so the
    # reload finds the same helpers.
    import shutil as _shutil

    for _lib in ("coder_lib_tools.py", "coder_lib_hooks.py", "coder_lib_wiring.py"):
        _src = workspace_dir / _lib
        if _src.is_file():
            _shutil.copy(_src, snap_dir / _lib)

    # The auto-emitted resources/file_cache.py builder reads
    # runtime['workspace']; pass it on reload.
    reloaded = workspace_to_preset(snap_dir, runtime={"workspace": str(target)})

    # All 8 declarative hooks survive the round-trip — including
    # EvalHook now that its evaluators + collectors live in
    # resources/eval_evaluators.py and resources/eval_collectors.py
    # (referenced via @ref instead of injected via setup.py).
    reloaded_names = [type(h).__name__ for h in reloaded.hooks]
    assert reloaded_names == [
        "TestGuardHook",
        "FileCacheHook",
        "StaleFileHook",
        "StagnationHook",
        "ThresholdCompactHook",
        "PerToolLimitHook",
        "LinterHook",
        "EvalHook",
    ]
    assert sorted(reloaded.tools._tools.keys()) == [
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


def test_threat_intel_workspace_attaches_compact_service(tmp_path) -> None:
    """setup.py wires LoopConfig.compact_service so the workspace
    matches the looplet.examples coder reference feature-for-feature."""
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "threat_intel.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)
    assert preset.config.compact_service is not None


def test_dep_doctor_workspace_attaches_compact_and_memory(tmp_path) -> None:
    """compact_service + memory/00_static.md both reach the preset."""
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "dep_doctor.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)
    assert preset.config.compact_service is not None
    sources = preset.config.memory_sources or []
    assert any("Audit Standards" in getattr(s, "text", "") for s in sources), (
        "memory/00_static.md should land in config.memory_sources"
    )


def test_git_detective_workspace_attaches_compact_and_memory(tmp_path) -> None:
    from pathlib import Path as _P

    workspace_dir = _P(__file__).resolve().parents[1] / "examples" / "git_detective.workspace"
    preset = workspace_to_preset(workspace_dir, strict=True)
    assert preset.config.compact_service is not None
    sources = preset.config.memory_sources or []
    assert any("Report Standards" in getattr(s, "text", "") for s in sources)


def test_runtime_substitution_in_hook_config(tmp_path) -> None:
    """${runtime.<key>} placeholders in hook config.yaml are substituted
    just like they are in the top-level config.yaml. Used by coder's
    LinterHook to receive the runtime workspace path declaratively."""
    out = tmp_path / "ws"
    out.mkdir()
    (out / "workspace.json").write_text(json.dumps({"name": "x", "schema_version": 1}))
    hook_dir = out / "hooks" / "00_PathReader"
    hook_dir.mkdir(parents=True)
    (hook_dir / "hook.py").write_text(
        "class PathReader:\n    def __init__(self, *, path):\n        self.path = path\n"
    )
    (hook_dir / "config.yaml").write_text(
        "class_name: PathReader\nkwargs:\n  path: ${runtime.workspace}\n"
    )

    preset = workspace_to_preset(out, strict=True, runtime={"workspace": "/tmp/some-path"})
    assert preset.hooks[0].path == "/tmp/some-path"


# ── Harness-shape stress tests (PR #31) ───────────────────────


def test_permission_engine_via_at_ref(tmp_path) -> None:
    """PermissionEngine + PermissionHook compose via @ref. Lets users
    declare permission policy in resources/perm_engine.py and reference
    it from hooks/00_PermissionHook/config.yaml."""
    import json as _json
    from pathlib import Path as _P

    ws = tmp_path
    (ws / "workspace.json").write_text(_json.dumps({"name": "x", "schema_version": 1}))
    (ws / "config.yaml").write_text("max_steps: 5\n")
    (ws / "tools/echo").mkdir(parents=True)
    (ws / "tools/echo/tool.yaml").write_text("name: echo\nparameters:\n  msg:\n    type: string\n")
    (ws / "tools/echo/execute.py").write_text("def execute(*, msg): return {'echoed': msg}\n")
    (ws / "tools/done").mkdir(parents=True)
    (ws / "tools/done/tool.yaml").write_text("name: done\nparameters:\n  s:\n    type: string\n")
    (ws / "tools/done/execute.py").write_text(
        "def execute(*, s): return {'status': 'completed', 's': s}\n"
    )
    (ws / "resources").mkdir()
    (ws / "resources/perm_engine.py").write_text(
        "from looplet import PermissionEngine, PermissionDecision\n"
        "def build(runtime=None):\n"
        "    eng = PermissionEngine(default=PermissionDecision.ALLOW)\n"
        "    eng.deny('echo', arg_matcher=lambda a: a.get('msg', '').startswith('SECRET'))\n"
        "    return eng\n"
    )
    (ws / "hooks/00_PermissionHook").mkdir(parents=True)
    (ws / "hooks/00_PermissionHook/hook.py").write_text(
        "from looplet import PermissionHook as _PH\n"
        "class PermissionHook(_PH):\n"
        "    def to_config(self): return {'engine': '@perm_engine'}\n"
    )
    (ws / "hooks/00_PermissionHook/config.yaml").write_text(
        'class_name: PermissionHook\nkwargs:\n  engine: "@perm_engine"\n'
    )

    from looplet import composable_loop
    from looplet.testing import MockLLMBackend

    preset = workspace_to_preset(ws, strict=True)
    llm = MockLLMBackend(
        responses=[
            _json.dumps({"thought": "ok", "tool": "echo", "args": {"msg": "hello"}}),
            _json.dumps({"thought": "block", "tool": "echo", "args": {"msg": "SECRET-leak"}}),
            _json.dumps({"thought": "f", "tool": "done", "args": {"s": "ok"}}),
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"q": "x"},
        )
    )
    assert steps[0].tool_result.error is None
    assert steps[1].tool_result.error and "permission" in steps[1].tool_result.error.lower()


def test_eval_hook_declarative_via_at_ref(tmp_path) -> None:
    """EvalHook with declarative evaluators + collectors via @ref.
    Confirms callable-graph hooks are NOT a permanent setup.py
    requirement — when callables are exposed as resource builders
    that return them, the @ref registry resolves them just like any
    other shared resource."""
    import json as _json

    ws = tmp_path
    (ws / "workspace.json").write_text(_json.dumps({"name": "x", "schema_version": 1}))
    (ws / "config.yaml").write_text("max_steps: 3\n")
    (ws / "tools/done").mkdir(parents=True)
    (ws / "tools/done/tool.yaml").write_text("name: done\nparameters:\n  s:\n    type: string\n")
    (ws / "tools/done/execute.py").write_text(
        "def execute(*, s): return {'status': 'completed', 's': s}\n"
    )
    (ws / "resources").mkdir()
    (ws / "resources/evaluators.py").write_text(
        "def eval_completed(ctx): return ctx.completed\n"
        "def build(runtime=None): return [eval_completed]\n"
    )
    (ws / "hooks/00_EvalHook").mkdir(parents=True)
    (ws / "hooks/00_EvalHook/hook.py").write_text(
        "from looplet import EvalHook as _EH\n"
        "class EvalHook(_EH):\n"
        "    def to_config(self): return {'evaluators': '@evaluators'}\n"
    )
    (ws / "hooks/00_EvalHook/config.yaml").write_text(
        'class_name: EvalHook\nkwargs:\n  evaluators: "@evaluators"\n'
    )

    preset = workspace_to_preset(ws, strict=True)
    eh = preset.hooks[0]
    evals = getattr(eh, "evaluators", None) or getattr(eh, "_evaluators", None)
    assert evals and len(evals) == 1


def test_streaming_hook_declarative_via_at_ref(tmp_path) -> None:
    """StreamingHook with declarative emitter via @ref."""
    import json as _json

    ws = tmp_path
    (ws / "workspace.json").write_text(_json.dumps({"name": "x", "schema_version": 1}))
    (ws / "config.yaml").write_text("max_steps: 3\n")
    (ws / "tools/done").mkdir(parents=True)
    (ws / "tools/done/tool.yaml").write_text("name: done\nparameters:\n  s:\n    type: string\n")
    (ws / "tools/done/execute.py").write_text(
        "def execute(*, s): return {'status': 'completed', 's': s}\n"
    )
    (ws / "resources").mkdir()
    (ws / "resources/emitter.py").write_text(
        "from looplet.streaming import CallbackEmitter\n"
        "EVENTS = []\n"
        "def build(runtime=None): return CallbackEmitter(EVENTS.append)\n"
    )
    (ws / "hooks/00_StreamingHook").mkdir(parents=True)
    (ws / "hooks/00_StreamingHook/hook.py").write_text(
        "from looplet import StreamingHook as _SH\n"
        "class StreamingHook(_SH):\n"
        "    def to_config(self): return {'emitter': '@emitter'}\n"
    )
    (ws / "hooks/00_StreamingHook/config.yaml").write_text(
        'class_name: StreamingHook\nkwargs:\n  emitter: "@emitter"\n'
    )

    preset = workspace_to_preset(ws, strict=True)
    hook = preset.hooks[0]
    emitter = getattr(hook, "emitter", None) or getattr(hook, "_emitter", None)
    assert emitter is not None


def test_workspace_extends_other_workspace(tmp_path) -> None:
    """A workspace's setup.py can load another workspace and merge
    its tools/hooks. Demonstrates inheritance/composition between
    bundles without forking the loop."""
    import json as _json

    base = tmp_path / "base"
    ext = tmp_path / "ext"
    (base / "tools/done").mkdir(parents=True)
    (base / "workspace.json").write_text(_json.dumps({"name": "base", "schema_version": 1}))
    (base / "config.yaml").write_text("max_steps: 5\n")
    (base / "tools/done/tool.yaml").write_text("name: done\nparameters:\n  s:\n    type: string\n")
    (base / "tools/done/execute.py").write_text(
        "def execute(*, s): return {'status': 'completed', 's': s}\n"
    )

    (ext / "tools").mkdir(parents=True)
    (ext / "workspace.json").write_text(_json.dumps({"name": "ext", "schema_version": 1}))
    (ext / "config.yaml").write_text("max_steps: 10\n")
    (ext / "setup.py").write_text(
        "from looplet import workspace_to_preset\n"
        f"BASE = {str(base)!r}\n"
        "def setup(preset, resources, runtime=None):\n"
        "    base = workspace_to_preset(BASE)\n"
        "    for name, spec in base.tools._tools.items():\n"
        "        if name not in preset.tools._tools:\n"
        "            preset.tools.register(spec)\n"
        "    return preset\n"
    )
    preset = workspace_to_preset(ext, strict=True)
    assert "done" in preset.tools._tools


# ── @ref resolution in config.yaml (declarative LoopConfig services) ──


def test_compact_service_via_at_ref_in_config(tmp_path) -> None:
    """A workspace can wire ``LoopConfig.compact_service`` declaratively
    via ``compact_service: "@compact_service"`` in config.yaml plus a
    ``resources/compact_service.py`` builder — no setup.py needed."""
    import json as _json

    ws = tmp_path
    (ws / "workspace.json").write_text(_json.dumps({"name": "x", "schema_version": 1}))
    (ws / "config.yaml").write_text('max_steps: 3\ncompact_service: "@compact_service"\n')
    (ws / "resources").mkdir()
    (ws / "resources/compact_service.py").write_text(
        "from looplet.compact import PruneToolResults, TruncateCompact, compact_chain\n"
        "def build(runtime=None):\n"
        "    return compact_chain(PruneToolResults(keep_recent=4), TruncateCompact(keep_recent=2))\n"
    )

    preset = workspace_to_preset(ws, strict=True)
    assert preset.config.compact_service is not None
    # _CompactChain from looplet.compact wires through.
    assert type(preset.config.compact_service).__module__ == "looplet.compact"


def test_tracer_via_at_ref_in_config(tmp_path) -> None:
    """``tracer`` callable wired declaratively via @ref."""
    import json as _json

    ws = tmp_path
    (ws / "workspace.json").write_text(_json.dumps({"name": "x", "schema_version": 1}))
    (ws / "config.yaml").write_text('max_steps: 3\ntracer: "@tracer"\n')
    (ws / "resources").mkdir()
    (ws / "resources/tracer.py").write_text(
        "EVENTS = []\n"
        "def _trace(event, payload=None):\n"
        "    EVENTS.append((event, payload))\n"
        "def build(runtime=None):\n"
        "    return _trace\n"
    )

    preset = workspace_to_preset(ws, strict=True)
    assert preset.config.tracer is not None
    assert callable(preset.config.tracer)


def test_unresolved_at_ref_in_config_raises_in_strict(tmp_path) -> None:
    """A typo'd @ref in config.yaml fails loud at load time, same as
    hook kwargs — no silent string-into-LoopConfig leakage."""
    import json as _json

    import pytest

    ws = tmp_path
    (ws / "workspace.json").write_text(_json.dumps({"name": "x", "schema_version": 1}))
    (ws / "config.yaml").write_text('max_steps: 3\ntracer: "@nonexistent_resource"\n')

    with pytest.raises(WorkspaceSerializationError, match="unresolved resource reference"):
        workspace_to_preset(ws, strict=True)


def test_callable_loop_config_field_auto_emits_resource(tmp_path: Path) -> None:
    """Setting a callable LoopConfig field (compact_service) auto-emits
    ``compact_service: "@compact_service"`` into config.yaml and a
    matching ``resources/compact_service.py`` builder so the snapshot
    round-trips without a setup.py detour."""
    from looplet.compact import PruneToolResults, TruncateCompact, compact_chain

    preset = _build_demo_preset()
    preset.config.compact_service = compact_chain(
        PruneToolResults(keep_recent=4), TruncateCompact(keep_recent=2)
    )
    ws = preset_to_workspace(preset, tmp_path / "ws")

    cfg_text = (tmp_path / "ws" / "config.yaml").read_text()
    assert '"@compact_service"' in cfg_text or "@compact_service" in cfg_text
    assert (tmp_path / "ws" / "resources" / "compact_service.py").is_file()

    # Reload — compact_service should come back via the @ref machinery.
    reloaded = workspace_to_preset(tmp_path / "ws", strict=True)
    assert reloaded.config.compact_service is not None
    # Loose preset round-trip records no warnings for this field.
    assert not any("compact_service" in w for w in ws.serialization_warnings)


def test_builtin_hooks_round_trip_without_subclassing(tmp_path: Path) -> None:
    """MetricsHook, PermissionHook, StreamingHook, and EvalHook now ship
    with default ``to_config()`` so they round-trip without forcing
    users to subclass + override. Previously these silently dropped on
    reload because the loader couldn't supply the required ctor args.
    """
    from looplet import (
        EvalHook,
        EvalResult,
        MetricsCollector,
        MetricsHook,
        PermissionDecision,
        PermissionEngine,
        PermissionHook,
        PermissionRule,
        StreamingHook,
        ToolSpec,
    )
    from looplet.streaming import CallbackEmitter
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="done",
            description="done",
            parameters={"s": "summary"},
            execute=_done_execute,
        )
    )
    events: list = []
    hooks = [
        MetricsHook(collector=MetricsCollector()),
        PermissionHook(
            engine=PermissionEngine(
                rules=[PermissionRule(tool="*", decision=PermissionDecision.ALLOW)]
            )
        ),
        StreamingHook(emitter=CallbackEmitter(events.append)),
        EvalHook(evaluators=[_trivial_evaluator]),
    ]
    cfg = LoopConfig(max_steps=3, done_tool="done")
    preset = AgentPreset(config=cfg, hooks=hooks, tools=tools, state=DefaultState(max_steps=3))

    ws = preset_to_workspace(preset, tmp_path / "ws")
    # Resource stubs were emitted for the four hook ctor args.
    resources = sorted(p.name for p in (tmp_path / "ws" / "resources").iterdir())
    assert "collector.py" in resources
    assert "engine.py" in resources
    assert "emitter.py" in resources
    assert "evaluators.py" in resources

    reloaded = workspace_to_preset(tmp_path / "ws")
    # All four hooks survive — no silent drops.
    cls_names = sorted(type(h).__name__ for h in reloaded.hooks)
    assert cls_names == ["EvalHook", "MetricsHook", "PermissionHook", "StreamingHook"]


# ── Auto-emit improvements: list-of-callables + live-instance kwarg derivation ──


def _scripted_done(*, summary: str = "") -> dict:
    return {"status": "completed", "summary": summary}


def _eval_passed(ctx) -> "EvalResult":  # noqa: F821
    from looplet import EvalResult  # noqa: PLC0415

    return EvalResult(name="passed", passed=True)


def _eval_smoke(ctx) -> "EvalResult":  # noqa: F821
    from looplet import EvalResult  # noqa: PLC0415

    return EvalResult(name="smoke", passed=True)


def _emit_callback(event) -> None:
    pass


def test_list_of_top_level_callables_auto_emits_real_imports(tmp_path: Path) -> None:
    """When a hook kwarg holds a list of importable callables (e.g.
    ``EvalHook(evaluators=[a, b])``), the auto-emit machinery should
    write a builder that re-imports each callable by name — not fall
    back to a None-stub for ``builtins.list``."""
    from looplet import (
        AgentPreset,
        EvalHook,
        LoopConfig,
        ToolSpec,
    )
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="done", description="d", parameters={"summary": "s"}, execute=_scripted_done)
    )
    hooks = [EvalHook(evaluators=[_eval_passed, _eval_smoke])]
    cfg = LoopConfig(max_steps=3, done_tool="done")
    preset = AgentPreset(config=cfg, hooks=hooks, tools=tools, state=DefaultState(max_steps=3))

    preset_to_workspace(preset, tmp_path / "ws", strict=True)
    builder = (tmp_path / "ws" / "resources" / "evaluators.py").read_text()
    # Real ``from M import F`` lines for each evaluator, not a None-stub.
    assert "from tests.test_workspace import _eval_passed" in builder
    assert "from tests.test_workspace import _eval_smoke" in builder
    assert "return [_eval_passed, _eval_smoke]" in builder

    # Reload + check the evaluators came back identical.
    reloaded = workspace_to_preset(tmp_path / "ws", strict=True)
    eh = reloaded.hooks[0]
    assert [fn.__name__ for fn in eh.evaluators] == ["_eval_passed", "_eval_smoke"]


def test_live_instance_kwarg_derivation_for_emitter(tmp_path: Path) -> None:
    """When a hook holds an instance whose required ctor kwarg is a
    top-level callable (e.g. ``StreamingHook(emitter=CallbackEmitter(fn))``
    with a module-level ``fn``), the auto-emit builder must re-import
    the callable rather than falling through to ``runtime.get(...)``
    (which would yield None and crash the loop at run time)."""
    from looplet import (
        AgentPreset,
        LoopConfig,
        StreamingHook,
        ToolSpec,
    )
    from looplet.streaming import CallbackEmitter
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="done", description="d", parameters={"summary": "s"}, execute=_scripted_done)
    )
    hooks = [StreamingHook(emitter=CallbackEmitter(_emit_callback))]
    cfg = LoopConfig(max_steps=3, done_tool="done")
    preset = AgentPreset(config=cfg, hooks=hooks, tools=tools, state=DefaultState(max_steps=3))

    preset_to_workspace(preset, tmp_path / "ws", strict=True)
    builder = (tmp_path / "ws" / "resources" / "emitter.py").read_text()
    assert "from looplet.streaming import CallbackEmitter" in builder
    assert "from tests.test_workspace import _emit_callback" in builder
    assert "callback=_emit_callback" in builder
    # Specifically must NOT regress to the old ``runtime.get('callback')``
    # template, which silently produced a None callback and crashed at
    # ``self._callback(event)``.
    assert "runtime.get('callback')" not in builder

    # Reload — the callback should be the same module-level function.
    reloaded = workspace_to_preset(tmp_path / "ws", strict=True)
    emitter = reloaded.hooks[0]._emitter
    cb = getattr(emitter, "_callback", None) or getattr(emitter, "callback", None)
    assert cb is _emit_callback


def test_triple_round_trip_is_byte_idempotent(tmp_path: Path) -> None:
    """preset -> ws1 -> preset' -> ws2: every file in ws1 and ws2 must
    be byte-identical. Catches non-deterministic ordering and
    auto-emit drift."""
    from looplet import (
        AgentPreset,
        EvalHook,
        LoopConfig,
        StreamingHook,
        ToolSpec,
    )
    from looplet.streaming import CallbackEmitter
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="done", description="d", parameters={"summary": "s"}, execute=_scripted_done)
    )
    hooks = [
        EvalHook(evaluators=[_eval_passed, _eval_smoke]),
        StreamingHook(emitter=CallbackEmitter(_emit_callback)),
    ]
    cfg = LoopConfig(max_steps=3, done_tool="done")
    preset = AgentPreset(config=cfg, hooks=hooks, tools=tools, state=DefaultState(max_steps=3))

    ws1 = tmp_path / "ws1"
    preset_to_workspace(preset, ws1, name="rt", strict=True)
    preset2 = workspace_to_preset(ws1, strict=True)
    ws2 = tmp_path / "ws2"
    preset_to_workspace(preset2, ws2, name="rt", strict=True)

    files1 = sorted(
        p.relative_to(ws1) for p in ws1.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    files2 = sorted(
        p.relative_to(ws2) for p in ws2.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    assert files1 == files2, f"file-set drift: {set(files1) ^ set(files2)}"

    drifts = []
    for rel in files1:
        a = (ws1 / rel).read_text(encoding="utf-8")
        b = (ws2 / rel).read_text(encoding="utf-8")
        if a != b:
            drifts.append(str(rel))
    assert not drifts, f"byte drift in: {drifts}"


# ── CallableMemorySource round-trip + dataclass auto-emit ─────────────


def _live_state_load(state) -> str:
    """Top-level memory loader so workspace round-trip can re-import it."""
    step = getattr(state, "step_count", 0) or len(getattr(state, "steps", []) or [])
    return f"[live] step={step}"


def test_callable_memory_source_round_trips(tmp_path: Path) -> None:
    """``CallableMemorySource(fn=top_level_fn)`` round-trips losslessly:
    writer emits ``memory/<idx>_callable.py`` with a re-import; loader
    wraps the exported ``load`` symbol back into a CallableMemorySource."""
    from looplet import (
        AgentPreset,
        CallableMemorySource,
        LoopConfig,
        StaticMemorySource,
        ToolSpec,
    )
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="done", description="d", parameters={"summary": "s"}, execute=_scripted_done)
    )
    cfg = LoopConfig(
        max_steps=3,
        done_tool="done",
        memory_sources=[
            StaticMemorySource(text="A"),
            CallableMemorySource(fn=_live_state_load),
            StaticMemorySource(text="B"),
        ],
    )
    preset = AgentPreset(config=cfg, hooks=[], tools=tools, state=DefaultState(max_steps=3))
    ws = preset_to_workspace(preset, tmp_path / "ws", strict=True)
    assert ws.serialization_warnings == []
    assert (tmp_path / "ws" / "memory" / "01_callable.py").is_file()

    reloaded = workspace_to_preset(tmp_path / "ws", strict=True)
    types = [type(s).__name__ for s in (reloaded.config.memory_sources or [])]
    assert types == ["StaticMemorySource", "CallableMemorySource", "StaticMemorySource"]

    # The reloaded callable behaves identically.
    cm = reloaded.config.memory_sources[1]
    assert cm.fn(DefaultState(max_steps=3)) == "[live] step=0"


def test_callable_memory_source_lambda_warns_in_loose_mode(tmp_path: Path) -> None:
    """Lambdas / closures cannot be re-imported; writer falls back to a
    warning instead of dropping silently or crashing."""
    from looplet import (
        AgentPreset,
        CallableMemorySource,
        LoopConfig,
        ToolSpec,
    )
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="done", description="d", parameters={"summary": "s"}, execute=_scripted_done)
    )
    cfg = LoopConfig(
        max_steps=3,
        done_tool="done",
        memory_sources=[CallableMemorySource(fn=lambda state: "x")],
    )
    preset = AgentPreset(config=cfg, hooks=[], tools=tools, state=DefaultState(max_steps=3))
    ws = preset_to_workspace(preset, tmp_path / "ws", strict=False)
    assert any("CallableMemorySource" in w for w in ws.serialization_warnings)


def test_dataclass_auto_emit_reproduces_field_state(tmp_path: Path) -> None:
    """When a hook holds a dataclass instance with non-default field
    values (e.g. ``PermissionEngine(rules=[PermissionRule(...), ...])``),
    the auto-emit builder must reproduce every field — not just the
    required ctor args. Previously ``PermissionEngine`` round-tripped
    with empty rules because ``rules`` has a default_factory and the
    generic builder skipped non-required kwargs."""
    from looplet import (
        AgentPreset,
        LoopConfig,
        PermissionDecision,
        PermissionEngine,
        PermissionHook,
        PermissionRule,
        ToolSpec,
    )
    from looplet.tools import BaseToolRegistry
    from looplet.types import DefaultState

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(name="done", description="d", parameters={"summary": "s"}, execute=_scripted_done)
    )
    engine = PermissionEngine(
        rules=[
            PermissionRule(tool="dangerous", decision=PermissionDecision.DENY, reason="audit-only"),
            PermissionRule(tool="*", decision=PermissionDecision.ALLOW),
        ]
    )
    hooks = [PermissionHook(engine=engine)]
    cfg = LoopConfig(max_steps=3, done_tool="done")
    preset = AgentPreset(config=cfg, hooks=hooks, tools=tools, state=DefaultState(max_steps=3))

    preset_to_workspace(preset, tmp_path / "ws", strict=True)
    builder = (tmp_path / "ws" / "resources" / "engine.py").read_text()
    # Real reproduction — not an empty-rules shell.
    assert "PermissionEngine(rules=[" in builder
    assert "tool='dangerous'" in builder
    assert "PermissionDecision.DENY" in builder

    reloaded = workspace_to_preset(tmp_path / "ws", strict=True)
    eng = reloaded.hooks[0].engine
    assert [r.tool for r in eng.rules] == ["dangerous", "*"]
    assert [r.decision for r in eng.rules] == [
        PermissionDecision.DENY,
        PermissionDecision.ALLOW,
    ]


# ── v2 tool.yaml optional-default + _chw_* dynamic-module round-trip ──


def test_v2_tool_yaml_default_marks_param_optional(tmp_path: Path) -> None:
    """``tool.yaml`` parameters declared as
    ``{name: {type, description, default}}`` (the format every shipped
    workspace uses) must treat ``default``-bearing entries as optional.
    Previously the loader fell into the simple-format branch which
    only recognised the ``"(optional) ..."`` description prefix, so
    every dict-shaped param was reported missing on dispatch."""
    from looplet import ToolSpec

    spec = ToolSpec(
        name="lst",
        description="d",
        parameters={
            "path": {"type": "string", "description": "p"},
            "depth": {"type": "integer", "description": "d", "default": 2},
        },
        execute=lambda **k: k,
    )
    assert spec.required_parameters() == ["path"]


def test_chw_resource_round_trip_copies_original_source(tmp_path: Path) -> None:
    """When a hook holds a live instance whose class came from the
    workspace's own ``_chw_resource_<name>`` dynamic module, the
    snapshot writer must copy the original ``resources/<name>.py`` file
    verbatim instead of dropping a None-stub. Catches the case where
    a workspace defines a custom resource class inline (e.g.
    ``GreetingLog`` in the hello example)."""
    src = tmp_path / "src_ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "src_ws"}')
    (src / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "def execute(*, s='ok'): return {'status': 'completed', 's': s}\n"
    )
    (src / "resources").mkdir()
    custom_src = (
        '"""Custom inline resource — exists only inside this workspace."""\n'
        "class CustomLog:\n"
        "    def __init__(self):\n"
        "        self.entries = []\n"
        "def build():\n"
        "    return CustomLog()\n"
    )
    (src / "resources/custom_log.py").write_text(custom_src)
    (src / "hooks" / "00_LogHook").mkdir(parents=True)
    (src / "hooks/00_LogHook/hook.py").write_text(
        "class LogHook:\n"
        "    def __init__(self, *, log): self.log = log\n"
        "    def to_config(self): return {'log': '@custom_log'}\n"
    )
    (src / "hooks/00_LogHook/config.yaml").write_text(
        'class_name: LogHook\nkwargs:\n  log: "@custom_log"\n'
    )

    preset = workspace_to_preset(src, strict=True)
    snap = tmp_path / "snap"
    ws = preset_to_workspace(preset, snap, name="snap", strict=False)
    # Resource source must come back verbatim — no None-stub warnings.
    assert not any("custom_log" in w for w in ws.serialization_warnings), (
        f"unexpected warnings: {ws.serialization_warnings}"
    )
    snap_resource = (snap / "resources" / "custom_log.py").read_text()
    assert snap_resource == custom_src

    # Reload the snapshot — the LogHook's log attribute must be a fresh
    # CustomLog instance (not None).
    reloaded = workspace_to_preset(snap, strict=True)
    assert type(reloaded.hooks[0].log).__name__ == "CustomLog"


def test_chw_hook_inline_class_round_trips(tmp_path: Path) -> None:
    """When a hook's class is defined inline in the workspace's own
    ``hooks/<name>/hook.py`` (not subclassing an installed class), the
    re-snapshot writer must copy the on-disk source instead of falling
    back to ``class X: pass``."""
    src = tmp_path / "src_ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "src_ws"}')
    (src / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "def execute(*, s='ok'): return {'status': 'completed', 's': s}\n"
    )
    (src / "hooks" / "00_GateHook").mkdir(parents=True)
    hook_src = (
        '"""Inline gate hook with a meaningful body."""\n'
        "class GateHook:\n"
        "    SENTINEL = 'inline-marker-XYZ'\n"
        "    def to_config(self): return {}\n"
    )
    (src / "hooks/00_GateHook/hook.py").write_text(hook_src)

    preset = workspace_to_preset(src, strict=True)
    snap = tmp_path / "snap"
    preset_to_workspace(preset, snap, name="snap", strict=True)
    snap_hook = (snap / "hooks" / "00_GateHook" / "hook.py").read_text()
    # Must contain the inline marker — proves source was preserved.
    assert "inline-marker-XYZ" in snap_hook


# ── Declarative @ref memory + workspace-helper copy + byte-identity ──


def _project_memory_loader(state) -> str:
    """Top-level loader returning a per-step memory line."""
    return f"step={getattr(state, 'step_count', 0)}"


def test_memory_sources_yaml_ref_resolves_via_resource_registry(tmp_path: Path) -> None:
    """``memory_sources: ['@ref']`` in config.yaml round-trips through
    the same resource-builder mechanism the hook kwargs use."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "w"}')
    (src / "config.yaml").write_text(
        'max_steps: 3\ndone_tool: done\nmemory_sources:\n  - "@dyn_memory"\n'
    )
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "def execute(*, s='ok'): return {'status': 'completed', 's': s}\n"
    )
    (src / "resources").mkdir()
    (src / "resources/dyn_memory.py").write_text(
        "from looplet import CallableMemorySource\n"
        "def build(runtime=None):\n"
        "    return CallableMemorySource(lambda state: 'dyn-memory-content')\n"
    )

    preset = workspace_to_preset(src, strict=True)
    sources = preset.config.memory_sources or []
    assert len(sources) == 1
    assert type(sources[0]).__name__ == "CallableMemorySource"
    assert sources[0].load(None) == "dyn-memory-content"


def test_callable_memory_via_chw_resource_round_trips_losslessly(tmp_path: Path) -> None:
    """A ``CallableMemorySource`` whose lambda was built inside a
    ``resources/<name>.py`` builder must snapshot back to a
    ``memory_sources: ['@<name>']`` entry + verbatim resource copy,
    not warn about a non-importable lambda."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "w"}')
    (src / "config.yaml").write_text(
        'max_steps: 3\ndone_tool: done\nmemory_sources:\n  - "@dyn_memory"\n'
    )
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "def execute(*, s='ok'): return {'status': 'completed', 's': s}\n"
    )
    (src / "resources").mkdir()
    builder_src = (
        "from looplet import CallableMemorySource\n"
        "def build(runtime=None):\n"
        "    runtime = runtime or {}\n"
        "    label = str(runtime.get('label', 'X'))\n"
        "    return CallableMemorySource(lambda state: f'lab={label}')\n"
    )
    (src / "resources/dyn_memory.py").write_text(builder_src)

    preset = workspace_to_preset(src, runtime={"label": "Z"}, strict=True)
    snap = tmp_path / "snap"
    ws = preset_to_workspace(preset, snap, name="snap")
    assert ws.serialization_warnings == [], f"unexpected warnings: {ws.serialization_warnings}"
    # Snapshot config.yaml carries the @ref entry, not a None-stub class:
    snap_cfg = (snap / "config.yaml").read_text()
    assert '"@dyn_memory"' in snap_cfg or "@dyn_memory" in snap_cfg
    # And the resource file was copied verbatim:
    assert (snap / "resources" / "dyn_memory.py").read_text() == builder_src


def test_workspace_helpers_are_copied_into_snapshot(tmp_path: Path) -> None:
    """Top-level ``*.py`` helper modules at the source workspace root
    (e.g. ``coder_lib_tools.py``) must be vendored into the snapshot
    so cross-process reload doesn't crash with ModuleNotFoundError."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "w"}')
    (src / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "from helper_lib import HELLO\n"
        "def execute(*, s='ok'): return {'status': 'completed', 'msg': HELLO + s}\n"
    )
    helper_src = "HELLO = 'hi-from-helper-'\n"
    (src / "helper_lib.py").write_text(helper_src)

    preset = workspace_to_preset(src, strict=True)
    snap = tmp_path / "snap"
    preset_to_workspace(preset, snap, name="snap")
    # The helper module was copied into the snapshot.
    assert (snap / "helper_lib.py").read_text() == helper_src


def test_two_pass_round_trip_is_byte_identical(tmp_path: Path) -> None:
    """preset → ws1 → preset' → ws2: ws1 and ws2 must be byte-identical
    for a non-trivial preset. Catches header-stacking, MRO-walk drift,
    and other normalization gaps."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "w"}')
    (src / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "def execute(*, s='ok'): return {'status': 'completed', 's': s}\n"
    )
    (src / "hooks" / "00_StagnationHook").mkdir(parents=True)
    (src / "hooks/00_StagnationHook/hook.py").write_text(
        "from looplet.stagnation import StagnationHook as StagnationHook\n"
    )
    (src / "hooks/00_StagnationHook/config.yaml").write_text(
        "class_name: StagnationHook\nkwargs: {}\n"
    )

    preset_a = workspace_to_preset(src, strict=True)
    ws1 = tmp_path / "ws1"
    preset_to_workspace(preset_a, ws1, name="rt")
    preset_b = workspace_to_preset(ws1, strict=True)
    ws2 = tmp_path / "ws2"
    preset_to_workspace(preset_b, ws2, name="rt")

    files1 = sorted(
        p.relative_to(ws1) for p in ws1.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    files2 = sorted(
        p.relative_to(ws2) for p in ws2.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    assert files1 == files2

    diffs = []
    for rel in files1:
        a = (ws1 / rel).read_text(encoding="utf-8")
        b = (ws2 / rel).read_text(encoding="utf-8")
        if a != b:
            diffs.append((str(rel), len(a), len(b)))
    assert not diffs, f"byte drift: {diffs}"


# ── Tool DI via requires + ctx.resources ──


def _di_tool_execute(ctx, *, x: int) -> dict:
    """Top-level tool that reads its dependency from ctx.resources."""
    cfg = ctx.resources.get("workspace_config")
    return {"x": x, "label": getattr(cfg, "label", "no-cfg")}


def test_tool_requires_round_trips_via_yaml(tmp_path: Path) -> None:
    """``tool.yaml`` ``requires:`` field round-trips through
    ``ToolSpec.requires`` and the dispatcher passes the resolved
    instances via ``ctx.resources[name]``."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "workspace.json").write_text('{"name": "w"}')
    (src / "config.yaml").write_text("max_steps: 3\ndone_tool: done\n")
    (src / "tools" / "done").mkdir(parents=True)
    (src / "tools/done/tool.yaml").write_text(
        "name: done\nparameters:\n  s: {type: string, description: s}\n"
    )
    (src / "tools/done/execute.py").write_text(
        "def execute(*, s='ok'): return {'status': 'completed', 's': s}\n"
    )
    (src / "tools" / "demo").mkdir(parents=True)
    (src / "tools/demo/tool.yaml").write_text(
        "name: demo\n"
        "parameters:\n  x: {type: integer, description: x}\n"
        "requires:\n  - workspace_config\n"
    )
    (src / "tools/demo/execute.py").write_text(
        "from tests.test_workspace import _di_tool_execute as execute\n"
    )
    (src / "resources").mkdir()
    (src / "resources/workspace_config.py").write_text(
        "class _Cfg:\n    label = 'configured'\ndef build(runtime=None):\n    return _Cfg()\n"
    )

    preset = workspace_to_preset(src, strict=True)
    spec = preset.tools._tools["demo"]
    assert spec.requires == ["workspace_config"]
    # Dispatch through the registry → ctx.resources receives the live cfg.
    from looplet.types import ToolCall

    result = preset.tools.dispatch(ToolCall(tool="demo", args={"x": 7}))
    assert result.error is None
    assert result.data == {"x": 7, "label": "configured"}

    # Snapshot round-trip preserves requires.
    snap = tmp_path / "snap"
    preset_to_workspace(preset, snap, name="snap")
    snap_yaml = (snap / "tools" / "demo" / "tool.yaml").read_text()
    assert "requires:" in snap_yaml
    assert "workspace_config" in snap_yaml


def test_tool_without_ctx_logs_warning_when_requires_set(tmp_path: Path, caplog) -> None:
    """A ``requires:`` declaration on a tool that doesn't accept ``ctx``
    is a configuration mistake — the dispatcher logs a warning so
    users notice instead of silently dropping the dependency."""
    import logging

    from looplet import ToolSpec
    from looplet.tools import BaseToolRegistry
    from looplet.types import ToolCall

    def _no_ctx_execute(*, x: int) -> dict:
        return {"x": x}

    reg = BaseToolRegistry()
    reg.set_resources({"workspace_config": object()})
    reg.register(
        ToolSpec(
            name="demo",
            description="d",
            parameters={"x": {"type": "integer", "description": "x"}},
            execute=_no_ctx_execute,
            requires=["workspace_config"],
        )
    )

    with caplog.at_level(logging.WARNING, logger="looplet.tools"):
        result = reg.dispatch(ToolCall(tool="demo", args={"x": 1}))
    assert result.error is None
    assert any("ctx" in rec.message for rec in caplog.records)
