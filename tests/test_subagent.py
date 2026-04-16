"""Tests for openharness.subagent — run_sub_loop and _MinimalState."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Mock setup helpers ────────────────────────────────────────────


def _make_mock_step(number: int, tool: str = "search") -> Any:
    """Create a minimal step-like object with .to_dict() and .summary()."""
    step = MagicMock()
    step.number = number
    step.to_dict.return_value = {
        "step": number,
        "call": {"tool": tool, "args": {}},
        "result": {"data": f"result_{number}"},
    }
    step.summary.return_value = f"S{number} ✓ {tool}"
    return step


def _make_mock_session_log(entities: set | None = None, findings: list | None = None,
                            highlights: list | None = None) -> Any:
    log = MagicMock()
    log.all_entities.return_value = entities if entities is not None else set()
    log.render.return_value = ""
    # entries is used by subagent to collect findings/highlights
    entry = MagicMock()
    entry.findings = findings or []
    entry.highlights = highlights or []
    log._entries = [entry] if (findings or highlights) else []
    return log


def _make_mock_tool_spec(name: str) -> Any:
    spec = MagicMock()
    spec.name = name
    spec.description = f"Tool {name}"
    spec.parameters = {}
    spec.execute = lambda **kw: None
    spec.concurrent_safe = False
    return spec


def _make_mock_registry(tool_names: list[str]) -> Any:
    registry = MagicMock()
    registry._tools = {name: _make_mock_tool_spec(name) for name in tool_names}
    return registry


def _make_loop_generator(steps: list, return_value: dict | None = None):
    """Create a generator that yields steps and returns a dict via StopIteration."""
    def _gen():
        for step in steps:
            yield step
        return return_value or {"llm_calls": 2, "total_time_ms": 100}
    return _gen()


def _inject_mocks(mock_session_log: Any, mock_registry: Any):
    """Inject mock cadence.session, cadence.loop, cadence.tools into sys.modules."""
    # Mock cadence.session
    mock_session_mod = MagicMock()
    mock_session_mod.SessionLog.return_value = mock_session_log

    # Mock cadence.tools
    mock_tools_mod = MagicMock()
    mock_tools_mod.BaseToolRegistry.return_value = _make_mock_registry([])

    # _clone_tools_excluding needs ToolSpec and BaseToolRegistry
    mock_new_registry = _make_mock_registry([])
    mock_new_registry._tools = {}
    mock_tools_mod.BaseToolRegistry.return_value = mock_new_registry

    mock_tool_spec_cls = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    mock_tools_mod.ToolSpec = mock_tool_spec_cls

    # Mock cadence.loop
    mock_loop_mod = MagicMock()

    return {
        "openharness.session": mock_session_mod,
        "openharness.tools": mock_tools_mod,
        "openharness.loop": mock_loop_mod,
    }


# ── _MinimalState tests ────────────────────────────────────────────


class TestMinimalState:
    def test_importable(self):
        from openharness.subagent import _MinimalState
        assert _MinimalState is not None

    def test_initial_state(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState(task={"id": "t1"}, max_steps=5)
        assert state.steps == []
        assert state.queries_used == 0
        assert state.step_count == 0
        assert state.budget_remaining == 5
        assert state.max_steps == 5

    def test_budget_remaining_decreases(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState(max_steps=5)
        mock_step = MagicMock()
        mock_step.summary.return_value = "S1 ✓ search"
        state.steps.append(mock_step)
        assert state.budget_remaining == 4

    def test_step_count_reflects_steps(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState()
        assert state.step_count == 0
        state.steps.append(MagicMock())
        assert state.step_count == 1

    def test_context_summary_empty(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState()
        result = state.context_summary()
        assert "no steps" in result.lower()

    def test_context_summary_with_steps(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState()
        step = MagicMock()
        step.summary.return_value = "S1 ✓ search"
        state.steps.append(step)
        result = state.context_summary()
        assert "S1" in result

    def test_snapshot(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState(max_steps=3)
        snap = state.snapshot()
        assert "step_count" in snap
        assert "budget_remaining" in snap
        assert snap["budget_remaining"] == 3

    def test_task_stored(self):
        from openharness.subagent import _MinimalState
        task = {"id": "abc", "description": "test"}
        state = _MinimalState(task=task)
        assert state.task == task

    def test_default_task_empty_dict(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState()
        assert state.task == {}

    def test_budget_never_negative(self):
        from openharness.subagent import _MinimalState
        state = _MinimalState(max_steps=1)
        for _ in range(5):
            state.steps.append(MagicMock())
        assert state.budget_remaining == 0


# ── run_sub_loop signature tests ────────────────────────────────────


class TestRunSubLoopSignature:
    def test_importable(self):
        from openharness.subagent import run_sub_loop
        assert callable(run_sub_loop)

    def test_no_alert_param(self):
        """Backward-compat 'alert' param must NOT be present."""
        import inspect
        from openharness.subagent import run_sub_loop
        sig = inspect.signature(run_sub_loop)
        assert "alert" not in sig.parameters, "alert param should be removed"

    def test_no_exploration_param(self):
        """Backward-compat 'exploration' param must NOT be present."""
        import inspect
        from openharness.subagent import run_sub_loop
        sig = inspect.signature(run_sub_loop)
        assert "exploration" not in sig.parameters, "exploration param should be removed"

    def test_has_required_params(self):
        import inspect
        from openharness.subagent import run_sub_loop
        sig = inspect.signature(run_sub_loop)
        params = sig.parameters
        assert "llm" in params
        assert "task" in params
        assert "tools" in params
        assert "max_steps" in params
        assert "system_prompt" in params
        assert "build_summary" in params
        assert "state_mutating_tools" in params


# ── run_sub_loop execution tests ────────────────────────────────────


class TestRunSubLoopExecution:
    def _run_with_mocks(self, steps=None, entities=None, findings=None, highlights=None,
                        build_summary=None, tools=None, state_mutating_tools=None):
        """Helper: run run_sub_loop with all cadence deps mocked."""
        from openharness.subagent import run_sub_loop

        mock_log = _make_mock_session_log(
            entities=entities or {"entity1", "entity2"},
            findings=findings or ["finding1"],
            highlights=highlights or ["highlight1"],
        )
        mock_steps = steps or [_make_mock_step(1), _make_mock_step(2)]

        # Mock composable_loop to return a generator over our mock steps
        def mock_composable_loop(**kwargs):
            return _make_loop_generator(mock_steps, {"llm_calls": 3, "total_time_ms": 50})

        mock_lc = MagicMock()
        mock_lc_instance = MagicMock()
        mock_loop_mod = MagicMock()
        mock_loop_mod.composable_loop = mock_composable_loop
        mock_loop_mod.LoopConfig = mock_lc
        mock_lc.return_value = mock_lc_instance

        mock_session_mod = MagicMock()
        mock_session_mod.SessionLog.return_value = mock_log

        parent_registry = tools or _make_mock_registry(["search", "done"])

        mock_new_registry = MagicMock()
        mock_new_registry._tools = {}
        mock_new_registry._register = MagicMock()

        mock_tools_mod = MagicMock()
        mock_tools_mod.BaseToolRegistry.return_value = mock_new_registry
        mock_tools_mod.ToolSpec = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

        with patch.dict(sys.modules, {
            "openharness.loop": mock_loop_mod,
            "openharness.session": mock_session_mod,
            "openharness.tools": mock_tools_mod,
        }):
            result = run_sub_loop(
                llm=MagicMock(),
                task={"id": "test"},
                tools=parent_registry,
                max_steps=5,
                build_summary=build_summary,
                state_mutating_tools=state_mutating_tools,
            )
        return result

    def test_returns_dict(self):
        result = self._run_with_mocks()
        assert isinstance(result, dict)

    def test_return_has_summary(self):
        result = self._run_with_mocks()
        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_return_has_entities(self):
        result = self._run_with_mocks(entities={"e1", "e2"})
        assert "entities" in result
        assert isinstance(result["entities"], list)

    def test_return_has_findings(self):
        result = self._run_with_mocks(findings=["f1", "f2"])
        assert "findings" in result
        assert isinstance(result["findings"], list)

    def test_return_has_highlights(self):
        result = self._run_with_mocks(highlights=["h1", "h2"])
        assert "highlights" in result
        assert isinstance(result["highlights"], list)

    def test_return_has_llm_calls(self):
        result = self._run_with_mocks()
        assert "llm_calls" in result
        assert isinstance(result["llm_calls"], int)

    def test_return_has_steps(self):
        result = self._run_with_mocks(steps=[_make_mock_step(1), _make_mock_step(2)])
        assert "steps" in result
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) == 2

    def test_all_required_keys_present(self):
        """All required keys from acceptance criteria must be present."""
        result = self._run_with_mocks()
        required = {"summary", "entities", "findings", "highlights", "llm_calls", "steps"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_steps_are_dicts(self):
        result = self._run_with_mocks(steps=[_make_mock_step(1)])
        assert all(isinstance(s, dict) for s in result["steps"])

    def test_llm_calls_from_generator_return(self):
        """llm_calls should come from the generator's StopIteration value."""
        result = self._run_with_mocks()
        assert result["llm_calls"] == 3  # from our mock return value

    def test_custom_build_summary_invoked(self):
        """build_summary callable receives (state, session_log, steps) and its dict is merged."""
        calls = []

        def my_summary(state, session_log, steps_dicts):
            calls.append((state, session_log, steps_dicts))
            return {"summary": "custom summary", "entities": ["e1"], "custom_key": "custom_val"}

        result = self._run_with_mocks(build_summary=my_summary)
        assert len(calls) == 1
        assert result["summary"] == "custom summary"
        assert "custom_key" in result
        assert result["custom_key"] == "custom_val"

    def test_custom_build_summary_receives_steps_list(self):
        """build_summary receives the steps dicts list."""
        received_steps = []

        def my_summary(state, session_log, steps_dicts):
            received_steps.extend(steps_dicts)
            return {"summary": "ok", "entities": []}

        mock_steps = [_make_mock_step(1), _make_mock_step(2)]
        self._run_with_mocks(steps=mock_steps, build_summary=my_summary)
        assert len(received_steps) == 2


# ── Tool registry cloning tests ────────────────────────────────────


class TestToolRegistryCloning:
    def _run_and_get_cloned_registry(self, parent_tools: list[str], exclude: list[str] | None = None):
        """Run sub_loop and capture what tools were registered in sub-registry."""
        from openharness.subagent import run_sub_loop

        registered: list[str] = []

        mock_log = _make_mock_session_log()

        def mock_composable_loop(**kwargs):
            return _make_loop_generator([], {"llm_calls": 0})

        mock_loop_mod = MagicMock()
        mock_loop_mod.composable_loop = mock_composable_loop
        mock_loop_mod.LoopConfig = MagicMock(return_value=MagicMock())

        mock_session_mod = MagicMock()
        mock_session_mod.SessionLog.return_value = mock_log

        # Track what gets registered
        mock_new_registry = MagicMock()
        mock_new_registry._tools = {}

        def fake_register(spec):
            registered.append(spec.name)
            mock_new_registry._tools[spec.name] = spec

        mock_new_registry.register = fake_register
        mock_new_registry._register = fake_register  # backward-compat

        mock_tools_mod = MagicMock()
        mock_tools_mod.BaseToolRegistry.return_value = mock_new_registry

        # ToolSpec returns an object with the name attribute
        def fake_tool_spec(**kw):
            obj = MagicMock()
            obj.name = kw["name"]
            for k, v in kw.items():
                setattr(obj, k, v)
            return obj

        mock_tools_mod.ToolSpec = fake_tool_spec

        parent_registry = _make_mock_registry(parent_tools)

        with patch.dict(sys.modules, {
            "openharness.loop": mock_loop_mod,
            "openharness.session": mock_session_mod,
            "openharness.tools": mock_tools_mod,
        }):
            run_sub_loop(
                llm=MagicMock(),
                task={},
                tools=parent_registry,
                state_mutating_tools=exclude,
            )
        return registered

    def test_done_tool_excluded_by_default(self):
        """'done' tool is excluded from sub-agent registry by default."""
        registered = self._run_and_get_cloned_registry(["search", "done", "analyze"])
        assert "done" not in registered

    def test_other_tools_included(self):
        """Non-excluded tools are copied to sub-registry."""
        registered = self._run_and_get_cloned_registry(["search", "done", "analyze"])
        assert "search" in registered
        assert "analyze" in registered

    def test_custom_exclusions(self):
        """Custom state_mutating_tools are excluded."""
        registered = self._run_and_get_cloned_registry(
            ["search", "done", "write_file"],
            exclude=["done", "write_file"]
        )
        assert "done" not in registered
        assert "write_file" not in registered
        assert "search" in registered


# ── Parent state isolation tests ─────────────────────────────────


class TestParentStateIsolation:
    def test_parent_state_unaffected(self):
        """Parent state budget/steps unchanged after sub_loop."""
        from openharness.subagent import run_sub_loop

        parent_state = SimpleNamespace()
        parent_state.steps = []
        parent_state.queries_used = 5

        mock_log = _make_mock_session_log()

        def mock_composable_loop(**kwargs):
            return _make_loop_generator([_make_mock_step(1), _make_mock_step(2)],
                                        {"llm_calls": 2})

        mock_loop_mod = MagicMock()
        mock_loop_mod.composable_loop = mock_composable_loop
        mock_loop_mod.LoopConfig = MagicMock(return_value=MagicMock())

        mock_session_mod = MagicMock()
        mock_session_mod.SessionLog.return_value = mock_log

        mock_new_registry = MagicMock()
        mock_new_registry._tools = {}
        mock_new_registry._register = MagicMock()
        mock_tools_mod = MagicMock()
        mock_tools_mod.BaseToolRegistry.return_value = mock_new_registry
        mock_tools_mod.ToolSpec = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

        with patch.dict(sys.modules, {
            "openharness.loop": mock_loop_mod,
            "openharness.session": mock_session_mod,
            "openharness.tools": mock_tools_mod,
        }):
            run_sub_loop(
                llm=MagicMock(),
                task={},
                tools=_make_mock_registry(["search"]),
            )

        # Parent state completely untouched
        assert parent_state.steps == []
        assert parent_state.queries_used == 5

    def test_sub_loop_uses_minimal_state_not_parent(self):
        """When no state provided, _MinimalState is used (not parent state)."""
        from openharness.subagent import run_sub_loop, _MinimalState

        captured_states = []

        def mock_composable_loop(**kwargs):
            captured_states.append(kwargs.get("state"))
            return _make_loop_generator([], {"llm_calls": 0})

        mock_loop_mod = MagicMock()
        mock_loop_mod.composable_loop = mock_composable_loop
        mock_loop_mod.LoopConfig = MagicMock(return_value=MagicMock())

        mock_log = _make_mock_session_log()
        mock_session_mod = MagicMock()
        mock_session_mod.SessionLog.return_value = mock_log

        mock_new_registry = MagicMock()
        mock_new_registry._tools = {}
        mock_new_registry._register = MagicMock()
        mock_tools_mod = MagicMock()
        mock_tools_mod.BaseToolRegistry.return_value = mock_new_registry
        mock_tools_mod.ToolSpec = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

        with patch.dict(sys.modules, {
            "openharness.loop": mock_loop_mod,
            "openharness.session": mock_session_mod,
            "openharness.tools": mock_tools_mod,
        }):
            run_sub_loop(
                llm=MagicMock(),
                task={"id": "t"},
                tools=_make_mock_registry(["search"]),
            )

        assert len(captured_states) == 1
        assert isinstance(captured_states[0], _MinimalState)


# ── No primal_security / backward-compat tests ────────────────────


class TestNoDomainSpecificCode:
    def test_no_primal_security_imports(self):
        import inspect
        import openharness.subagent as mod
        source = inspect.getsource(mod)
        assert "primal_security" not in source

    def test_no_alert_in_source(self):
        """No backward-compat 'alert' param in source."""
        import inspect
        import openharness.subagent as mod
        source = inspect.getsource(mod)
        assert "alert" not in source or "# alert" in source or "alert_id" in source
        # More precise: check function signature
        import openharness.subagent
        import inspect as ins
        sig = ins.signature(openharness.subagent.run_sub_loop)
        assert "alert" not in sig.parameters
