"""Smoke tests for runnable skill bundles."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from looplet import SkillRuntime, load_skill_bundle, run_skill_bundle, validate_skill_bundle
from looplet.__main__ import main as cli_main
from looplet.presets import AgentPreset
from looplet.testing import MockLLMBackend

pytestmark = pytest.mark.smoke

CODER_BUNDLE = Path(__file__).resolve().parents[1] / "examples" / "coder" / "skill"


class TestSkillBundles:
    def test_imports_from_top_level(self):
        from looplet import BundleValidation, SkillBundle  # noqa: F401

    def test_loads_coder_bundle_card(self):
        bundle = load_skill_bundle(CODER_BUNDLE)

        assert bundle.skill.name == "coder"
        assert bundle.card.name == "coder"
        assert "coding" in bundle.card.tags

    def test_bundle_entrypoint_can_import_project_local_helpers_from_external_cwd(
        self,
        tmp_path,
        monkeypatch,
    ):
        project = tmp_path / "project"
        bundle_root = project / "nested" / "skill"
        bundle_root.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (project / "helper.py").write_text(
            "from looplet.presets import AgentPreset\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "def build_preset():\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=3),\n"
            "        state=DefaultState(max_steps=3),\n"
            "    )\n",
            encoding="utf-8",
        )
        (bundle_root / "SKILL.md").write_text(
            "---\nname: demo\ndescription: Demo bundle.\nentrypoint: looplet.py\n---\n# Demo\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from helper import build_preset\ndef build(runtime):\n    return build_preset()\n",
            encoding="utf-8",
        )
        external = tmp_path / "external"
        external.mkdir()
        monkeypatch.chdir(external)

        result = validate_skill_bundle(bundle_root)

        assert result.ok, result.errors

    def test_bundle_entrypoint_directory_precedes_project_root_imports(self, tmp_path):
        project = tmp_path / "project"
        bundle_root = project / "skill"
        bundle_root.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (project / "helper.py").write_text("VALUE = 'project'\n", encoding="utf-8")
        (bundle_root / "helper.py").write_text("VALUE = 'bundle'\n", encoding="utf-8")
        (bundle_root / "SKILL.md").write_text(
            "---\nname: import-order\ndescription: Import order bundle.\nentrypoint: looplet.py\n---\n# Order\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from helper import VALUE\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def loaded_value():\n"
            "    return VALUE\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        loaded = load_skill_bundle(bundle_root)

        assert loaded.module.loaded_value() == "bundle"

    def test_bundle_entrypoint_project_local_imports_are_isolated(self, tmp_path):
        loaded_values = []
        for name, value in (("one", "FIRST"), ("two", "SECOND")):
            project = tmp_path / name
            bundle_root = project / "skill"
            bundle_root.mkdir(parents=True)
            (project / "pyproject.toml").write_text(
                f"[project]\nname = '{name}'\n",
                encoding="utf-8",
            )
            (project / "helper.py").write_text(f"VALUE = '{value}'\n", encoding="utf-8")
            (bundle_root / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {name} bundle.\nentrypoint: looplet.py\n---\n# {name}\n",
                encoding="utf-8",
            )
            (bundle_root / "looplet.py").write_text(
                "from helper import VALUE\n"
                "from looplet import DefaultState, LoopConfig, tools_from\n"
                "from looplet.presets import AgentPreset\n"
                "def build(runtime):\n"
                "    return AgentPreset(\n"
                "        tools=tools_from([], include_done=True),\n"
                "        hooks=[],\n"
                "        config=LoopConfig(max_steps=1),\n"
                "        state=DefaultState(max_steps=1),\n"
                "    )\n"
                "def loaded_value():\n"
                "    return VALUE\n",
                encoding="utf-8",
            )

            loaded_values.append(load_skill_bundle(bundle_root).module.loaded_value())

        assert loaded_values == ["FIRST", "SECOND"]

    def test_bundle_import_context_keeps_protected_modules_loaded(self, tmp_path):
        project = tmp_path / "project"
        bundle_root = project / "skill"
        bundle_root.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (project / "pytest.py").write_text("VALUE = 'local pytest'\n", encoding="utf-8")
        (bundle_root / "SKILL.md").write_text(
            "---\nname: demo\ndescription: Demo bundle.\nentrypoint: looplet.py\n---\n# Demo\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )
        loaded_pytest = sys.modules["pytest"]

        result = validate_skill_bundle(bundle_root)

        assert result.ok, result.errors
        assert sys.modules["pytest"] is loaded_pytest

    def test_bundle_local_examples_package_can_shadow_loaded_examples(self, tmp_path):
        from examples.coder import agent as coder  # noqa: F401

        original_agent_module = sys.modules["examples.coder.agent"]
        project = tmp_path / "project"
        bundle_root = project / "skill"
        examples_pkg = project / "examples"
        bundle_root.mkdir(parents=True)
        examples_pkg.mkdir()
        (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (examples_pkg / "helper.py").write_text("VALUE = 'local examples'\n", encoding="utf-8")
        (bundle_root / "SKILL.md").write_text(
            "---\nname: examples-shadow\ndescription: Examples shadow bundle.\nentrypoint: looplet.py\n---\n# Shadow\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from examples.helper import VALUE\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def loaded_value():\n"
            "    return VALUE\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        loaded = load_skill_bundle(bundle_root)

        assert loaded.module.loaded_value() == "local examples"
        assert "examples.helper" not in sys.modules
        assert sys.modules["examples.coder.agent"] is original_agent_module
        from examples.coder import agent as restored_coder

        assert restored_coder is coder

    def test_standalone_bundle_import_roots_do_not_include_filesystem_root(self, tmp_path):
        first_bundle = tmp_path / "first"
        first_bundle.mkdir()
        (first_bundle / "SKILL.md").write_text(
            "---\nname: first\ndescription: First bundle.\nentrypoint: looplet.py\n---\n# First\n",
            encoding="utf-8",
        )
        (first_bundle / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )
        marked_project = tmp_path / "marked"
        second_bundle = marked_project / "skill"
        second_bundle.mkdir(parents=True)
        (marked_project / "pyproject.toml").write_text(
            "[project]\nname = 'marked'\n",
            encoding="utf-8",
        )
        (second_bundle / "SKILL.md").write_text(
            "---\nname: second\ndescription: Second bundle.\nentrypoint: looplet.py\n---\n# Second\n",
            encoding="utf-8",
        )
        (second_bundle / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        first = load_skill_bundle(first_bundle)
        second = load_skill_bundle(second_bundle)

        assert Path("/").resolve() not in first.import_roots
        assert second.skill.name == "second"
        assert os.path is not None

    def test_bundle_local_stdlib_name_does_not_replace_loaded_stdlib(self, tmp_path):
        bundle_root = tmp_path / "stdlib_shadow_bundle"
        bundle_root.mkdir()
        (bundle_root / "os.py").write_text("VALUE = 'local os'\n", encoding="utf-8")
        (bundle_root / "SKILL.md").write_text(
            "---\nname: stdlib-shadow\ndescription: Stdlib shadow bundle.\nentrypoint: looplet.py\n---\n# Shadow\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "import os\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def os_path_name():\n"
            "    return os.path.__name__\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        loaded = load_skill_bundle(bundle_root)

        assert loaded.module.os_path_name() == os.path.__name__
        assert os.path is not None

    def test_bundle_local_unloaded_stdlib_name_does_not_shadow_stdlib(self, tmp_path):
        sys.modules.pop("mailbox", None)
        bundle_root = tmp_path / "unloaded_stdlib_shadow_bundle"
        bundle_root.mkdir()
        local_mailbox = bundle_root / "mailbox.py"
        local_mailbox.write_text("VALUE = 'local mailbox'\n", encoding="utf-8")
        (bundle_root / "SKILL.md").write_text(
            "---\nname: unloaded-stdlib-shadow\ndescription: Unloaded stdlib shadow bundle.\nentrypoint: looplet.py\n---\n# Shadow\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "import mailbox\n"
            "from pathlib import Path\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def mailbox_path():\n"
            "    return Path(mailbox.__file__).resolve()\n"
            "def mailbox_value():\n"
            "    return getattr(mailbox, 'VALUE', None)\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        loaded = load_skill_bundle(bundle_root)

        assert loaded.module.mailbox_path() != local_mailbox.resolve()
        assert loaded.module.mailbox_value() is None
        mailbox_file = sys.modules["mailbox"].__file__
        assert mailbox_file is not None
        assert Path(mailbox_file).resolve() != local_mailbox.resolve()

    def test_validates_coder_bundle(self, tmp_path):
        result = validate_skill_bundle(
            CODER_BUNDLE,
            SkillRuntime(workspace=tmp_path, max_steps=8),
        )

        assert result.ok, result.errors
        assert result.skill_name == "coder"
        assert result.errors == []

    def test_validation_warns_when_bundle_ignores_runtime_max_steps(self, tmp_path):
        bundle_root = tmp_path / "fixed_budget_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: fixed-budget\ndescription: Fixed budget bundle.\nentrypoint: looplet.py\n---\n# Fixed\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=2),\n"
            "        state=DefaultState(max_steps=2),\n"
            "    )\n",
            encoding="utf-8",
        )

        result = validate_skill_bundle(
            bundle_root,
            SkillRuntime(workspace=tmp_path / "workspace", max_steps=8),
        )

        assert result.ok, result.errors
        assert "config.max_steps differs from runtime.max_steps (2 != 8)" in result.warnings

    def test_cli_reports_generic_bundle_runtime_step_warning(self, tmp_path, capsys):
        bundle_root = tmp_path / "fixed_budget_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: fixed-budget\ndescription: Fixed budget bundle.\nentrypoint: looplet.py\n---\n# Fixed\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--max-steps",
                "8",
                "--scripted-response",
                '{"tool": "done", "args": {"answer": "ok"}, "reasoning": "r"}',
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert "warning: config.max_steps differs from runtime.max_steps (1 != 8)" in captured.err

    def test_coder_bundle_matches_coder_example_primitives(self, tmp_path):
        from examples.coder import agent as coder

        bundle = load_skill_bundle(CODER_BUNDLE)
        preset = bundle.build_preset(SkillRuntime(workspace=tmp_path, max_steps=8))
        reference_tools = coder.make_tools(str(tmp_path.resolve()), coder.FileCache(str(tmp_path)))

        assert isinstance(preset, AgentPreset)
        assert preset.tools.tool_names == reference_tools.tool_names
        assert preset.config.system_prompt == coder.SYSTEM_PROMPT
        assert preset.config.max_steps == 8
        assert preset.state.max_steps == 8
        assert any(isinstance(hook, coder.TestGuardHook) for hook in preset.hooks)
        assert any(isinstance(hook, coder.StaleFileHook) for hook in preset.hooks)
        assert any(isinstance(hook, coder.LinterHook) for hook in preset.hooks)

    def test_coder_bundle_runs_scripted_loop_like_example(self, tmp_path):
        from examples.coder import agent as coder

        bundle = load_skill_bundle(CODER_BUNDLE)
        llm = MockLLMBackend(responses=coder.scripted_responses())
        steps = list(
            run_skill_bundle(
                bundle,
                llm=llm,
                task="Create a tiny add function with tests",
                runtime=SkillRuntime(workspace=tmp_path, max_steps=8),
            )
        )

        assert [step.tool_call.tool for step in steps] == [
            "list_dir",
            "write_file",
            "write_file",
            "bash",
            "done",
        ]
        assert steps[-1].tool_result.error is None
        assert (tmp_path / "math_utils.py").read_text(encoding="utf-8").rstrip() == (
            "def add(left: int, right: int) -> int:\n    return left + right"
        )
        assert "test_add" in (tmp_path / "test_math_utils.py").read_text(encoding="utf-8")
        assert list((tmp_path / ".looplet" / "traces").glob("coder-*/trajectory.json"))

    def test_run_skill_bundle_can_disable_default_trace(self, tmp_path):
        from examples.coder import agent as coder

        bundle = load_skill_bundle(CODER_BUNDLE)
        llm = MockLLMBackend(responses=coder.scripted_responses())
        list(
            run_skill_bundle(
                bundle,
                llm=llm,
                task="Create a tiny add function with tests",
                runtime=SkillRuntime(workspace=tmp_path, max_steps=8),
                provenance=False,
            )
        )

        assert not (tmp_path / ".looplet").exists()

    def test_run_skill_bundle_reports_invalid_bundle_contract(self, tmp_path):
        bundle_root = tmp_path / "bad_direct_run_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad-direct\ndescription: Bad direct bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "def build(runtime):\n    return object()\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="build returned object, expected AgentPreset"):
            list(
                run_skill_bundle(
                    bundle_root,
                    llm=MockLLMBackend(responses=[]),
                    task="Task",
                    runtime=SkillRuntime(workspace=tmp_path),
                    provenance=False,
                )
            )

    def test_run_skill_bundle_runtime_tools_use_bundle_import_context(self, tmp_path):
        project = tmp_path / "project"
        bundle_root = project / "skill"
        bundle_root.mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            "[project]\nname = 'runtime-import'\n",
            encoding="utf-8",
        )
        (project / "runtime_helper.py").write_text(
            "VALUE = 'runtime-ok'\n",
            encoding="utf-8",
        )
        (bundle_root / "SKILL.md").write_text(
            "---\nname: runtime-import\ndescription: Runtime import bundle.\nentrypoint: looplet.py\n---\n# Runtime\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tool, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "@tool\n"
            "def local_value() -> dict:\n"
            "    import runtime_helper\n"
            "    return {'value': runtime_helper.VALUE}\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([local_value], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=2),\n"
            "        state=DefaultState(max_steps=2),\n"
            "    )\n",
            encoding="utf-8",
        )

        steps = list(
            run_skill_bundle(
                bundle_root,
                llm=MockLLMBackend(
                    responses=[
                        '{"tool": "local_value", "args": {}}',
                        '{"tool": "done", "args": {"summary": "ok"}}',
                    ]
                ),
                task="Task",
                runtime=SkillRuntime(workspace=tmp_path / "workspace", max_steps=2),
                provenance=False,
            )
        )

        assert steps[0].tool_result.error is None
        assert steps[0].tool_result.data == {"value": "runtime-ok"}

    def test_cli_coder_bundle_matches_original_coder_stdout_byte_for_byte(self, tmp_path, capsys):
        from examples.coder import agent as coder

        workspace = tmp_path / "workspace"
        task = "Create a tiny add function with tests"
        original_rc = coder.main(
            [task, "--workspace", str(workspace), "--max-steps", "8", "--scripted"]
        )
        original_out = capsys.readouterr().out
        shutil.rmtree(workspace)

        skill_rc = cli_main(
            [
                "run",
                str(CODER_BUNDLE),
                task,
                "--workspace",
                str(workspace),
                "--max-steps",
                "8",
                "--scripted",
            ]
        )
        skill_out = capsys.readouterr().out

        assert original_rc == 0
        assert skill_rc == 0
        assert skill_out == original_out
        assert list((workspace / ".looplet" / "traces").glob("coder-*/trajectory.json"))

    def test_cli_runs_coder_bundle_with_scripted_responses(self, tmp_path, capsys):
        from examples.coder import agent as coder

        args = [
            "run",
            str(CODER_BUNDLE),
            "Create a tiny add function with tests",
            "--workspace",
            str(tmp_path),
            "--max-steps",
            "8",
        ]
        for response in coder.scripted_responses():
            args.extend(["--scripted-response", response])

        rc = cli_main(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "looplet coder" in out
        assert "Done: Created math_utils.add with tests." in out
        assert (tmp_path / "math_utils.py").exists()

    def test_cli_runs_coder_bundle_with_bundle_scripted_mode(self, tmp_path, capsys):
        rc = cli_main(
            [
                "run",
                str(CODER_BUNDLE),
                "Create a tiny add function with tests",
                "--workspace",
                str(tmp_path),
                "--max-steps",
                "8",
                "--scripted",
            ]
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "✏️  #2 write: math_utils.py" in out
        assert "Done: Created math_utils.add with tests." in out
        assert (tmp_path / "math_utils.py").exists()

    def test_cli_reports_invalid_bundle_without_traceback(self, tmp_path, capsys):
        missing_bundle = tmp_path / "missing"

        rc = cli_main(
            [
                "run",
                str(missing_bundle),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "error: invalid bundle" in captured.err
        assert "Traceback" not in captured.err

    def test_cli_reports_scripted_provider_error_without_traceback(self, tmp_path, capsys):
        bundle_root = tmp_path / "bad_scripted_provider_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad-provider\ndescription: Bad provider bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            "    raise RuntimeError('boom from provider')\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert (
            "error: bundle 'bad-provider' failed while loading scripted responses" in captured.err
        )
        assert "RuntimeError: boom from provider" in captured.err
        assert "Traceback" not in captured.err

    def test_cli_reports_invalid_bundle_before_scripted_provider_error(self, tmp_path, capsys):
        bundle_root = tmp_path / "bad_provider_bad_build_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad-mask\ndescription: Bad masking bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "def scripted_responses():\n"
            "    raise RuntimeError('provider blew up')\n"
            "def build(runtime):\n"
            "    return object()\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "error: invalid bundle" in captured.err
        assert "build returned object, expected AgentPreset" in captured.err
        assert "provider blew up" not in captured.err

    def test_cli_validates_bundle_before_running_scripted_provider_side_effect(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "bad_provider_side_effect_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad-sidefx\ndescription: Bad side-effect bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from pathlib import Path\n"
            "marker = Path(__file__).with_name('provider_ran.txt')\n"
            "def scripted_responses():\n"
            "    marker.write_text('ran')\n"
            "    return ['{}']\n"
            "def build(runtime):\n"
            "    return object()\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "build returned object, expected AgentPreset" in captured.err
        assert not (bundle_root / "provider_ran.txt").exists()

    def test_cli_validates_run_owned_provider_with_explicit_trace_dir(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "trace_sensitive_run_provider_bundle"
        trace_dir = tmp_path / "trace"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: trace-sensitive-provider\ndescription: Trace-sensitive provider bundle.\nentrypoint: looplet.py\n---\n# Trace\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            "    return ['{}']\n"
            "def build(runtime):\n"
            "    if runtime.output_dir is None:\n"
            "        return object()\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            "    print('trace_dir=' + str(kwargs['trace_dir']))\n"
            "    return 0\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--trace-dir",
                str(trace_dir),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == f"trace_dir={trace_dir}"

    def test_cli_reports_invalid_bundle_before_scripted_provider_contract_error(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "bad_provider_contract_bad_build_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad-contract-mask\ndescription: Bad contract masking bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "def scripted_responses():\n    return []\ndef build(runtime):\n    return object()\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "error: invalid bundle" in captured.err
        assert "build returned object, expected AgentPreset" in captured.err
        assert "scripted_responses() returned no responses" not in captured.err

    def test_cli_reports_empty_scripted_provider_without_mock_fallback(self, tmp_path, capsys):
        bundle_root = tmp_path / "empty_scripted_provider_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: empty-provider\ndescription: Empty provider bundle.\nentrypoint: looplet.py\n---\n# Empty\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            "    return []\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "scripted_responses() returned no responses" in captured.err
        assert "mock response" not in captured.out

    def test_cli_reports_non_string_scripted_provider_response(self, tmp_path, capsys):
        bundle_root = tmp_path / "non_string_scripted_provider_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: non-string-provider\ndescription: Non-string provider bundle.\nentrypoint: looplet.py\n---\n# Non-string\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            "    return [{'tool': 'done', 'args': {'summary': 'ok'}}]\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "scripted_responses() item 1 must be str, got dict" in captured.err
        assert "__parse_error__" not in captured.out

    def test_cli_reports_empty_scripted_provider_response_item(self, tmp_path, capsys):
        bundle_root = tmp_path / "empty_scripted_provider_item_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: empty-provider-item\ndescription: Empty provider item bundle.\nentrypoint: looplet.py\n---\n# Empty\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            "    return ['   ']\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "scripted_responses() item 1 must not be empty" in captured.err
        assert "__parse_error__" not in captured.out

    def test_cli_reports_scripted_provider_returning_single_string(self, tmp_path, capsys):
        bundle_root = tmp_path / "string_scripted_provider_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: string-provider\ndescription: String provider bundle.\nentrypoint: looplet.py\n---\n# String\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            '    return \'{"tool": "done", "args": {"summary": "ok"}}\'\n'
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "scripted_responses() must return an iterable of response strings" in captured.err
        assert "__parse_error__" not in captured.out

    def test_cli_reports_scripted_provider_generator_error_without_traceback(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "generator_scripted_provider_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: generator-provider\ndescription: Generator provider bundle.\nentrypoint: looplet.py\n---\n# Generator\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            '    yield \'{"tool": "done", "args": {"summary": "ok"}}\'\n'
            "    raise RuntimeError('generator blew up')\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "failed while loading scripted responses" in captured.err
        assert "RuntimeError: generator blew up" in captured.err
        assert "Traceback" not in captured.err

    def test_cli_validates_bundle_before_delegating_run(self, tmp_path, capsys):
        bundle_root = tmp_path / "bad_run_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad\ndescription: Bad run bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "def build(runtime):\n"
            "    return object()\n"
            "def run(**kwargs):\n"
            "    print('ran invalid bundle')\n"
            "    return 0\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "build returned object, expected AgentPreset" in captured.err
        assert "ran invalid bundle" not in captured.out

    def test_cli_validates_generic_bundle_before_backend_setup(self, tmp_path, capsys):
        bundle_root = tmp_path / "bad_generic_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: bad-generic\ndescription: Bad generic bundle.\nentrypoint: looplet.py\n---\n# Bad\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "def build(runtime):\n    return object()\n",
            encoding="utf-8",
        )

        class ExplodingBackend:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("backend should not be touched")

        with patch("looplet.backends.OpenAIBackend", ExplodingBackend):
            rc = cli_main(
                [
                    "run",
                    str(bundle_root),
                    "Task",
                    "--workspace",
                    str(tmp_path / "workspace"),
                ]
            )

        captured = capsys.readouterr()
        assert rc == 1
        assert "build returned object, expected AgentPreset" in captured.err
        assert "backend should not be touched" not in captured.err

    def test_cli_validates_loaded_bundle_without_reimporting_entrypoint(self, tmp_path, capsys):
        bundle_root = tmp_path / "side_effect_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: sidefx\ndescription: Side-effect bundle.\nentrypoint: looplet.py\n---\n# Sidefx\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from pathlib import Path\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "marker = Path(__file__).with_name('imports.txt')\n"
            "count = int(marker.read_text() or '0') if marker.exists() else 0\n"
            "marker.write_text(str(count + 1))\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            "    print('import_count=' + marker.read_text())\n"
            "    return 0\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "import_count=1"
        assert (bundle_root / "imports.txt").read_text(encoding="utf-8") == "1"

    def test_cli_reports_run_owned_bundle_runtime_error_without_traceback(self, tmp_path, capsys):
        bundle_root = tmp_path / "runtime_error_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: runtime-error\ndescription: Runtime error bundle.\nentrypoint: looplet.py\n---\n# Runtime\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            "    raise RuntimeError('boom from run')\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "error: bundle 'runtime-error' failed while running" in captured.err
        assert "RuntimeError: boom from run" in captured.err
        assert "Traceback" not in captured.err

    def test_cli_reports_run_owned_bundle_validation_warnings(self, tmp_path, capsys):
        bundle_root = tmp_path / "run_warning_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: run-warning\ndescription: Run warning bundle.\nentrypoint: looplet.py\n---\n# Warning\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            "    print('ran')\n"
            "    return 0\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--max-steps",
                "9",
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "ran"
        assert "warning: config.max_steps differs from runtime.max_steps (1 != 9)" in captured.err

    @pytest.mark.parametrize(
        "return_expr, type_name",
        [("None", "NoneType"), ("'oops'", "str"), ("True", "bool")],
    )
    def test_cli_reports_invalid_run_owned_bundle_status(
        self,
        tmp_path,
        capsys,
        return_expr,
        type_name,
    ):
        bundle_root = tmp_path / "invalid_status_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: invalid-status\ndescription: Invalid status bundle.\nentrypoint: looplet.py\n---\n# Invalid\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            f"    return {return_expr}\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "error: bundle 'invalid-status' returned invalid status" in captured.err
        assert f"expected int exit code, got {type_name}" in captured.err

    def test_cli_passes_scripted_provider_responses_to_run_owned_bundle(self, tmp_path, capsys):
        bundle_root = tmp_path / "scripted_run_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: scripted-run\ndescription: Scripted run bundle.\nentrypoint: looplet.py\n---\n# Scripted\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def scripted_responses():\n"
            "    return ['provided']\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            "    print(kwargs['scripted_responses'])\n"
            "    return 0\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "['provided']"

    def test_cli_treats_explicit_responses_as_scripted_for_run_owned_bundle(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "explicit_response_run_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: explicit-response\ndescription: Explicit response bundle.\nentrypoint: looplet.py\n---\n# Explicit\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n"
            "def run(**kwargs):\n"
            "    print(kwargs['scripted'])\n"
            "    print(kwargs['scripted_responses'])\n"
            "    return 0\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--max-steps",
                "1",
                "--scripted-response",
                '{"tool": "done", "args": {"summary": "ok"}}',
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.splitlines() == [
            "True",
            '[\'{"tool": "done", "args": {"summary": "ok"}}\']',
        ]

    def test_cli_reports_empty_explicit_scripted_response(self, tmp_path, capsys):
        bundle_root = tmp_path / "empty_explicit_response_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: empty-explicit\ndescription: Empty explicit response bundle.\nentrypoint: looplet.py\n---\n# Empty\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted-response",
                "",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "--scripted-response 1 must not be empty" in captured.err
        assert "__parse_error__" not in captured.out

    def test_cli_runs_generic_bundle_without_rebuilding_after_validation(self, tmp_path, capsys):
        bundle_root = tmp_path / "build_side_effect_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: build-sidefx\ndescription: Build side-effect bundle.\nentrypoint: looplet.py\n---\n# Build\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from pathlib import Path\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "marker = Path(__file__).with_name('builds.txt')\n"
            "def scripted_responses():\n"
            "    import json\n"
            "    return [json.dumps({'tool': 'done', 'args': {'summary': 'ok'}, 'reasoning': 'finish'})]\n"
            "def build(runtime):\n"
            "    count = int(marker.read_text() or '0') if marker.exists() else 0\n"
            "    marker.write_text(str(count + 1))\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        capsys.readouterr()
        assert rc == 0
        assert (bundle_root / "builds.txt").read_text(encoding="utf-8") == "1"

    def test_cli_runs_native_generic_bundle_without_rebuilding_after_validation(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "native_build_side_effect_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: native-build-sidefx\ndescription: Native build side-effect bundle.\nentrypoint: looplet.py\n---\n# Native\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from pathlib import Path\n"
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "marker = Path(__file__).with_name('builds.txt')\n"
            "def build(runtime):\n"
            "    count = int(marker.read_text() or '0') if marker.exists() else 0\n"
            "    marker.write_text(str(count + 1))\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(\n"
            "            max_steps=1,\n"
            "            use_native_tools=bool(runtime.option('use_native_tools', False)),\n"
            "        ),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        class NativeBackend:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, *args, **kwargs):
                raise AssertionError("JSON-text path should not run")

            def generate_with_tools(self, prompt, *, tools, **kwargs):
                if any(tool.get("name") == "test_probe" for tool in tools):
                    return [{"type": "tool_use", "id": "probe", "name": "test_probe", "input": {}}]
                return [
                    {"type": "tool_use", "id": "done-1", "name": "done", "input": {"summary": "ok"}}
                ]

        with patch("looplet.backends.OpenAIBackend", NativeBackend):
            rc = cli_main(
                [
                    "run",
                    str(bundle_root),
                    "Task",
                    "--workspace",
                    str(tmp_path / "workspace"),
                    "--max-steps",
                    "1",
                    "--no-trace",
                ]
            )

        captured = capsys.readouterr()
        assert rc == 0
        assert "Tool protocol: native" in captured.out
        assert (bundle_root / "builds.txt").read_text(encoding="utf-8") == "1"

    def test_cli_does_not_force_native_when_bundle_only_reads_native_option(
        self,
        tmp_path,
        capsys,
    ):
        bundle_root = tmp_path / "native_option_reader_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: native-option-reader\ndescription: Native option reader bundle.\nentrypoint: looplet.py\n---\n# Native\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "def build(runtime):\n"
            "    runtime.option('use_native_tools', False)\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[],\n"
            "        config=LoopConfig(max_steps=1, use_native_tools=False),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        class NativeBackend:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, *args, **kwargs):
                return '{"tool": "done", "args": {"summary": "json ok"}}'

            def generate_with_tools(self, prompt, *, tools, **kwargs):
                if any(tool.get("name") == "test_probe" for tool in tools):
                    return [{"type": "tool_use", "id": "probe", "name": "test_probe", "input": {}}]
                raise AssertionError("Native path should not be forced")

        with patch("looplet.backends.OpenAIBackend", NativeBackend):
            rc = cli_main(
                [
                    "run",
                    str(bundle_root),
                    "Task",
                    "--workspace",
                    str(tmp_path / "workspace"),
                    "--max-steps",
                    "1",
                    "--no-trace",
                ]
            )

        captured = capsys.readouterr()
        assert rc == 0
        assert "Tool protocol: json-text" in captured.out
        assert "Native path should not be forced" not in captured.err

    def test_cli_reports_generic_bundle_runtime_error_without_traceback(self, tmp_path, capsys):
        bundle_root = tmp_path / "generic_runtime_error_bundle"
        bundle_root.mkdir()
        (bundle_root / "SKILL.md").write_text(
            "---\nname: generic-runtime-error\ndescription: Generic runtime error bundle.\nentrypoint: looplet.py\n---\n# Runtime\n",
            encoding="utf-8",
        )
        (bundle_root / "looplet.py").write_text(
            "from looplet import DefaultState, LoopConfig, LoopHook, tools_from\n"
            "from looplet.presets import AgentPreset\n"
            "class BoomHook(LoopHook):\n"
            "    def pre_prompt(self, state, session_log, context, step_num):\n"
            "        raise RuntimeError('boom from generic hook')\n"
            "def scripted_responses():\n"
            "    return ['{}']\n"
            "def build(runtime):\n"
            "    return AgentPreset(\n"
            "        tools=tools_from([], include_done=True),\n"
            "        hooks=[BoomHook()],\n"
            "        config=LoopConfig(max_steps=1),\n"
            "        state=DefaultState(max_steps=1),\n"
            "    )\n",
            encoding="utf-8",
        )

        rc = cli_main(
            [
                "run",
                str(bundle_root),
                "Task",
                "--workspace",
                str(tmp_path / "workspace"),
                "--scripted",
                "--no-trace",
            ]
        )

        captured = capsys.readouterr()
        assert rc == 1
        assert "error: bundle 'generic-runtime-error' failed while running" in captured.err
        assert "RuntimeError: boom from generic hook" in captured.err
        assert "Traceback" not in captured.err

    def test_cli_can_write_trace_for_coder_bundle(self, tmp_path, capsys):
        workspace = tmp_path / "workspace"
        trace_dir = tmp_path / "trace"

        rc = cli_main(
            [
                "run",
                str(CODER_BUNDLE),
                "Create a tiny add function with tests",
                "--workspace",
                str(workspace),
                "--trace-dir",
                str(trace_dir),
                "--max-steps",
                "8",
                "--scripted",
            ]
        )

        assert rc == 0
        capsys.readouterr()
        assert (trace_dir / "trajectory.json").exists()
        assert (trace_dir / "manifest.jsonl").exists()

    def test_cli_can_disable_trace_for_coder_bundle(self, tmp_path, capsys):
        workspace = tmp_path / "workspace"
        trace_dir = tmp_path / "trace"

        rc = cli_main(
            [
                "run",
                str(CODER_BUNDLE),
                "Create a tiny add function with tests",
                "--workspace",
                str(workspace),
                "--trace-dir",
                str(trace_dir),
                "--max-steps",
                "8",
                "--scripted",
                "--no-trace",
            ]
        )

        assert rc == 0
        capsys.readouterr()
        assert not trace_dir.exists()

    def test_coder_bundle_no_trace_preserves_non_scripted_call_count(self, tmp_path, capsys):
        from examples.coder import agent as coder

        class FakeOpenAIBackend:
            def __init__(self, *args, **kwargs):
                self._responses = list(coder.scripted_responses())
                self.calls = 0

            def generate(
                self,
                prompt: str,
                *,
                max_tokens: int = 2000,
                system_prompt: str = "",
                temperature: float = 0.2,
            ) -> str:
                index = min(self.calls, len(self._responses) - 1)
                self.calls += 1
                return self._responses[index]

        bundle = load_skill_bundle(CODER_BUNDLE)
        with patch.object(bundle.module, "OpenAIBackend", FakeOpenAIBackend):
            rc = bundle.module.run(
                task="Create a tiny add function with tests",
                workspace=tmp_path,
                max_steps=8,
                scripted=False,
                scripted_responses=[],
                require_tests=True,
                trace_dir=None,
                provenance=False,
            )

        assert rc == 0
        out = capsys.readouterr().out
        assert "Steps: 5 | LLM calls: 5 (0 tool-internal)" in out
        assert not (tmp_path / ".looplet").exists()
