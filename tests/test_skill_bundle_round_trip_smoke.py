from __future__ import annotations

import importlib.util
import subprocess
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from looplet import SkillRuntime, load_skill_bundle
from looplet.__main__ import main as cli_main
from looplet.blueprints import (
    blueprint_from_bundle,
    blueprint_from_preset,
    claude_skill_compatibility,
    compare_blueprints,
    export_bundle_to_library_code,
    package_agent_factory_as_bundle,
    wrap_claude_skill_as_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
CODER_BUNDLE = ROOT / "tests" / "fixtures" / "coder_skill_bundle"

# Bundle tests load real bundles, exec spec-loaded modules, and run
# full blueprint comparisons. Under coverage instrumentation in CI the
# combined import + exec + compare pass repeatedly busts the default
# 30s timeout. Bump the per-test budget for the whole module.
pytestmark = pytest.mark.timeout(120)


def _import_python_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coder_bundle_exports_structural_blueprint(tmp_path):
    runtime = SkillRuntime(workspace=tmp_path / "workspace", max_steps=8)

    blueprint = blueprint_from_bundle(CODER_BUNDLE, runtime)

    assert blueprint.schema_version == "looplet.agent-blueprint.v1"
    assert blueprint.name == "coder"
    assert blueprint.source.kind == "bundle"
    assert blueprint.config["max_steps"] == 8
    assert {tool.name for tool in blueprint.tools} >= {
        "bash",
        "done",
        "read_file",
        "write_file",
    }
    assert [hook.kind for hook in blueprint.hooks][:3] == [
        "examples.coder.hooks:TestGuardHook",
        "examples.coder.hooks:FileCacheHook",
        "examples.coder.hooks:StaleFileHook",
    ]
    assert blueprint.fingerprint() == blueprint_from_bundle(CODER_BUNDLE, runtime).fingerprint()


def test_export_bundle_to_library_code_builds_equivalent_preset(tmp_path):
    runtime = SkillRuntime(workspace=tmp_path / "workspace", max_steps=8)
    exported = tmp_path / "coder_export.py"

    export_bundle_to_library_code(CODER_BUNDLE, exported, function_name="build_agent")
    module = _import_python_file(exported)

    original_preset = load_skill_bundle(CODER_BUNDLE).build_preset(runtime)
    exported_preset = module.build_agent(runtime)

    comparison = compare_blueprints(
        blueprint_from_preset(original_preset, name="coder"),
        blueprint_from_preset(exported_preset, name="coder"),
    )
    assert comparison.ok, comparison.differences


def test_export_bundle_to_library_code_documents_local_wrapper(tmp_path):
    exported = tmp_path / "coder_export.py"

    export_bundle_to_library_code(CODER_BUNDLE, exported, function_name="build_agent")
    text = exported.read_text(encoding="utf-8")

    assert "Generated local looplet library wrapper" in text
    assert "absolute path" in text


def test_package_agent_factory_as_bundle_builds_equivalent_preset(tmp_path):
    runtime = SkillRuntime(workspace=tmp_path / "workspace", max_steps=8)
    packaged = tmp_path / "coder-packaged"

    package_agent_factory_as_bundle(
        "tests.fixtures.coder_skill_bundle.looplet:build",
        packaged,
        name="coder-packaged",
        description="Packaged copy of the coder example.",
        tags=["coding", "round-trip"],
    )

    original = blueprint_from_bundle(CODER_BUNDLE, runtime)
    repackaged = blueprint_from_bundle(packaged, runtime)

    comparison = compare_blueprints(original, repackaged, ignore_metadata=True)
    assert comparison.ok, comparison.differences


def test_wrap_instruction_only_claude_skill_as_runnable_bundle(tmp_path):
    claude_skill = tmp_path / "claude-skills" / "pdf"
    claude_skill.mkdir(parents=True)
    (claude_skill / "SKILL.md").write_text(
        """---
name: pdf
description: Work with PDF files.
tags: [documents]
---

# PDF Skill

Use careful extraction and preserve table structure.
""",
        encoding="utf-8",
    )

    report = claude_skill_compatibility(claude_skill)
    assert report.level == "instruction-only"
    assert report.can_wrap
    assert report.can_run_exactly

    wrapped = wrap_claude_skill_as_bundle(claude_skill, tmp_path / "wrapped-pdf")
    bundle = load_skill_bundle(wrapped)
    preset = bundle.build_preset(SkillRuntime(workspace=tmp_path / "workspace", max_steps=3))

    assert bundle.skill.name == "pdf"
    assert "preserve table structure" in preset.config.system_prompt
    assert preset.config.max_steps == 3
    assert preset.tools.tool_names == ["done"]


def test_claude_skill_with_scripts_reports_adapter_gap(tmp_path):
    claude_skill = tmp_path / "claude-skills" / "chart"
    scripts = claude_skill / "scripts"
    scripts.mkdir(parents=True)
    (claude_skill / "SKILL.md").write_text(
        """---
name: chart
description: Render charts.
---

# Chart Skill

Run the helper script when chart data must be normalized.
""",
        encoding="utf-8",
    )
    (scripts / "normalize.py").write_text("print('normalize')\n", encoding="utf-8")

    report = claude_skill_compatibility(claude_skill)

    assert report.level == "scripts-present"
    assert report.can_wrap
    assert not report.can_run_exactly
    assert "scripts require an explicit looplet tool adapter" in report.warnings


def test_cli_exports_packages_and_wraps_bundles(tmp_path, capsys):
    exported = tmp_path / "coder_export.py"
    packaged = tmp_path / "coder-packaged"

    export_rc = cli_main(
        [
            "export-code",
            str(CODER_BUNDLE),
            str(exported),
            "--function-name",
            "build_agent",
        ]
    )
    assert export_rc == 0
    assert exported.exists()
    assert "exported" in capsys.readouterr().out

    package_rc = cli_main(
        [
            "package",
            "tests.fixtures.coder_skill_bundle.looplet:build",
            str(packaged),
            "--name",
            "coder-packaged",
            "--description",
            "Packaged copy of the coder example.",
            "--tag",
            "coding",
        ]
    )
    assert package_rc == 0
    assert (packaged / "SKILL.md").exists()
    assert "packaged" in capsys.readouterr().out

    blueprint_rc = cli_main(["blueprint", str(packaged), "--max-steps", "8"])
    assert blueprint_rc == 0
    blueprint_out = capsys.readouterr().out
    assert '"name": "coder-packaged"' in blueprint_out
    assert '"bash"' in blueprint_out

    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: pdf
description: Work with PDF files.
---

# PDF Skill

Preserve table structure.
""",
        encoding="utf-8",
    )
    wrapped = tmp_path / "wrapped-pdf"

    wrap_rc = cli_main(["wrap-claude-skill", str(claude_skill), str(wrapped)])

    assert wrap_rc == 0
    assert (wrapped / "looplet.py").exists()
    assert "instruction-only" in capsys.readouterr().out


def test_export_bundle_to_library_code_handles_triple_quotes_in_skill_body(tmp_path):
    bundle = tmp_path / "quote-skill"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        '''---
name: quote-skill
description: Skill with awkward markdown.
entrypoint: looplet.py
---

# Quote Skill

The docs contain """triple quotes""" and should still export.
''',
        encoding="utf-8",
    )
    (bundle / "looplet.py").write_text(
        """from looplet import minimal_preset


def build(runtime):
    return minimal_preset(max_steps=runtime.max_steps, system_prompt='quote skill')
""",
        encoding="utf-8",
    )
    exported = tmp_path / "quote_export.py"

    export_bundle_to_library_code(bundle, exported)
    module = _import_python_file(exported)

    assert module.BLUEPRINT.name == "quote-skill"
    assert module.build(SkillRuntime(max_steps=2)).config.max_steps == 2


def test_export_bundle_to_library_code_rejects_keyword_function_name(tmp_path):
    with pytest.raises(ValueError, match="function_name must be a valid Python function name"):
        export_bundle_to_library_code(
            CODER_BUNDLE, tmp_path / "bad_export.py", function_name="class"
        )


def test_package_agent_factory_rejects_missing_factory(tmp_path):
    with pytest.raises(ValueError, match="could not import factory"):
        package_agent_factory_as_bundle(
            "missing.module:build",
            tmp_path / "bad-package",
            name="bad",
            description="Bad package.",
        )


def test_package_agent_factory_rejects_entrypoint_outside_bundle(tmp_path):
    with pytest.raises(ValueError, match="entrypoint must stay inside bundle"):
        package_agent_factory_as_bundle(
            "tests.fixtures.coder_skill_bundle.looplet:build",
            tmp_path / "bad-entrypoint",
            name="bad-entrypoint",
            description="Bad entrypoint.",
            entrypoint="../looplet.py",
        )


def test_wrap_claude_skill_rejects_nested_output_directory(tmp_path):
    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: nested
description: Nested output should be rejected.
---

# Nested
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the source skill directory"):
        wrap_claude_skill_as_bundle(claude_skill, claude_skill / "wrapped")


def test_wrap_claude_skill_rejects_same_output_directory(tmp_path):
    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: same-dir
description: Same output should be rejected.
owner: original
---

# Same Dir
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="out_dir must be different"):
        wrap_claude_skill_as_bundle(claude_skill, claude_skill)
    assert "owner: original" in (claude_skill / "SKILL.md").read_text(encoding="utf-8")


def test_wrap_claude_skill_rejects_existing_output_without_deleting(tmp_path):
    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: existing-target
description: Existing output should be preserved.
---

# Existing Target
""",
        encoding="utf-8",
    )
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("important", encoding="utf-8")

    with pytest.raises(ValueError, match="out_dir already exists"):
        wrap_claude_skill_as_bundle(claude_skill, target)

    assert marker.read_text(encoding="utf-8") == "important"


def test_wrap_claude_skill_cleans_temp_output_when_copy_fails(tmp_path):
    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: copy-fail
description: Failed copies should not leave targets.
---

# Copy Fail
""",
        encoding="utf-8",
    )
    target = tmp_path / "target"

    with patch("looplet.blueprints.shutil.copytree", side_effect=RuntimeError("copy failed")):
        with pytest.raises(RuntimeError, match="copy failed"):
            wrap_claude_skill_as_bundle(claude_skill, target)

    assert not target.exists()
    assert not list(tmp_path.glob(".target.tmp-*"))


def test_wrap_claude_skill_cleans_temp_output_when_render_fails(tmp_path):
    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: render-fail
description: Failed rendering should not leave targets.
---

# Render Fail
""",
        encoding="utf-8",
    )
    target = tmp_path / "target"

    with patch(
        "looplet.blueprints._render_skill_markdown",
        side_effect=RuntimeError("render failed"),
    ):
        with pytest.raises(RuntimeError, match="render failed"):
            wrap_claude_skill_as_bundle(claude_skill, target)

    assert not target.exists()
    assert not list(tmp_path.glob(".target.tmp-*"))


def test_wrap_claude_skill_rejects_file_parent_for_output(tmp_path):
    claude_skill = tmp_path / "claude-skill"
    claude_skill.mkdir()
    (claude_skill / "SKILL.md").write_text(
        """---
name: file-parent
description: File parents should be rejected clearly.
---

# File Parent
""",
        encoding="utf-8",
    )
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file", encoding="utf-8")

    with pytest.raises(ValueError, match="out_dir parent must be a directory"):
        wrap_claude_skill_as_bundle(claude_skill, parent_file / "target")


def test_wrap_looplet_bundle_preserves_existing_entrypoint(tmp_path):
    source = tmp_path / "source-bundle"
    source.mkdir()
    (source / "SKILL.md").write_text(
        """---
name: existing
description: Existing looplet bundle.
entrypoint: custom.py
---

# Existing
""",
        encoding="utf-8",
    )
    (source / "custom.py").write_text(
        """from looplet import minimal_preset


def build(runtime):
    return minimal_preset(max_steps=runtime.max_steps, system_prompt='custom bundle')
""",
        encoding="utf-8",
    )

    wrapped = wrap_claude_skill_as_bundle(source, tmp_path / "wrapped-bundle")
    bundle = load_skill_bundle(wrapped)
    preset = bundle.build_preset(SkillRuntime(max_steps=2))

    assert bundle.skill.metadata["entrypoint"] == "custom.py"
    assert preset.config.system_prompt == "custom bundle"
    assert not (wrapped / "looplet.py").exists()
