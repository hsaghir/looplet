"""Release metadata and duplicated public docs must not drift."""

from __future__ import annotations

import json
import tomllib
from datetime import date
from pathlib import Path

import looplet

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_is_consistent_and_changelog_has_release() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    major, minor, _patch = (int(part) for part in version.split("."))
    readme = (ROOT / "README.md").read_text()

    assert version == looplet.__version__
    assert f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text()
    assert f"release is `{version}`" in readme
    assert f"looplet>={major}.{minor},<{major}.{minor + 1}" in readme


def test_lockfile_and_latest_release_match_project_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    lockfile = tomllib.loads((ROOT / "uv.lock").read_text())
    changelog = (ROOT / "CHANGELOG.md").read_text()

    looplet_packages = [package for package in lockfile["package"] if package["name"] == "looplet"]
    assert len(looplet_packages) == 1
    assert looplet_packages[0]["version"] == version

    release_headings = [line for line in changelog.splitlines() if line.startswith("## [0")]
    assert release_headings
    heading_prefix = f"## [{version}] - "
    assert release_headings[0].startswith(heading_prefix)
    release_date = date.fromisoformat(release_headings[0].removeprefix(heading_prefix))
    assert release_date <= date.today()


def test_site_changelog_matches_canonical_history() -> None:
    canonical = (ROOT / "CHANGELOG.md").read_text()
    expected_site = canonical.replace("](docs/provenance.md)", "](provenance.md)")

    assert (ROOT / "docs" / "changelog.md").read_text() == expected_site


def test_site_roadmap_matches_canonical_direction() -> None:
    canonical = (ROOT / "ROADMAP.md").read_text()
    expected_site = canonical.replace(
        "](docs/regression-demo.md)",
        "](regression-demo.md)",
    )

    assert (ROOT / "docs" / "roadmap.md").read_text() == expected_site


def test_site_contributing_guide_matches_canonical_guide() -> None:
    assert (ROOT / "docs" / "contributing.md").read_text() == (ROOT / "CONTRIBUTING.md").read_text()


def test_portable_coder_is_in_both_distribution_formats() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel_files = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    sdist_files = project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    source = "examples/coder_portable.cartridge"
    assert wheel_files[source] == "looplet/_bundled/coder_portable.cartridge"
    assert source in sdist_files
    assert (
        wheel_files["tests/fixtures/coder_skill_bundle/__init__.py"]
        == "tests/fixtures/coder_skill_bundle/__init__.py"
    )


def test_regression_proof_is_in_both_distribution_formats() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    wheel_files = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    sdist_files = project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    assert scripts["looplet-proof"] == "looplet.examples.regression_demo:main"
    assert (
        wheel_files["examples/regression_demo/report_agent.cartridge"]
        == "looplet/_bundled/regression_demo/report_agent.cartridge"
    )
    assert (
        wheel_files["examples/regression_demo/before_publish_report.py"]
        == "looplet/_bundled/regression_demo/before_publish_report.py"
    )
    assert wheel_files["tests/conformance/fixtures"] == "looplet/_bundled/conformance"
    assert "examples/regression_demo" in sdist_files
    assert "SPEC.md" in sdist_files
    assert "cartridge.schema.json" in sdist_files


def test_private_loop_migration_recipe_is_available_to_sdist_tests() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    sdist_files = project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    migrate = (ROOT / "docs" / "migrate.md").read_text()

    assert "examples/private_loop_migration" in sdist_files
    assert "source checkout" in migrate


def test_manifest_schema_matches_supported_cartridge_version() -> None:
    from looplet.cartridge._layout import SCHEMA_VERSION

    schema = json.loads((ROOT / "cartridge.schema.json").read_text())
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_tag_publish_creates_a_github_release_from_built_artifacts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text()

    assert "name: create GitHub release" in workflow
    assert "needs: [build, publish]" in workflow
    assert 'gh release create "$GITHUB_REF_NAME" dist/*' in workflow
