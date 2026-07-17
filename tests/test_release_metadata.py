"""Release metadata and duplicated public docs must not drift."""

from __future__ import annotations

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
