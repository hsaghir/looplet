"""Inlined v1 ``dep_doctor`` agent code - package registry, license
compat data, and the 5 tool functions previously hosted at
``examples/dep_doctor/agent.py``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from looplet.types import ToolContext

PACKAGE_DB = {
    # Python packages
    "requests": {
        "latest_version": "2.32.3",
        "last_release": "2024-05-29",
        "maintainers": 3,
        "license": "Apache-2.0",
        "weekly_downloads": 45_000_000,
        "cves": [],
        "status": "healthy",
        "description": "HTTP library for Python",
    },
    "flask": {
        "latest_version": "3.1.0",
        "last_release": "2024-11-15",
        "maintainers": 4,
        "license": "BSD-3-Clause",
        "weekly_downloads": 12_000_000,
        "cves": [],
        "status": "healthy",
        "description": "Lightweight WSGI web framework",
    },
    "django": {
        "latest_version": "5.2",
        "last_release": "2025-04-01",
        "maintainers": 15,
        "license": "BSD-3-Clause",
        "weekly_downloads": 8_000_000,
        "cves": [{"id": "CVE-2025-1234", "severity": "MEDIUM", "fixed_in": "5.1.5"}],
        "status": "healthy",
        "description": "High-level Python web framework",
    },
    "pyyaml": {
        "latest_version": "6.0.2",
        "last_release": "2024-08-06",
        "maintainers": 1,
        "license": "MIT",
        "weekly_downloads": 35_000_000,
        "cves": [],
        "status": "warning",
        "description": "YAML parser - single maintainer risk",
    },
    "cryptography": {
        "latest_version": "44.0.0",
        "last_release": "2025-01-15",
        "maintainers": 5,
        "license": "Apache-2.0 OR BSD-3-Clause",
        "weekly_downloads": 50_000_000,
        "cves": [{"id": "CVE-2024-9876", "severity": "HIGH", "fixed_in": "43.0.1"}],
        "status": "healthy",
        "description": "Cryptographic recipes and primitives",
    },
    "urllib3": {
        "latest_version": "2.3.0",
        "last_release": "2025-02-01",
        "maintainers": 2,
        "license": "MIT",
        "weekly_downloads": 60_000_000,
        "cves": [],
        "status": "healthy",
        "description": "HTTP client library",
    },
    "pillow": {
        "latest_version": "11.1.0",
        "last_release": "2025-01-02",
        "maintainers": 4,
        "license": "MIT-CMU",
        "weekly_downloads": 15_000_000,
        "cves": [],
        "status": "healthy",
        "description": "Python Imaging Library fork",
    },
    "setuptools": {
        "latest_version": "75.8.0",
        "last_release": "2025-01-20",
        "maintainers": 3,
        "license": "MIT",
        "weekly_downloads": 40_000_000,
        "cves": [],
        "status": "healthy",
        "description": "Build system for Python packages",
    },
    "abandoned-lib": {
        "latest_version": "0.3.1",
        "last_release": "2021-06-15",
        "maintainers": 1,
        "license": "GPL-3.0",
        "weekly_downloads": 500,
        "cves": [{"id": "CVE-2023-5555", "severity": "CRITICAL", "fixed_in": "none"}],
        "status": "abandoned",
        "description": "Abandoned library with unpatched CVE",
    },
    "numpy": {
        "latest_version": "2.2.0",
        "last_release": "2025-01-10",
        "maintainers": 20,
        "license": "BSD-3-Clause",
        "weekly_downloads": 55_000_000,
        "cves": [],
        "status": "healthy",
        "description": "Numerical computing library",
    },
    "pandas": {
        "latest_version": "2.2.3",
        "last_release": "2024-09-20",
        "maintainers": 12,
        "license": "BSD-3-Clause",
        "weekly_downloads": 30_000_000,
        "cves": [],
        "status": "healthy",
        "description": "Data analysis and manipulation",
    },
    # Node.js packages
    "express": {
        "latest_version": "4.21.0",
        "last_release": "2024-09-10",
        "maintainers": 5,
        "license": "MIT",
        "weekly_downloads": 30_000_000,
        "cves": [],
        "status": "healthy",
        "description": "Fast, minimalist web framework for Node.js",
    },
    "lodash": {
        "latest_version": "4.17.21",
        "last_release": "2021-02-20",
        "maintainers": 1,
        "license": "MIT",
        "weekly_downloads": 50_000_000,
        "cves": [],
        "status": "warning",
        "description": "Utility library - no updates since 2021",
    },
    "event-stream": {
        "latest_version": "4.0.1",
        "last_release": "2019-11-22",
        "maintainers": 1,
        "license": "MIT",
        "weekly_downloads": 2_000_000,
        "cves": [{"id": "CVE-2018-16487", "severity": "CRITICAL", "fixed_in": "4.0.0"}],
        "status": "compromised",
        "description": "Known supply chain attack target",
    },
}

LICENSE_COMPATIBILITY = {
    "MIT": {"compatible_with": ["MIT", "BSD-3-Clause", "Apache-2.0", "ISC", "BSD-2-Clause"]},
    "BSD-3-Clause": {"compatible_with": ["MIT", "BSD-3-Clause", "Apache-2.0", "ISC"]},
    "Apache-2.0": {"compatible_with": ["MIT", "BSD-3-Clause", "Apache-2.0"]},
    "GPL-3.0": {
        "compatible_with": ["GPL-3.0", "AGPL-3.0"],
        "note": "Copyleft - may require open-sourcing your code",
    },
    "LGPL-3.0": {"compatible_with": ["GPL-3.0", "LGPL-3.0", "AGPL-3.0"], "note": "Weak copyleft"},
}


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════


def detect_dep_files(*, project_dir: str) -> dict:
    """Scan project directory for dependency files."""
    p = Path(project_dir)
    found = []
    for name in [
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "package.json",
        "package-lock.json",
        "Cargo.toml",
        "go.mod",
        "go.sum",
        "Gemfile",
    ]:
        path = p / name
        if path.exists():
            found.append({"file": name, "size_bytes": path.stat().st_size})
    return {"project": p.name, "dep_files": found, "count": len(found)}


def parse_deps(*, file_path: str) -> dict:
    """Parse a dependency file and extract package names + versions."""
    p = Path(file_path)
    if not p.exists():
        return {"error": f"File not found: {file_path}"}

    content = p.read_text()
    name = p.name
    deps: list[dict] = []

    if name == "pyproject.toml":
        # Simple TOML parsing for dependencies
        in_deps = False
        for line in content.split("\n"):
            if "dependencies" in line and "=" in line and "[" not in line:
                continue
            if line.strip().startswith("[") and "dependencies" in line.lower():
                in_deps = True
                continue
            if line.strip().startswith("[") and in_deps:
                in_deps = False
                continue
            if in_deps:
                line = line.strip().strip(",").strip('"').strip("'")
                if not line or line.startswith("#"):
                    continue
                # Parse "package>=1.0" or "package"
                match = re.match(r"^([a-zA-Z0-9_-]+)", line)
                if match:
                    pkg = match.group(1).lower()
                    version_match = re.search(r'[><=!~]+\s*([0-9][^\s,"\']*)', line)
                    deps.append(
                        {
                            "name": pkg,
                            "constraint": version_match.group(0).strip() if version_match else "*",
                        }
                    )

    elif name == "requirements.txt":
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            match = re.match(r"^([a-zA-Z0-9_-]+)", line)
            if match:
                pkg = match.group(1).lower()
                version_match = re.search(r"[><=!~]+\s*([0-9][^\s]*)", line)
                deps.append(
                    {
                        "name": pkg,
                        "constraint": version_match.group(0) if version_match else "*",
                    }
                )

    elif name == "package.json":
        try:
            data = json.loads(content)
            for section in ["dependencies", "devDependencies"]:
                for pkg, ver in data.get(section, {}).items():
                    deps.append(
                        {"name": pkg, "constraint": ver, "dev": section == "devDependencies"}
                    )
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in package.json"}

    return {"file": name, "dependencies": deps, "count": len(deps)}


def check_package(*, package_name: str) -> dict:
    """Check a package's health: version, last release, CVEs, license, maintainers."""
    name = package_name.lower().strip()
    if name in PACKAGE_DB:
        pkg = dict(PACKAGE_DB[name])
        # Calculate staleness
        try:
            release_date = datetime.strptime(pkg["last_release"], "%Y-%m-%d")
            days_since = (datetime.now() - release_date).days
            pkg["days_since_release"] = days_since
            pkg["stale"] = days_since > 365
        except (ValueError, KeyError):
            pkg["days_since_release"] = None
            pkg["stale"] = False
        return pkg
    return {
        "name": name,
        "error": "Package not found in registry",
        "status": "unknown",
    }


def check_license_compat(*, project_license: str, dep_license: str) -> dict:
    """Check if a dependency's license is compatible with the project license."""
    proj = project_license.strip()
    dep = dep_license.strip()

    proj_info = LICENSE_COMPATIBILITY.get(proj, {})
    compatible = dep in proj_info.get("compatible_with", [proj])

    return {
        "project_license": proj,
        "dependency_license": dep,
        "compatible": compatible,
        "note": proj_info.get("note", "") if not compatible else "",
        "risk": "HIGH" if not compatible and "GPL" in dep else "LOW" if compatible else "MEDIUM",
    }


def find_alternatives(*, package_name: str, ctx: ToolContext) -> dict:
    """Suggest alternative packages for a risky dependency using LLM."""
    if ctx.llm is not None:
        try:
            response = ctx.llm.generate(
                f"Suggest 2-3 well-maintained alternatives to the Python/Node.js "
                f"package '{package_name}'. For each alternative, give the package "
                f"name and a one-line reason. Respond as a JSON array of objects "
                f"with 'name' and 'reason' fields. Only the JSON array, nothing else.",
                max_tokens=200,
            )
            ctx.warn(f"Used LLM to find alternatives for {package_name}")
            try:
                alternatives = json.loads(response.strip())
                if isinstance(alternatives, list):
                    return {"package": package_name, "alternatives": alternatives}
            except json.JSONDecodeError:
                pass
            return {"package": package_name, "alternatives_text": response.strip()}
        except Exception as e:
            return {"package": package_name, "error": str(e)}
    return {"package": package_name, "alternatives": []}


# ═══════════════════════════════════════════════════════════════════
# AGENT
