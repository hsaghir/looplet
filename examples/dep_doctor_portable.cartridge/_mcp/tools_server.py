"""Stdio MCP server for the dep_doctor_portable cartridge.

Serves all 7 audit tools that were in-process ``tools/*/execute.py``
bodies in the original ``dep_doctor`` cartridge - ``detect_dep_files``,
``parse_deps``, ``check_package``, ``check_license_compat``,
``find_alternatives``, ``think``, ``done`` - over the MCP stdio
transport. Moving them out of process is what makes the twin fully
portable: no Python tool body is required by the host.

The package registry (``PACKAGE_DB``) and license compatibility table
(``LICENSE_COMPATIBILITY``) are vendored here verbatim from the original
``dep_doctor_lib.py``.

``find_alternatives`` reaches the host model through the Model Gateway
(MGP): the loader exports ``LOOPLET_LLM_SOCKET`` and this server connects
to it lazily, so the tool returns real LLM-suggested alternatives - full
parity with the in-process original. When no gateway is present (or no
backend is bound yet) it degrades to empty alternatives, exactly the
fallback the original tool takes when ``ctx.llm is None``.

Relative paths (``detect_dep_files``/``parse_deps``) resolve against the
server's working directory (``os.getcwd()``), which the loader sets to
the host project root.

Standard-library only.
Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import os
import re
import socket
import sys
from datetime import datetime
from pathlib import Path


class _HostLLM:
    """Minimal stdlib-only client to the host Model Gateway (MGP).

    Connects to ``$LOOPLET_LLM_SOCKET`` (set by the loader) and forwards
    ``generate`` to the host's live LLM backend. ``generate`` raises if no
    gateway/backend is reachable, so callers degrade exactly like the
    in-process original's ``ctx.llm is None`` branch.
    """

    def __init__(self):
        self._sock = None
        self._buf = b""
        self._id = 0
        path = os.environ.get("LOOPLET_LLM_SOCKET")
        if not path or not hasattr(socket, "AF_UNIX"):
            return
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30.0)
            sock.connect(path)
            self._sock = sock
            self._rpc("llm/initialize", {})
        except OSError:
            self._sock = None

    def _readline(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                line, self._buf = self._buf, b""
                return line
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _rpc(self, method, params):
        self._id += 1
        rid = self._id
        self._sock.sendall(
            (json.dumps({"id": rid, "method": method, "params": params}) + "\n").encode("utf-8")
        )
        line = self._readline()
        if not line:
            raise OSError("model gateway closed the connection")
        msg = json.loads(line.decode("utf-8"))
        if msg.get("error"):
            raise RuntimeError(msg["error"].get("message", "model gateway error"))
        return msg.get("result") or {}

    def available(self):
        """True iff the gateway has a live LLM backend bound *right now*.

        Re-checks per call because the host binds the backend lazily at
        run time (``AgentPreset.run(llm)``), which may happen after this
        client connected. Maps to the original's ``ctx.llm is not None``
        guard: when False, callers take their no-LLM degradation branch
        instead of treating absence as an error.
        """
        if self._sock is None:
            return False
        try:
            return bool(self._rpc("llm/initialize", {}).get("ready"))
        except (OSError, RuntimeError):
            return False

    def generate(self, prompt, **kwargs):
        if self._sock is None:
            raise RuntimeError("no host LLM gateway is reachable")
        return str(self._rpc("llm/generate", {"prompt": prompt, "kwargs": kwargs}).get("text", ""))


_HOST_LLM = None
_HOST_LLM_TRIED = False


def _host_llm():
    """Return the host Model Gateway client only when a backend is bound.

    Returns ``None`` when there is no reachable gateway *or* no backend is
    currently bound - i.e. exactly the cases where the in-process original
    sees ``ctx.llm is None`` and degrades.
    """
    global _HOST_LLM, _HOST_LLM_TRIED
    if not _HOST_LLM_TRIED:
        _HOST_LLM_TRIED = True
        client = _HostLLM()
        if client._sock is not None:
            _HOST_LLM = client
    if _HOST_LLM is not None and _HOST_LLM.available():
        return _HOST_LLM
    return None


PACKAGE_DB = {
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


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return p


def detect_dep_files(project_dir):
    p = _resolve(project_dir)
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


def parse_deps(file_path):
    p = _resolve(file_path)
    if not p.exists():
        return {"error": f"File not found: {file_path}"}

    content = p.read_text()
    name = p.name
    deps = []

    if name == "pyproject.toml":
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


def check_package(package_name):
    name = package_name.lower().strip()
    if name in PACKAGE_DB:
        pkg = dict(PACKAGE_DB[name])
        try:
            release_date = datetime.strptime(pkg["last_release"], "%Y-%m-%d")
            days_since = (datetime.now() - release_date).days
            pkg["days_since_release"] = days_since
            pkg["stale"] = days_since > 365
        except (ValueError, KeyError):
            pkg["days_since_release"] = None
            pkg["stale"] = False
        return pkg
    return {"name": name, "error": "Package not found in registry", "status": "unknown"}


def check_license_compat(project_license, dep_license):
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


def find_alternatives(package_name):
    # Reach the host LLM through the Model Gateway when available; else
    # degrade to the same empty result the original returns with no LLM.
    llm = _host_llm()
    if llm is not None:
        try:
            response = llm.generate(
                f"Suggest 2-3 well-maintained alternatives to the Python/Node.js "
                f"package '{package_name}'. For each alternative, give the package "
                f"name and a one-line reason. Respond as a JSON array of objects "
                f"with 'name' and 'reason' fields. Only the JSON array, nothing else.",
                max_tokens=200,
            )
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


TOOLS = [
    {
        "name": "detect_dep_files",
        "description": "Scan a project directory for dependency files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Path to the project root."}
            },
            "required": ["project_dir"],
        },
    },
    {
        "name": "parse_deps",
        "description": "Parse a dependency file and extract package names plus version constraints.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to a dependency file (requirements.txt, "
                    "pyproject.toml, package.json, etc.).",
                }
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "check_package",
        "description": "Check a package's health - latest version, release date, CVEs, "
        "license, maintainer activity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "package_name": {"type": "string", "description": "PyPI / npm package name."}
            },
            "required": ["package_name"],
        },
    },
    {
        "name": "check_license_compat",
        "description": "Check if a dependency license is compatible with the project license.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_license": {
                    "type": "string",
                    "description": "SPDX-style identifier for the project's license.",
                },
                "dep_license": {
                    "type": "string",
                    "description": "SPDX-style identifier for the dependency's license.",
                },
            },
            "required": ["project_license", "dep_license"],
        },
    },
    {
        "name": "find_alternatives",
        "description": "Suggest alternative packages for a risky dependency using the host "
        "LLM via the Model Gateway (degrades to empty alternatives when no host LLM is "
        "reachable from the MCP subprocess).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "Package to find alternatives for.",
                }
            },
            "required": ["package_name"],
        },
    },
    {
        "name": "think",
        "description": "Record an analytical step. No side effects.",
        "inputSchema": {
            "type": "object",
            "properties": {"thought": {"type": "string", "description": "Brief reasoning note."}},
            "required": ["thought"],
        },
    },
    {
        "name": "done",
        "description": "Signal that the audit is complete.",
        "inputSchema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Audit summary."}},
            "required": ["summary"],
        },
    },
]


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _content(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


def _dispatch(name, args):
    if name == "detect_dep_files":
        return detect_dep_files(args.get("project_dir", ""))
    if name == "parse_deps":
        return parse_deps(args.get("file_path", ""))
    if name == "check_package":
        return check_package(args.get("package_name", ""))
    if name == "check_license_compat":
        return check_license_compat(args.get("project_license", ""), args.get("dep_license", ""))
    if name == "find_alternatives":
        return find_alternatives(args.get("package_name", ""))
    if name == "think":
        return {"thought": args.get("thought"), "noted": True}
    if name == "done":
        return {"status": "completed", "summary": args.get("summary")}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        msg_id = req.get("id")
        if method == "initialize":
            respond(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "dep-doctor-tools", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            result = _dispatch(name, args)
            if result is None:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
            else:
                respond(msg_id, _content(result))
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
