from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_code", [0, 17])
def test_pre_push_isolates_git_environment(tmp_path: Path, exit_code: int) -> None:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    repository = tmp_path / "outer"
    subprocess.run(["git", "init", "--quiet", str(repository)], env=environment, check=True)
    for name, value in (
        ("user.name", "hsaghir"),
        ("user.email", "7474909+hsaghir@users.noreply.github.com"),
    ):
        subprocess.run(
            ["git", "config", "--local", name, value],
            cwd=repository,
            env=environment,
            check=True,
        )
    config_path = repository / ".git" / "config"
    original_config = config_path.read_bytes()
    local_variables = subprocess.check_output(
        ["git", "rev-parse", "--local-env-vars"],
        cwd=repository,
        env=environment,
        text=True,
    ).splitlines()
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    make = executable_directory / "make"
    make.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "assert sys.argv[1:] == ['check']\n"
        "local_variables = json.loads(os.environ['HOOK_TEST_LOCAL_VARIABLES'])\n"
        "leaked = sorted(set(local_variables).intersection(os.environ))\n"
        "assert not leaked, f'Git-local environment leaked into checks: {leaked}'\n"
        "subprocess.run(['git', 'init', '--quiet', 'nested'], check=True)\n"
        "subprocess.run(['git', '-C', 'nested', 'config', 'user.name', "
        "'fixture-author'], check=True)\n"
        "print('GIT_LOCAL_VARIABLES_CLEARED')\n"
        "raise SystemExit(int(os.environ['HOOK_TEST_EXIT']))\n",
        encoding="utf-8",
    )
    make.chmod(0o755)
    hook_environment = {
        **environment,
        "PATH": f"{executable_directory}{os.pathsep}{environment['PATH']}",
        "GIT_DIR": str(repository / ".git"),
        "GIT_WORK_TREE": str(repository),
        "GIT_INDEX_FILE": str(repository / ".git" / "index"),
        "GIT_PREFIX": "",
        "GIT_CONFIG_PARAMETERS": "'user.name=fixture-override'",
        "HOOK_TEST_LOCAL_VARIABLES": json.dumps(local_variables),
        "HOOK_TEST_EXIT": str(exit_code),
    }
    hook = Path(__file__).resolve().parents[1] / "scripts" / "pre-push.sh"
    result = subprocess.run(
        ["bash", str(hook)],
        cwd=repository,
        env=hook_environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert "GIT_LOCAL_VARIABLES_CLEARED" in result.stdout
    assert config_path.read_bytes() == original_config
    assert (
        subprocess.check_output(
            ["git", "-C", "nested", "config", "--local", "user.name"],
            cwd=repository,
            env=environment,
            text=True,
        ).strip()
        == "fixture-author"
    )
