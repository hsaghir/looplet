"""Cartridge-free recipe: a private tool loop becomes a regression contract."""

from .execution import (
    build_registry,
    run_owned_loop,
    run_raw_loop,
)
from .handoff_contract import (
    CASE,
    build_eval_hook,
    eval_open_incidents_are_correct,
    live_task,
    make_handoff_collector,
    required_score,
)
from .handoff_tools import (
    HANDOFF_FILE,
    INCIDENTS_FILE,
    ToolSuite,
    build_tool_suite,
    changed_implementation_lines,
    select_open_incidents,
    select_open_incidents_with_bug,
)
from .result import MigrationResult
from .run_recipe import (
    render,
    run_migration,
)

__all__ = [
    "CASE",
    "HANDOFF_FILE",
    "INCIDENTS_FILE",
    "MigrationResult",
    "ToolSuite",
    "build_eval_hook",
    "build_registry",
    "build_tool_suite",
    "changed_implementation_lines",
    "eval_open_incidents_are_correct",
    "live_task",
    "make_handoff_collector",
    "render",
    "required_score",
    "run_migration",
    "run_owned_loop",
    "run_raw_loop",
    "select_open_incidents",
    "select_open_incidents_with_bug",
]
