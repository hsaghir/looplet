"""Result record for the private-loop migration recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from looplet import BaseToolRegistry


@dataclass(frozen=True)
class MigrationResult:
    """Evidence and control identities produced by the four stages."""

    output_dir: Path
    raw_decisions: tuple[str, ...]
    owned_decisions: tuple[str, ...]
    graded_decisions: tuple[str, ...]
    replayed_decisions: tuple[str, ...]
    raw_invocations: tuple[str, ...]
    owned_invocations: tuple[str, ...]
    raw_handoff: dict[str, Any]
    owned_handoff: dict[str, Any]
    before_fix_artifacts: dict[str, Any]
    after_fix_artifacts: dict[str, Any]
    before_fix_score: float
    after_fix_score: float
    recorded_calls: int
    failure_run: Path
    fixed_run: Path
    changed_lines: tuple[str, str]
    raw_registry: BaseToolRegistry
    owned_registry: BaseToolRegistry
    private_callables: dict[str, Any]

    @property
    def reuses_private_tools(self) -> bool:
        """True when both loops dispatched the same private callables."""
        return bool(self.raw_invocations) and self.raw_invocations == self.owned_invocations

    @property
    def loop_swap_kept_the_outcome(self) -> bool:
        """True when replacing the loop changed neither decisions nor world state."""
        return self.raw_decisions == self.owned_decisions and self.raw_handoff == self.owned_handoff

    @property
    def same_recorded_decisions(self) -> bool:
        """True when replay consumed the decisions the failing run recorded."""
        return self.replayed_decisions == self.graded_decisions
