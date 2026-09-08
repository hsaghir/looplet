"""Tests for the shared run lifecycle transition policy."""

from __future__ import annotations

import pytest

from looplet import (
    InvalidRunTransition,
    RunPhase,
    RunStatus,
    validate_run_transition,
)


def test_valid_run_transition_chain() -> None:
    status = RunStatus.CREATED
    phase = RunPhase.STARTING
    transitions = [
        (RunStatus.RUNNING, RunPhase.PROMPTING),
        (RunStatus.RUNNING, RunPhase.LLM),
        (RunStatus.RUNNING, RunPhase.DISPATCHING),
        (RunStatus.RUNNING, RunPhase.FINALIZING),
        (RunStatus.COMPLETED, RunPhase.TERMINAL),
    ]

    for next_status, next_phase in transitions:
        validate_run_transition(
            status,
            phase,
            next_status=next_status,
            next_phase=next_phase,
        )
        status, phase = next_status, next_phase


@pytest.mark.parametrize(
    ("current_status", "current_phase", "next_status", "next_phase"),
    [
        (RunStatus.CREATED, RunPhase.STARTING, RunStatus.COMPLETED, RunPhase.TERMINAL),
        (RunStatus.RUNNING, RunPhase.LLM, RunStatus.RUNNING, RunPhase.STARTING),
        (RunStatus.COMPLETED, RunPhase.TERMINAL, RunStatus.RUNNING, RunPhase.PROMPTING),
        (RunStatus.RUNNING, RunPhase.TERMINAL, RunStatus.RUNNING, RunPhase.TERMINAL),
    ],
)
def test_invalid_run_transition_is_explicit(
    current_status: RunStatus,
    current_phase: RunPhase,
    next_status: RunStatus,
    next_phase: RunPhase,
) -> None:
    with pytest.raises(InvalidRunTransition):
        validate_run_transition(
            current_status,
            current_phase,
            next_status=next_status,
            next_phase=next_phase,
        )


def test_terminal_status_requires_terminal_phase() -> None:
    with pytest.raises(InvalidRunTransition):
        validate_run_transition(
            RunStatus.RUNNING,
            RunPhase.LLM,
            next_status=RunStatus.STOPPED,
            next_phase=RunPhase.STOPPING,
        )
