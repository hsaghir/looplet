from __future__ import annotations

import pytest

from looplet.execution_session import ExecutionSession


def test_session_attaches_and_forks() -> None:
    session = ExecutionSession(session_id="session-1")
    assert session.attach("run-1") == "run-1"
    assert session.attach("run-1") == "run-1"
    child = session.fork()
    assert child.parent_run_id == "run-1"
    assert child.session_id != session.session_id


def test_closed_session_rejects_new_work() -> None:
    session = ExecutionSession()
    session.close()
    with pytest.raises(RuntimeError):
        session.attach("run-1")
