"""Tests for SteeringQueue + SteeringHook."""

from __future__ import annotations

import threading

from looplet.steering import SteeringHook, SteeringQueue


def test_empty_queue_drain_returns_none() -> None:
    q = SteeringQueue()
    assert q.drain() is None
    assert len(q) == 0


def test_one_at_a_time_default() -> None:
    q = SteeringQueue()
    q.steer("focus on tests")
    q.steer("then refactor")

    msg1 = q.drain()
    assert msg1 is not None and "focus on tests" in msg1
    assert "then refactor" not in msg1
    assert len(q) == 1

    msg2 = q.drain()
    assert msg2 is not None and "then refactor" in msg2
    assert q.drain() is None


def test_all_mode_delivers_everything_at_once() -> None:
    q = SteeringQueue(mode="all")
    q.steer("a")
    q.steer("b")
    q.steer("c")

    msg = q.drain()
    assert msg is not None
    assert "a" in msg and "b" in msg and "c" in msg
    assert len(q) == 0


def test_empty_messages_ignored() -> None:
    q = SteeringQueue()
    q.steer("")
    q.steer("   ")
    q.steer("\n")
    assert len(q) == 0
    assert q.drain() is None


def test_clear_drops_pending() -> None:
    q = SteeringQueue()
    q.steer("a")
    q.steer("b")
    q.clear()
    assert q.drain() is None


def test_pending_does_not_consume() -> None:
    q = SteeringQueue()
    q.steer("hi")
    assert q.pending() == ["hi"]
    assert q.pending() == ["hi"]  # still there
    assert q.drain() is not None
    assert q.pending() == []


def test_thread_safe_concurrent_producers() -> None:
    q = SteeringQueue()

    def producer(i: int) -> None:
        for j in range(50):
            q.steer(f"msg-{i}-{j}")

    threads = [threading.Thread(target=producer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(q) == 5 * 50


def test_steering_hook_pre_prompt_returns_drain() -> None:
    q = SteeringQueue()
    q.steer("hello")

    hook = SteeringHook(q)
    out = hook.pre_prompt(state=None, session_log=None, context=None, step_num=1)
    assert out is not None and "hello" in out

    # Next call: queue empty → None (loop will skip injection)
    assert hook.pre_prompt(state=None, session_log=None, context=None, step_num=2) is None


def test_prefix_customizable() -> None:
    q = SteeringQueue(prefix="!! USER !!")
    q.steer("ping")
    out = q.drain() or ""
    assert out.startswith("!! USER !!")
