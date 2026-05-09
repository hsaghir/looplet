"""Async human steering for an in-flight loop.

Pi popularised a queue model: while the agent is mid-tool-call the
operator types a "steering" message; it is delivered before the next
prompt without aborting the current step. looplet exposes the same
pattern as a thread-safe :class:`SteeringQueue` plus a
:class:`SteeringHook` that drains the queue in ``pre_prompt``.

The queue is intentionally minimal — no UI, no transport, no async
runtime. Producers push from any thread (or signal handler / web
endpoint / CLI input thread); the loop reads on its turn. Two delivery
modes mirror Pi:

* ``"one-at-a-time"`` (default): one message per turn. Remaining
  messages stay in the queue.
* ``"all"``: all queued messages joined and delivered in one turn.

Usage::

    from looplet.steering import SteeringQueue, SteeringHook

    q = SteeringQueue()
    hooks = [SteeringHook(q)]

    # ── from another thread ───────────────────────────────────────
    q.steer("Stop editing tests; focus on the API surface first.")

    for step in composable_loop(llm=..., hooks=hooks, ...):
        ...

Timing — what steers can and cannot do
--------------------------------------

A steer lands in the briefing of the **next** prompt. That means
steers influence the next *decision*, not files the model already
wrote. Empirically (verified against Claude Sonnet 4.5):

* "Use the X library when you call Y" — landed reliably; Y had not
  been called yet.
* "Add a docstring to the function you just wrote" — usually ignored;
  the model considered the file done and moved on.
* "Refactor the class you wrote in step 3 to do Z" — sometimes ignored,
  sometimes triggered a partial rewrite, never deterministic.

Treat steering as **forward guidance**: prefer "from now on, …" or
"when you write X, do Y" over "go back and fix what you wrote in step
N". For retroactive changes, append the requirement to the task
description and let the model re-plan from scratch — that is what the
loop is built to do.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "SteeringQueue",
    "SteeringHook",
    "DeliveryMode",
]

DeliveryMode = Literal["one-at-a-time", "all"]


@dataclass
class SteeringQueue:
    """Thread-safe FIFO queue of operator steering messages.

    Producers call :meth:`steer` from any thread; the loop calls
    :meth:`drain` once per turn (via :class:`SteeringHook`).
    """

    mode: DeliveryMode = "one-at-a-time"
    prefix: str = "[OPERATOR STEERING]"
    _q: deque[str] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def steer(self, message: str) -> None:
        """Enqueue a steering message. No-op for empty/whitespace text."""
        s = message.strip()
        if not s:
            return
        with self._lock:
            self._q.append(s)

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def pending(self) -> list[str]:
        """Return a copy of pending messages without consuming."""
        with self._lock:
            return list(self._q)

    def drain(self) -> str | None:
        """Pop messages per ``mode`` and render as a single string.

        Returns None when the queue is empty so :meth:`pre_prompt` can
        skip injection cleanly.
        """
        with self._lock:
            if not self._q:
                return None
            if self.mode == "one-at-a-time":
                msgs = [self._q.popleft()]
            else:  # "all"
                msgs = list(self._q)
                self._q.clear()
        body = "\n\n".join(f"- {m}" for m in msgs)
        return f"{self.prefix}\n{body}"

    def clear(self) -> None:
        """Drop all pending messages without delivering them."""
        with self._lock:
            self._q.clear()


@dataclass
class SteeringHook:
    """Hook that drains a :class:`SteeringQueue` in ``pre_prompt``.

    Install via the ``hooks=`` list on :func:`composable_loop`. The
    returned text is appended to the briefing section of the next
    prompt — the same channel used by ``InjectContext``.
    """

    queue: SteeringQueue

    def pre_prompt(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        return self.queue.drain()
