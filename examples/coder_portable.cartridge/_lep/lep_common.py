"""Shared helpers for the portable coder LEP hook servers.

Two pieces the in-process hooks took for granted now cross process
boundaries:

* ``FileCacheProxy`` - the StaleFile/FileCache hooks were constructed
  with the shared ``@file_cache`` instance. Here that cache lives in a
  State Service; this proxy forwards the two methods those hooks call
  (``stale_files`` / ``render``) over the socket the loader exported as
  ``LOOPLET_STATE_FILE_CACHE`` - the SAME socket the MCP file tools use,
  so hook and tool processes observe one cache.

* ``view_to_call`` - the LEP adapter ships a declared *view* dict; the
  vendored hook methods expect ``tool_call`` / ``tool_result`` objects.
  This rebuilds the minimal duck-typed objects those methods read.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from looplet.hook_decision import HookDecision, InjectContext
from looplet.state_service import StateServiceClient


class FileCacheProxy:
    """SSP-backed stand-in for the shared FileCache (hook surface only)."""

    def __init__(self) -> None:
        self._c: StateServiceClient | None = None
        socket_path = os.environ.get("LOOPLET_STATE_FILE_CACHE")
        if socket_path:
            try:
                self._c = StateServiceClient(socket_path)
            except Exception:  # noqa: BLE001 - degrade gracefully
                self._c = None

    def stale_files(self) -> list:
        if self._c is None:
            return []
        return list(self._c.stale_files() or [])

    def render(self) -> str:
        if self._c is None:
            return ""
        return str(self._c.render() or "")


def view_to_call(view: dict[str, Any]) -> SimpleNamespace:
    """Rebuild a ``tool_call``-shaped object from a LEP view."""
    return SimpleNamespace(tool=view.get("tool"), args=view.get("args") or {})


def view_to_result(view: dict[str, Any]) -> SimpleNamespace:
    """Rebuild a ``tool_result``-shaped object from a LEP view."""
    tr = view.get("tool_result") or {}
    return SimpleNamespace(data=tr.get("data") or {}, error=tr.get("error"))


def normalize(decision: Any) -> Any:
    """Coerce a vendored hook return into a LEP-acceptable effect.

    The in-process hook slots return ``HookDecision``/``InjectContext``
    (already fine) - but ``pre_prompt`` returns a bare ``str`` briefing.
    LEP's ``_effect_dict`` only accepts ``HookDecision|dict|None``, so a
    plain string is wrapped as an ``InjectContext`` (the adapter then
    surfaces it via ``decision.additional_context``).
    """
    if decision is None:
        return None
    if isinstance(decision, HookDecision):
        return decision
    if isinstance(decision, str):
        return InjectContext(decision) if decision else None
    return decision
