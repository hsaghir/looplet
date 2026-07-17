"""LinterHook subclass with to_config() so it round-trips.

The constructor accepts an optional ``workspace`` path. When omitted
(the recommended setup), it is resolved via
:func:`looplet.cartridge.runtime_helpers.resolve_project_root` - so
the hook works whether or not the host passes a runtime dict, and
without a stale "workspace" name in the config.
"""

from coder_lib_hooks import LinterHook as _LinterHook

from looplet.cartridge.runtime_helpers import resolve_project_root


class LinterHook(_LinterHook):
    def __init__(self, workspace: str | None = None) -> None:
        super().__init__(workspace=workspace or resolve_project_root())

    def to_config(self) -> dict:
        return {"workspace": self._workspace}
