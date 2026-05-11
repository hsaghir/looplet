"""LinterHook subclass with to_config() so it round-trips.

The constructor takes ``workspace: str``; the workspace dir is supplied
by the host via the runtime kwarg, threaded into the hook's config.yaml
through ``${runtime.workspace}`` template substitution.
"""

from coder_lib_hooks import LinterHook as _LinterHook


class LinterHook(_LinterHook):
    def to_config(self) -> dict:
        return {"workspace": self._workspace}
