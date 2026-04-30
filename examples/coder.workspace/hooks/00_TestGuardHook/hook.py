# Re-import the original TestGuardHook so its full closure (typing
# imports, looplet helpers) stays in scope. Add to_config() so the
# workspace round-trip captures the constructor kwargs.
from examples.coder.hooks import TestGuardHook as _OriginalTestGuardHook


class TestGuardHook(_OriginalTestGuardHook):
    """Workspace-friendly subclass with a to_config() method."""

    def to_config(self) -> dict:
        return {"strict": self._strict}
