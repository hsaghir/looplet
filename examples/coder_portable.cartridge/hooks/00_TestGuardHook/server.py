"""TestGuardHook as a portable ``kind: lep`` server.

Wraps the vendored :class:`coder_lib_hooks.TestGuardHook` (observe-only
by default). It keeps its own ``_tests_passed`` / ``_files_written``
state across slot calls IN THIS PROCESS - the LEP server is long-lived,
so the in-process statefulness survives unchanged. Two slots:

* ``post_dispatch`` - watch bash test runs + record written files.
* ``check_done`` - surface the finishing-without-tests nudge
  (strict=False never blocks).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_lep"))

from coder_lib_hooks import TestGuardHook  # noqa: E402
from lep_common import normalize, view_to_call, view_to_result  # noqa: E402

from looplet.lep import LEPServerBase  # noqa: E402


class TestGuardServer(LEPServerBase):
    slots = ("post_dispatch", "check_done")
    effects = ("Continue", "InjectContext", "Block", "HookDecision")
    view_fields = ("tool", "args", "tool_result", "step")
    view_fidelity = "full"

    def __init__(self) -> None:
        self._hook = TestGuardHook(strict=False)

    def decide(self, slot, view):
        step = int(view.get("step") or 0)
        if slot == "post_dispatch":
            return normalize(
                self._hook.post_dispatch(None, None, view_to_call(view), view_to_result(view), step)
            )
        if slot == "check_done":
            return normalize(self._hook.check_done(None, None, None, step))
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(TestGuardServer().serve())
