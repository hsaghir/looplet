"""LinterHook as a portable ``kind: lep`` server.

Runs ``ruff check`` after Python ``edit_file`` / ``write_file`` steps
and surfaces diagnostics. Self-contained: the vendored hook resolves
``ruff`` in the project venv / PATH and shells out itself (stdlib
``subprocess``), so no shared host state is needed. The workspace is
resolved via :func:`looplet.cartridge.runtime_helpers.resolve_project_root`
(``$LOOPLET_PROJECT_ROOT`` / git toplevel / cwd). Only ``post_dispatch``
is implemented.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_lep"))

from coder_lib_hooks import LinterHook  # noqa: E402
from lep_common import normalize, view_to_call, view_to_result  # noqa: E402

from looplet.cartridge.runtime_helpers import resolve_project_root  # noqa: E402
from looplet.lep import LEPServerBase  # noqa: E402


class LinterServer(LEPServerBase):
    slots = ("post_dispatch",)
    effects = ("Continue", "InjectContext")
    view_fields = ("tool", "args", "tool_result")
    view_fidelity = "full"

    def __init__(self) -> None:
        self._hook = LinterHook(resolve_project_root())

    def decide(self, slot, view):
        if slot == "post_dispatch":
            return normalize(
                self._hook.post_dispatch(None, None, view_to_call(view), view_to_result(view), 0)
            )
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(LinterServer().serve())
