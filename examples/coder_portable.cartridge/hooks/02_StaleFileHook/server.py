"""StaleFileHook as a portable ``kind: lep`` server.

After a bash step, asks the shared file cache (State Service) which
previously-read files were modified, and nudges the model to re-read
them. The cache proxy connects to the SAME socket the file tools and
FileCacheHook use. Only ``post_dispatch`` is implemented.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_lep"))

from coder_lib_hooks import StaleFileHook  # noqa: E402
from lep_common import FileCacheProxy, normalize, view_to_call, view_to_result  # noqa: E402

from looplet.lep import LEPServerBase  # noqa: E402


class StaleFileServer(LEPServerBase):
    slots = ("post_dispatch",)
    effects = ("Continue", "InjectContext")
    view_fields = ("tool", "args", "tool_result")
    view_fidelity = "full"

    def __init__(self) -> None:
        self._hook = StaleFileHook(FileCacheProxy())

    def decide(self, slot, view):
        if slot == "post_dispatch":
            return normalize(
                self._hook.post_dispatch(None, None, view_to_call(view), view_to_result(view), 0)
            )
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(StaleFileServer().serve())
