"""FileCacheHook as a portable ``kind: lep`` server.

Re-injects the shared file cache into the briefing after step 3
(post-compaction safety). The cache lives in a State Service; this
server reaches it through :class:`lep_common.FileCacheProxy` over the
SAME socket the MCP file tools use, so the rendered cache reflects the
tools' reads. Only the ``pre_prompt`` slot is implemented.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_lep"))

from coder_lib_hooks import FileCacheHook  # noqa: E402
from lep_common import FileCacheProxy, normalize  # noqa: E402

from looplet.lep import LEPServerBase  # noqa: E402


class FileCacheServer(LEPServerBase):
    slots = ("pre_prompt",)
    effects = ("Continue", "InjectContext")
    view_fields = ("step",)
    view_fidelity = "digest"

    def __init__(self) -> None:
        self._hook = FileCacheHook(FileCacheProxy())

    def decide(self, slot, view):
        if slot == "pre_prompt":
            step = int(view.get("step") or 0)
            return normalize(self._hook.pre_prompt(None, None, None, step))
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(FileCacheServer().serve())
