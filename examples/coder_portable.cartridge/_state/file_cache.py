"""Shared file cache - ported from the in-process ``@ref`` resource to an
out-of-process **State Service** (State Service Protocol).

In the original ``coder`` cartridge the ``FileCache`` lived at
``resources/file_cache.py`` as a single instance shared (via
``"@file_cache"`` refs) between the read/write/edit tools AND the
StaleFile/FileCache hooks - all running in the SAME Python process.
That shared-address-space requirement is exactly what pinned the
cartridge to a Python host.

Here the same cache lives in ITS OWN process behind a Unix socket. The
loader spawns this server and exports the socket path as
``LOOPLET_STATE_FILE_CACHE`` so the MCP tools server (read_file,
write_file, edit_file, multi_edit, notebook_edit, bash) and the LEP
hooks (StaleFile, FileCache) - each a SEPARATE process - connect to and
mutate the SAME cache, reproducing the in-process ``@ref`` sharing
across the process boundary.

The underlying :class:`FileCache` logic is vendored unchanged in
``../_mcp/coder_lib_tools.py``; this service is a thin SSP delegator
over its public surface. All method calls are serialized under one
lock so concurrent readers/writers (tool process + hook processes)
see a consistent cache.
"""

import os
import sys

# The vendored FileCache implementation lives next to the MCP tools
# server. Put that dir on the path so this separate process can reuse
# the exact same logic the in-process original used.
_MCP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_mcp")
sys.path.insert(0, _MCP_DIR)

from coder_lib_tools import FileCache  # noqa: E402

from looplet.state_service import StateServiceBase  # noqa: E402


def _workspace() -> str:
    return os.environ.get("LOOPLET_PROJECT_ROOT") or os.getcwd()


class FileCacheService(StateServiceBase):
    """The shared coder file cache, served over SSP.

    Delegates to a single :class:`FileCache` bound to the project root.
    Every method the tools and hooks call in-process on the ``@ref``
    cache is exposed here verbatim so the proxy on the other side of
    the socket is a drop-in replacement.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache = FileCache(_workspace())

    # ── reads/writes recorded by the file tools ──────────────────
    def record(self, path: str) -> None:
        self._cache.record(path)

    def invalidate(self, path: str) -> None:
        self._cache.invalidate(path)

    def is_unchanged(self, path: str) -> bool:
        return self._cache.is_unchanged(path)

    def was_read(self, path: str) -> bool:
        return self._cache.was_read(path)

    # ── bash loop-detection used by the bash tool ────────────────
    def record_bash(self, command: str) -> int:
        return self._cache.record_bash(command)

    def recent_bash_repeats(self, command: str) -> int:
        return self._cache.recent_bash_repeats(command)

    # ── stale detection + briefing render used by the hooks ──────
    def stale_files(self) -> list:
        return self._cache.stale_files()

    def render(self) -> str:
        return self._cache.render()


if __name__ == "__main__":
    raise SystemExit(FileCacheService().serve())
