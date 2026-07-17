"""Shared file_cache - content cache for write/read coordination.

The coder agent's read_file tool stores file content here; write_file
and edit_file evict / refresh entries. StaleFileHook reads the same
cache to detect when bash commands change files the model already read.

All three (read_file/write_file/edit_file tools, FileCacheHook,
StaleFileHook) reference this via ``"@file_cache"`` so they share
one instance. Without the @ref shared registry each would silently
get its own empty cache and the staleness detection would break.

Resolved via :func:`looplet.cartridge.runtime_helpers.resolve_project_root`
so a host running the agent from inside the target repo doesn't need
to pass any runtime kwargs.
"""

from coder_lib_tools import FileCache

from looplet.cartridge.runtime_helpers import resolve_project_root


def build(runtime=None):
    return FileCache(workspace=resolve_project_root(runtime))
