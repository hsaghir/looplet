"""Shared file_cache — content cache for write/read coordination.

The coder agent's read_file tool stores file content here; write_file
and edit_file evict / refresh entries. StaleFileHook reads the same
cache to detect when bash commands change files the model already read.

All three (read_file/write_file/edit_file tools, FileCacheHook,
StaleFileHook) reference this via ``"@file_cache"`` so they share
one instance. Without the @ref shared registry each would silently
get its own empty cache and the staleness detection would break.

Reads ``runtime["workspace"]`` so the FileCache is bound to the same
path the host CLI gave the workspace.
"""

from coder_lib_tools import FileCache


def build(runtime=None):
    runtime = runtime or {}
    return FileCache(workspace=str(runtime.get("workspace", ".")))
