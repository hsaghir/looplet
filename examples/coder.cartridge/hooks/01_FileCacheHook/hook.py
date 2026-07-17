"""FileCacheHook - observes file reads/writes and updates the shared cache.

Re-imports the original class from examples.coder.hooks. Adds to_config()
so the workspace round-trip captures the @ref to the shared FileCache.
"""

from coder_lib_hooks import FileCacheHook as _OriginalFileCacheHook


class FileCacheHook(_OriginalFileCacheHook):
    def to_config(self) -> dict:
        # The shared FileCache instance becomes a @ref string in
        # config.yaml; the loader resolves it back to the SAME object
        # the read_file/write_file/edit_file tools mutate.
        return {"cache": "@file_cache"}
