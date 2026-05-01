"""StaleFileHook — detects bash-induced file changes that bypass the cache."""

from coder_lib_hooks import StaleFileHook as _OriginalStaleFileHook


class StaleFileHook(_OriginalStaleFileHook):
    def to_config(self) -> dict:
        return {"cache": "@file_cache"}
