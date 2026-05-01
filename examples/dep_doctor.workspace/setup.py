"""Wire LoopConfig.compact_service for the dep_doctor workspace."""

from __future__ import annotations


def setup(preset, resources, runtime=None):
    from looplet.compact import (  # noqa: PLC0415
        PruneToolResults,
        TruncateCompact,
        compact_chain,
    )

    preset.config.compact_service = compact_chain(
        PruneToolResults(keep_recent=8),
        TruncateCompact(keep_recent=3),
    )
    return preset
