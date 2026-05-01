"""Wire LoopConfig.compact_service for the threat-intel workspace.

The v1 cartridge sets compact_service=compact_chain(PruneToolResults
(keep_recent=6), TruncateCompact(keep_recent=3)). compact_service is
a non-JSON-able callable, so it can't go in config.yaml — setup.py
attaches it after declarative load.
"""

from __future__ import annotations


def setup(preset, resources, runtime=None):
    from looplet.compact import (  # noqa: PLC0415
        PruneToolResults,
        TruncateCompact,
        compact_chain,
    )

    preset.config.compact_service = compact_chain(
        PruneToolResults(keep_recent=6),
        TruncateCompact(keep_recent=3),
    )
    return preset
