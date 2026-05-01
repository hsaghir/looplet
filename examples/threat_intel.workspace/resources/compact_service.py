"""Compaction service for the threat_intel workspace.

PruneToolResults(keep_recent=6) + TruncateCompact(keep_recent=3).
Lifted out of setup.py so the workspace is fully declarative.
"""

from __future__ import annotations

from looplet.compact import PruneToolResults, TruncateCompact, compact_chain


def build(runtime=None):
    return compact_chain(
        PruneToolResults(keep_recent=6),
        TruncateCompact(keep_recent=3),
    )
