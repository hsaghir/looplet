"""Compaction service for the coder workspace.

Two-stage chain: prune old tool results first, then truncate any
remaining historical turns. Mirrors the pattern the looplet.examples coder reference
used; lifts it out of ``setup.py`` so the entire workspace is
declarative.
"""

from __future__ import annotations

from looplet.compact import PruneToolResults, TruncateCompact, compact_chain


def build(runtime=None):
    return compact_chain(
        PruneToolResults(keep_recent=10),
        TruncateCompact(keep_recent=5),
    )
