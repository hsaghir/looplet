"""Compaction service for the coder workspace.

Uses looplet's production default: prune old tool payloads, summarize
older working context, keep recent steps verbatim, and truncate only as
a fallback. Kept as a declarative resource so ``config.yaml`` can refer
to it with ``compact_service: "@compact_service"``.
"""

from __future__ import annotations

from looplet.compact import default_compact_service


def build(runtime=None):
    return default_compact_service(keep_recent=5, keep_recent_tool_results=10)
