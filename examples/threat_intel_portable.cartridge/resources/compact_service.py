"""Compaction service for the threat_intel workspace.

Uses looplet's production default as a declarative resource.
"""

from __future__ import annotations

from looplet.compact import default_compact_service


def build(runtime=None):
    return default_compact_service(keep_recent=3, keep_recent_tool_results=6)
