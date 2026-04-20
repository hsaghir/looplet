"""Prompt-cache primitives for Anthropic-style ``cache_control``.

Long agent sessions re-send the same system prompt + tool schemas +
persistent memory on every turn. Providers that expose prompt caching
(Anthropic, Bedrock, Vertex) charge ~10% of normal input price on
cache hits and skip prefill compute, yielding ~50% end-to-end latency
savings on long sessions.

OpenHarness historically had **zero** cache awareness — every turn
rebuilt the full prompt and emitted it with no cache hints. This
module adds the plumbing:

* :class:`CachePolicy` — declarative: which sections the caller
  considers stable enough to cache, plus per-section TTL.
* :class:`CacheBreakpoint` — a (label, content_hash, ttl) tuple
  emitted per turn and handed to cache-aware backends.
* :func:`compute_breakpoints` — deterministic hash of the three
  canonical stable sections (system prompt, tool schemas, memory).
* :class:`CacheBreakDetector` — observer :class:`LoopHook` that
  records hashes per turn and logs / emits events when any stable
  section's hash changes (i.e. a cache break has occurred).

Backend integration is opt-in: backends that expose
``generate_with_cache(prompt, *, cache_breakpoints=[...], ...)`` get
invoked with the computed breakpoints; backends without that method
keep working unchanged. The loop never forces caching.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "CacheControl",
    "CachePolicy",
    "CacheBreakpoint",
    "CacheSection",
    "compute_breakpoints",
    "CacheBreakDetector",
]

logger = logging.getLogger(__name__)

# Anthropic supports "ephemeral" (5-min) + "1h" (Sonnet 3.7+). We keep
# the vocabulary neutral so Bedrock/Vertex can map onto their own TTL
# models. Unknown values are forwarded to the backend untouched.
_CacheTTL = Literal["ephemeral", "1h"]

CacheSection = Literal["system_prompt", "tool_schemas", "memory"]
"""The three sections the default policy knows about. These are the
only prompt regions that are (a) stable across turns within a
session and (b) large enough to justify a cache entry."""

@dataclass(frozen=True)
class CacheControl:
    """Per-section caching declaration. Frozen so it's safe to share
    across workers / threads and usable as a dict key for diagnostics.

    Mirrors Anthropic's ``cache_control`` block shape
    (``{"type": "ephemeral", "ttl": "1h"}``) with enough fidelity to
    round-trip and enough neutrality to work across providers.
    """

    ttl: _CacheTTL = "ephemeral"
    """Cache TTL. Default 5-min ephemeral; use ``"1h"`` when you know
    sessions will last long enough to amortise the higher write cost."""

@dataclass(frozen=True)
class CacheBreakpoint:
    """A single cache-break marker produced per turn.

    The ``hash`` fingerprints the content covered by this breakpoint
    (system prompt bytes, concatenated tool schemas, rendered memory).
    When two consecutive turns emit different hashes for the same
    label, the provider's cache is invalidated — a "cache break".

    Cache-aware backends consume ``label`` + ``content`` to place
    ``cache_control`` blocks in the right provider-specific slot.
    """

    label: CacheSection
    hash: str
    content: str
    control: CacheControl = field(default_factory=CacheControl)

@dataclass
class CachePolicy:
    """Declarative cache policy attached to :class:`LoopConfig`.

    Caller specifies which sections are stable enough to cache and
    the TTL for each. The loop computes hashes per turn and hands
    the resulting :class:`CacheBreakpoint` list to the backend via
    ``generate_with_cache`` (opt-in). Unknown backends see nothing —
    caching is strictly additive.

    Sections default to ``None`` (= not cached). Enable only what's
    actually stable — e.g. don't cache memory if you rewrite it
    every turn, or you'll eat write costs for zero hit rate.
    """

    system_prompt: CacheControl | None = None
    tool_schemas: CacheControl | None = None
    memory: CacheControl | None = None

    def sections(self) -> list[tuple[CacheSection, CacheControl]]:
        """Enumerate configured sections in stable order (important
        for cache-key stability — reordering breaks caches)."""
        out: list[tuple[CacheSection, CacheControl]] = []
        if self.system_prompt is not None:
            out.append(("system_prompt", self.system_prompt))
        if self.tool_schemas is not None:
            out.append(("tool_schemas", self.tool_schemas))
        if self.memory is not None:
            out.append(("memory", self.memory))
        return out

def _hash(text: str) -> str:
    """Stable 16-char hex hash for cache-break detection. SHA-256
    truncated — collision risk is irrelevant at session scale."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def compute_breakpoints(
    policy: CachePolicy,
    *,
    system_prompt: str,
    tool_schemas_text: str,
    memory_text: str,
) -> list[CacheBreakpoint]:
    """Build the per-turn breakpoint list from the current policy.

    Callers pass the raw text of each stable section; this function
    hashes + packages. Returns ``[]`` when the policy is empty so
    callers can skip the cache-aware backend path entirely.
    """
    out: list[CacheBreakpoint] = []
    for label, ctl in policy.sections():
        content = {
            "system_prompt": system_prompt,
            "tool_schemas": tool_schemas_text,
            "memory": memory_text,
        }[label]
        out.append(CacheBreakpoint(
            label=label, hash=_hash(content), content=content, control=ctl,
        ))
    return out

class CacheBreakDetector:
    """Observer :class:`LoopHook` that records section hashes per turn
    and logs when any previously-cached section changes.

    Use when you want production telemetry on cache miss rate without
    changing any dispatch logic. The hook never blocks or mutates —
    it just records the hash trail and emits ``cache_break`` log
    entries so you can count breaks in your log pipeline.

    Call :meth:`breaks` after a run to see (turn, label, old, new)
    tuples; useful in tests to assert "session X had ≤ N cache
    breaks".
    """

    def __init__(self, policy: CachePolicy) -> None:
        self._policy = policy
        self._last: dict[CacheSection, str] = {}
        self._breaks: list[tuple[int, CacheSection, str, str]] = []

    def record(
        self,
        step_num: int,
        *,
        system_prompt: str,
        tool_schemas_text: str,
        memory_text: str,
    ) -> list[CacheBreakpoint]:
        """Compute breakpoints for the current turn and diff against
        the previous turn. Logs + records any section-level change.

        Returns the breakpoint list so callers can forward it straight
        to a cache-aware backend without re-hashing.
        """
        bps = compute_breakpoints(
            self._policy,
            system_prompt=system_prompt,
            tool_schemas_text=tool_schemas_text,
            memory_text=memory_text,
        )
        for bp in bps:
            prior = self._last.get(bp.label)
            if prior is not None and prior != bp.hash:
                self._breaks.append((step_num, bp.label, prior, bp.hash))
                logger.warning(
                    "cache_break at step=%d section=%s prior=%s new=%s",
                    step_num, bp.label, prior, bp.hash,
                )
            self._last[bp.label] = bp.hash
        return bps

    @property
    def breaks(self) -> list[tuple[int, CacheSection, str, str]]:
        """All cache-break events recorded so far (step, section,
        prior_hash, new_hash). Empty means no breaks occurred — a
        perfectly cache-stable session."""
        return list(self._breaks)

    # ── LoopHook Protocol no-ops ───────────────────────────────
    # So an instance can be registered in ``hooks=[...]`` directly
    # without wrapping. The loop calls record() itself from the
    # prompt-build path; these are just here to satisfy the protocol.

