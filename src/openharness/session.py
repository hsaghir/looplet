"""Session log — the agent's memory of what it did and found.

Records per-step: theory, tool called, entities discovered,
findings, recall keys. Domain-agnostic — works for any agent pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class LogEntry:
    """One step in the session log.

    Captures the agent's working theory at that step, which tool was
    called, the reasoning, entities observed, structured findings,
    notable items (highlights), and an optional recall key for
    retrieving stored data.
    """

    step: int
    """1-based step index within the agent run."""

    theory: str
    """The agent's working theory or hypothesis at this step."""

    tool: str
    """Name of the tool invoked during this step."""

    reasoning: str
    """The agent's stated reason for choosing this tool."""

    entities_seen: list[str] = field(default_factory=list)
    """Entities observed or extracted during this step."""

    findings: list[str] = field(default_factory=list)
    """Structured findings produced by this step."""

    highlights: list[str] = field(default_factory=list)
    """Notable items surfaced during this step (e.g. anomalies, key values)."""

    recall_key: str = ""
    """Optional key for retrieving stored result data."""

    def render(self, *, highlights_label: str = "highlights") -> str:
        """Render this entry for prompt context inclusion."""
        parts = [f"  S{self.step}: {self.tool} — {self.reasoning[:80]}"]
        if self.entities_seen:
            parts.append(f"    entities: {', '.join(self.entities_seen[:8])}")
        if self.highlights:
            parts.append(f"    {highlights_label}: {', '.join(self.highlights)}")
        if self.findings:
            for f in self.findings[:2]:
                parts.append(f"    → {f}")
        if self.recall_key:
            parts.append(f"    [recall: {self.recall_key}]")
        return "\n".join(parts)


class SessionLog:
    """Session memory — records what the agent has done and found.

    Organized as a linear narrative log: each step records tool, reasoning,
    entities discovered, findings, and a recall key for data retrieval.

    Use record() to append steps, render() for full context inclusion,
    render_compact() for a deterministic compressed summary, and
    compact() to compress old steps in place.
    """

    def __init__(
        self,
        *,
        entity_formatter: Callable[[set[str]], list[str]] | None = None,
        title: str = "SESSION LOG",
        highlights_label: str = "highlights",
    ) -> None:
        self.entries: list[LogEntry] = []
        self.current_theory: str = ""
        self._entity_formatter = entity_formatter
        self._title = title
        self._highlights_label = highlights_label

    def record(
        self,
        step: int,
        theory: str,
        tool: str,
        reasoning: str,
        entities: list[str] | None = None,
        findings: list[str] | None = None,
        highlights: list[str] | None = None,
        recall_key: str = "",
    ) -> None:
        """Record one agent step into the session log.

        Args:
            step: Step number (1-based).
            theory: Working theory at this step; persists to future steps if blank.
            tool: Tool invoked.
            reasoning: Agent's reasoning for choosing this tool.
            entities: Entities observed or extracted.
            findings: Structured findings from the tool result.
            highlights: Notable items surfaced (anomalies, key values, etc.).
            recall_key: Optional key for stored result retrieval.
        """
        if theory:
            self.current_theory = theory
        self.entries.append(LogEntry(
            step=step,
            theory=theory or self.current_theory,
            tool=tool,
            reasoning=reasoning,
            entities_seen=entities or [],
            findings=findings or [],
            highlights=highlights or [],
            recall_key=recall_key,
        ))

    def render(self) -> str:
        """Render the full session log for prompt context inclusion."""
        if not self.entries:
            return ""

        lines = [f"═══ {self._title} ═══"]

        if self.current_theory:
            lines.append(f"Current theory: {self.current_theory}")
            lines.append("")

        all_entities = self.all_entities()

        if all_entities and self._entity_formatter:
            entity_lines = self._entity_formatter(all_entities)
            lines.extend(entity_lines)
            lines.append("")
        elif all_entities:
            sample = sorted(all_entities)[:20]
            lines.append(f"Entities discovered: {', '.join(sample)}")
            lines.append("")

        lines.append("Steps:")
        for entry in self.entries:
            lines.append(entry.render(highlights_label=self._highlights_label))

        return "\n".join(lines)

    def all_entities(self) -> set[str]:
        """Return all entities discovered across all steps, including highlights."""
        entities: set[str] = set()
        for e in self.entries:
            entities.update(e.entities_seen)
            entities.update(e.highlights)
        return entities

    def render_compact(self, entries: list[LogEntry] | None = None) -> str:
        """Build a deterministic compact summary from tracked data.

        No LLM call. Extracts structured metadata: entity set, theory
        progression, tool usage histogram, key findings, highlights,
        and a compact step timeline.
        """
        target = entries if entries is not None else self.entries
        if not target:
            return ""

        all_ents: set[str] = set()
        all_highlights: set[str] = set()
        for e in target:
            all_ents.update(e.entities_seen)
            all_highlights.update(e.highlights)
        all_ents.update(all_highlights)

        tool_counts: dict[str, int] = {}
        for e in target:
            if e.tool == "__summary__":
                continue
            tool_counts[e.tool] = tool_counts.get(e.tool, 0) + 1

        theories: list[str] = []
        for e in target:
            if e.theory and (not theories or e.theory != theories[-1]):
                theories.append(e.theory)

        findings: list[str] = []
        seen_findings: set[str] = set()
        for e in target:
            for f in e.findings:
                if f not in seen_findings:
                    seen_findings.add(f)
                    findings.append(f)

        parts: list[str] = []
        step_range = f"steps {target[0].step}-{target[-1].step}"
        parts.append(f"[Compressed {step_range}, {len(target)} entries]")

        if tool_counts:
            tool_str = ", ".join(
                f"{t}×{c}" for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])
            )
            parts.append(f"Tools used: {tool_str}")

        if theories:
            parts.append(f"Theories: {' → '.join(theories[:5])}")

        if all_highlights:
            parts.append(f"{self._highlights_label}: {', '.join(sorted(all_highlights)[:10])}")

        if all_ents - all_highlights:
            other = sorted(all_ents - all_highlights)[:15]
            parts.append(f"Entities: {', '.join(other)}")

        if findings:
            parts.append("Key findings:")
            for f in findings[:8]:
                parts.append(f"  → {f[:120]}")

        parts.append("Timeline:")
        for e in target:
            if e.tool == "__summary__":
                continue
            line = f"  S{e.step}: {e.tool}"
            if e.highlights:
                line += f" [{', '.join(e.highlights[:3])}]"
            elif e.entities_seen:
                line += f" ({len(e.entities_seen)} entities)"
            parts.append(line)

        return "\n".join(parts)

    def compact(
        self,
        max_entries_to_keep: int = 5,
        must_preserve: Callable[[LogEntry], bool] | None = None,
    ) -> bool:
        """Compact old entries into a deterministic summary in-place.

        Returns True if compaction occurred, False if not needed.
        No LLM call — uses render_compact() for the summary.

        Args:
            max_entries_to_keep: Number of recent entries to preserve verbatim.
            must_preserve: Optional predicate. Entries for which this returns
                True are kept verbatim even if they would otherwise be
                compacted.
        """
        if len(self.entries) <= max_entries_to_keep:
            return False

        old_entries = self.entries[:-max_entries_to_keep]
        recent_entries = self.entries[-max_entries_to_keep:]

        to_compress = [e for e in old_entries if e.tool != "__summary__"]
        if len(to_compress) < 3:
            return False

        existing_summaries = [e for e in old_entries if e.tool == "__summary__"]

        preserved: list[LogEntry] = []
        if must_preserve is not None:
            still_compress = []
            for e in to_compress:
                if must_preserve(e):
                    preserved.append(e)
                else:
                    still_compress.append(e)
            to_compress = still_compress
            if len(to_compress) < 3 and not preserved:
                return False

        summary_text = self.render_compact(to_compress) if to_compress else ""

        compressed_entities: list[str] = []
        for e in to_compress:
            compressed_entities.extend(e.entities_seen)
            compressed_entities.extend(e.highlights)
        for e in existing_summaries:
            compressed_entities.extend(e.entities_seen)

        all_findings: list[str] = []
        for e in existing_summaries:
            all_findings.extend(e.findings)
        if summary_text:
            all_findings.append(summary_text)

        new_entries: list[LogEntry] = list(existing_summaries)
        if to_compress:
            summary_entry = LogEntry(
                step=0,
                theory="",
                tool="__summary__",
                reasoning=f"[Compressed steps {to_compress[0].step}-{to_compress[-1].step}]",
                entities_seen=sorted(set(compressed_entities)),
                findings=all_findings,
            )
            new_entries.append(summary_entry)
        new_entries.extend(preserved)
        new_entries.extend(recent_entries)
        self.entries = new_entries
        return True

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize entries for state snapshot."""
        return [
            {
                "step": e.step,
                "theory": e.theory,
                "tool": e.tool,
                "reasoning": e.reasoning,
                "entities_seen": e.entities_seen,
                "findings": e.findings,
                "highlights": e.highlights,
                "recall_key": e.recall_key,
            }
            for e in self.entries
        ]
