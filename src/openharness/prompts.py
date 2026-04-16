"""Structured prompt assembly for tool-using agents.

Sections are in FIXED positions every turn so the LLM can reliably
find information in the same place across steps. This aids coherence
in multi-turn agent loops.

Section order (stable → volatile):
  1. TASK         — Task description (never changes)
  2. TOOLS        — Available tools (never changes)
  3. FACTS        — Established state facts (grows monotonically)
  4. SESSION      — Session log / memory (grows, compacts)
  5. ASSESSMENT   — Guidance, briefing (changes per step)
  6. RESULTS      — Last few steps of raw results (changes per step)
  7. STEP         — Step/budget counter + response instruction (changes per step)
"""

from __future__ import annotations

from typing import Any, Callable

_DEFAULT_HEADERS: dict[str, str] = {
    "task": "TASK",
    "tools": "TOOLS",
    "facts": "ESTABLISHED FACTS",
    "session": "SESSION LOG",
    "assessment": "ASSESSMENT",
    "results": "RECENT RESULTS",
    "step": "STEP",
}


def build_prompt(
    task: dict | None = None,
    tool_catalog: str = "",
    state_summary: dict | None = None,
    context_history: str = "",
    step_number: int = 1,
    max_steps: int = 15,
    session_log: str = "",
    briefing: str = "",
    *,
    render_facts: Callable[[dict], list[str]] | None = None,
    task_fields: list[str] | None = None,
    action_prompt: str = "What do you want to do next? Respond with JSON.",
    low_budget_warning: str = "⚠ LOW BUDGET — consolidate and prepare your conclusion.",
    section_headers: dict[str, str] | None = None,
) -> str:
    """Build the user prompt with stable section ordering.

    Args:
        task: The task description dict.
        tool_catalog: String description of available tools.
        state_summary: Current agent state facts dict.
        context_history: Recent step results as a string.
        step_number: Current step index.
        max_steps: Maximum steps allowed.
        session_log: Running session memory / log string.
        briefing: Per-step guidance text (from hooks).
        render_facts: Optional callable to render state_summary into
            FACTS lines. If None, uses generic dict rendering.
        task_fields: Optional list of task dict fields to display.
            If None, renders all non-empty scalar keys.
        action_prompt: The instruction text for the STEP section.
        low_budget_warning: Warning shown when budget_remaining <= 3.
        section_headers: Optional dict overriding section header labels.
            Keys: task, tools, facts, session, assessment, results, step.
    """
    if task is None:
        task = {}
    if state_summary is None:
        state_summary = {}

    headers = {**_DEFAULT_HEADERS, **(section_headers or {})}

    parts: list[str] = []

    # ── §1 TASK (stable — never changes) ────────────────────
    parts.append(f"═══ {headers['task']} ═══")
    if task_fields:
        for key in task_fields:
            val = task.get(key)
            if val:
                parts.append(f"{key}: {val}")
    else:
        for key, val in task.items():
            if val and isinstance(val, (str, int, float, bool)):
                parts.append(f"{key}: {val}")
    parts.append("")

    # ── §2 TOOLS (stable — never changes) ───────────────────
    parts.append(f"═══ {headers['tools']} ═══")
    parts.append(tool_catalog)
    parts.append("")

    # ── §3 ESTABLISHED FACTS (grows monotonically) ──────────
    if render_facts is not None:
        facts_lines = render_facts(state_summary)
    else:
        facts_lines = _render_facts_generic(state_summary)

    if facts_lines:
        parts.append(f"═══ {headers['facts']} ═══")
        parts.extend(facts_lines)
        parts.append("")

    # ── §4 SESSION LOG (grows, compacts) ────────────────────
    if session_log:
        parts.append(f"═══ {headers['session']} ═══")
        parts.append(session_log)
        parts.append("")

    # ── §5 ASSESSMENT (per-step guidance from hooks) ────────
    if briefing:
        parts.append(f"═══ {headers['assessment']} ═══")
        parts.append(briefing)
        parts.append("")

    # ── §6 RECENT RESULTS (volatile — last few raw results) ─
    if context_history and context_history.strip() != "(no steps taken yet)":
        parts.append(f"═══ {headers['results']} ═══")
        parts.append(context_history)
        parts.append("")

    # ── §7 STEP (counter + action instruction) ───────────────
    budget = state_summary.get("budget_remaining", "?")
    parts.append(f"═══ {headers['step']} {step_number}/{max_steps} — budget: {budget} steps remaining ═══")
    if isinstance(budget, int) and budget <= 3:
        parts.append(low_budget_warning)
    parts.append(action_prompt)

    return "\n".join(parts)


def _render_facts_generic(state_summary: dict) -> list[str]:
    """Render state_summary entries generically for the FACTS section.

    Renders lists of dicts as bulleted items, scalar counts as key-value pairs.
    Skips internal bookkeeping keys.
    """
    skip_keys = {"step_count", "budget_remaining", "task_id"}
    lines: list[str] = []

    for key, val in state_summary.items():
        if key in skip_keys or not val:
            continue
        if isinstance(val, list) and val:
            lines.append(f"{key} ({len(val)}):")
            for item in val[-10:]:
                if isinstance(item, dict):
                    desc = (
                        item.get("description")
                        or item.get("value")
                        or item.get("label")
                        or str(item)
                    )
                    src = ""
                    if item.get("source_step"):
                        src = f" [step {item['source_step']}]"
                    elif item.get("step"):
                        src = f" [step {item['step']}]"
                    lines.append(f"  • {desc}{src}")
                else:
                    lines.append(f"  • {item}")
        elif isinstance(val, (int, float)):
            if key.endswith("_count"):
                lines.append(f"{key}: {val}")

    return lines
