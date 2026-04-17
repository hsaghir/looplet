"""Agent scaffolding — domain-agnostic loop mechanics.

Provides the sophisticated scaffolding that makes agentic loops effective:
  - LLM retry with exponential backoff
  - Parse error recovery (re-prompt on malformed JSON)
  - Diminishing returns detection (auto-stop on spinning)
  - Tool result truncation at capture (prevent context blow-up)
  - Context overflow detection and compression

Each mechanism is independent and stateless — pure functions operating
on agent state and loop counters.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────

MAX_LLM_RETRIES = 2
RETRY_BACKOFF_BASE = 1.0  # seconds
PARSE_RECOVERY_MAX = 2  # max consecutive parse recovery attempts
TOOL_RESULT_MAX_CHARS = 6000  # truncate tool result data
TOOL_RESULT_MAX_ROWS = 50  # max rows per result
DIMINISHING_RETURNS_WINDOW = 5  # steps to track
DIMINISHING_RETURNS_THRESHOLD = 0  # new items in window to be "diminishing"

RESULT_BUDGET_PER_RESULT = 50_000   # 50K chars per individual result
RESULT_BUDGET_AGGREGATE = 500_000   # 500K chars total across all results



# ── LLM Error Types ──────────────────────────────────────────────

# Strings that indicate a prompt-too-long error in common LLM APIs
_PROMPT_TOO_LONG_MARKERS = (
    "prompt is too long",
    "prompt_too_long",
    "context_length_exceeded",
    "maximum context length",
    "token limit",
    "too many tokens",
    "input is too long",
    "request too large",
    "413",
)


def _is_prompt_too_long(error: Exception) -> bool:
    """Check if an LLM error indicates the prompt exceeded context window."""
    msg = str(error).lower()
    return any(marker in msg for marker in _PROMPT_TOO_LONG_MARKERS)


class LLMResult:
    """Result of an LLM call with error discrimination.

    Uses __slots__ for minimal memory overhead on the hot path.
    ``ok`` is True iff the LLM returned a non-None response.
    ``is_prompt_too_long`` is computed once in __init__ so callers
    can branch without re-checking the error string.

    ``text`` is normally a plain ``str``. When the backend is invoked
    via ``generate_with_tools`` (native tool-calling path), it may be
    a ``list`` of provider-native content blocks (e.g. Anthropic's
    ``[{"type": "text"|"tool_use", ...}, ...]``). Callers on the native
    path should branch on ``isinstance(result.text, list)``.
    """

    __slots__ = ("text", "error", "is_prompt_too_long")

    def __init__(
        self,
        text: "str | list[Any] | None",
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.is_prompt_too_long = error is not None and _is_prompt_too_long(error)

    @property
    def ok(self) -> bool:
        """True if the call succeeded and text is available."""
        return self.text is not None


# ── Token Estimation ─────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Estimate token count using 4 chars-per-token heuristic.

    Returns at least 1 to avoid zero-token edge cases.
    """
    return max(1, len(text) // 4)


def estimate_prompt_tokens(text: str) -> int:
    """Alias for estimate_tokens."""
    return estimate_tokens(text)


# ── LLM Retry with Backoff ──────────────────────────────────────


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Return True iff ``fn`` declares ``name`` as a parameter."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters


_ACCEPTS_CT_CACHE: dict[tuple[type, str], bool] = {}


def _backend_accepts_cancel_token(backend: Any, method_name: str) -> bool:
    """Check if ``getattr(backend, method_name)`` accepts ``cancel_token``.

    Cached by ``(type(backend), method_name)`` so the ``inspect.signature``
    cost is paid once per backend *class*, not per call. Safe across GC
    cycles because we key on the class, not ``id()`` of a transient bound
    method.
    """
    key = (type(backend), method_name)
    hit = _ACCEPTS_CT_CACHE.get(key)
    if hit is None:
        fn = getattr(backend, method_name, None)
        hit = _accepts_kwarg(fn, "cancel_token") if fn is not None else False
        _ACCEPTS_CT_CACHE[key] = hit
    return hit


def llm_call_with_retry(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int = 2000,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_retries: int = MAX_LLM_RETRIES,
    tools: list[dict[str, Any]] | None = None,
    cancel_token: Any | None = None,
) -> LLMResult:
    """Call LLM with exponential backoff retry on failure.

    Returns LLMResult with error discrimination:
      - result.ok: True if successful
      - result.text: response string (None on failure); when ``tools`` is
        provided and the backend supports ``generate_with_tools``, this is a
        list of Anthropic-style content blocks instead of a string.
      - result.is_prompt_too_long: True if prompt exceeded context window

    Prompt-too-long errors are not retried — retrying the same prompt
    against the same context window will always fail.

    When ``tools`` is provided and the backend exposes ``generate_with_tools``,
    native tool calling is used; otherwise the call falls back to ``generate``
    (plain text → JSON-text tool parsing upstream).

    When ``cancel_token`` is provided:
      * If already cancelled before the call, returns an error result
        without invoking the backend.
      * If the backend's ``generate`` / ``generate_with_tools`` declares a
        ``cancel_token`` parameter, the token is forwarded so the backend
        can interrupt HTTP streams. Backends without the parameter keep
        working unchanged.
    """
    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
        return LLMResult(None, RuntimeError("cancelled before LLM call"))

    use_native = tools is not None and hasattr(llm, "generate_with_tools")
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        # Re-check cancellation between retries — a long backoff could span it.
        if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
            return LLMResult(None, RuntimeError("cancelled during retry backoff"))
        try:
            if use_native:
                call = llm.generate_with_tools
                extra_kwargs: dict[str, Any] = {}
                if cancel_token is not None and _backend_accepts_cancel_token(llm, "generate_with_tools"):
                    extra_kwargs["cancel_token"] = cancel_token
                blocks = call(
                    prompt,
                    tools=tools,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **extra_kwargs,
                )
                return LLMResult(blocks)
            call = llm.generate
            extra_kwargs = {}
            if cancel_token is not None and _backend_accepts_cancel_token(llm, "generate"):
                extra_kwargs["cancel_token"] = cancel_token
            text = call(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                **extra_kwargs,
            )
            return LLMResult(text)
        except Exception as e:
            last_error = e
            if _is_prompt_too_long(e):
                logger.warning("Prompt too long (not retrying): %s", e)
                return LLMResult(None, e)
            if attempt < max_retries:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries + 1, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    max_retries + 1, last_error,
                )
    return LLMResult(None, last_error)


# ── Parse Error Recovery ─────────────────────────────────────────

PARSE_RECOVERY_PROMPT = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Please respond with ONLY a valid JSON object, no extra text:\n"
    '{{"tool": "<tool_name>", "args": {{...}}, "reasoning": "..."}}\n\n'
    "Previous (unparseable) response:\n{raw}\n\n"
    "Please try again with valid JSON only."
)


def build_parse_recovery_prompt(original_prompt: str, raw_response: str) -> str:
    """Build a recovery prompt after a parse failure.

    Appends instructions and a truncated copy of the bad response to
    help the LLM understand what went wrong.
    """
    return original_prompt + PARSE_RECOVERY_PROMPT.format(
        raw=raw_response[:300],
    )


# ── Diminishing Returns Detection ────────────────────────────────


class DiminishingReturnsTracker:
    """Track agent productivity to detect spinning.

    Records new-item counts per step.  When the rolling window shows
    no improvement, ``is_diminishing`` is True and the loop should
    consider stopping or changing strategy.
    """

    def __init__(self, window: int = DIMINISHING_RETURNS_WINDOW) -> None:
        self.window = window
        self._history: list[int] = []
        self.consecutive_empty = 0

    def record_step(
        self,
        new_findings: int = 0,
        new_highlights: int = 0,
        new_entities: int = 0,
        *,
        new_items: int | None = None,
    ) -> None:
        """Record a completed step.

        Pass ``new_items`` directly, or use the named keyword args
        (new_findings, new_highlights, new_entities) which are summed.
        """
        total_new = new_items if new_items is not None else (new_findings + new_highlights + new_entities)
        self._history.append(total_new)
        if total_new == 0:
            self.consecutive_empty += 1
        else:
            self.consecutive_empty = 0

    @property
    def is_diminishing(self) -> bool:
        """True when the last ``window`` steps all produced zero new items."""
        if len(self._history) < self.window:
            return False
        return sum(self._history[-self.window:]) <= DIMINISHING_RETURNS_THRESHOLD

    @property
    def total_steps(self) -> int:
        """Total steps recorded."""
        return len(self._history)

    def guidance_text(self) -> str:
        """Return a guidance string for the LLM when stagnating, else empty str."""
        if self.consecutive_empty >= 5:
            return (
                "\n⚠ CRITICAL STAGNATION: %d consecutive steps with zero new information. "
                "You MUST either:\n"
                "  1. Call done() with your current results\n"
                "  2. Try a COMPLETELY different approach\n"
                "Repeating the same pattern will not help." % self.consecutive_empty
            )
        if self.consecutive_empty >= 3:
            return (
                "\n⚠ DIMINISHING RETURNS: Last %d steps found nothing new. "
                "Consider: (1) trying a different approach, "
                "(2) using a different tool, "
                "(3) calling done if you have enough results." % self.consecutive_empty
            )
        return ""


# ── Step Progress Tracking ───────────────────────────────────────


def _normalize_call(tool_name: str, tool_args: dict) -> str:
    """Deterministic key for a tool call (for dedup detection)."""
    clean = sorted(
        (k, str(v)) for k, v in tool_args.items() if not k.startswith("__")
    )
    return f"{tool_name}:{clean}"


class StepProgressTracker:
    """Per-step classification and dedup detection.

    Domain-agnostic. Classifies each step based on whether new
    information was found and whether the same tool+args were called
    before. Provides raw data — domain-specific guidance is generated
    by hooks, not by this tracker.

    Also provides backward-compatibility with ``DiminishingReturnsTracker``
    API so existing code can migrate incrementally.
    """

    PRODUCTIVE = "productive"
    REDUNDANT = "redundant"  # same tool+args called before, no new info
    EMPTY = "empty"          # new call, no new info
    ERROR = "error"          # tool returned error

    def __init__(self, window: int = DIMINISHING_RETURNS_WINDOW) -> None:
        self._window = window
        self._seen_calls: dict[str, int] = {}  # call_key → step_number
        self._classifications: list[str] = []
        self._new_items_history: list[int] = []
        self._consecutive_unproductive: int = 0

    def record_call(self, tool_name: str, tool_args: dict, step_num: int) -> None:
        """Record a tool call for dedup tracking (call before classify_turn)."""
        key = _normalize_call(tool_name, tool_args)
        self._seen_calls[key] = step_num

    def check_seen(self, tool_name: str, tool_args: dict) -> int | None:
        """If this exact call was made before, return its step number."""
        return self._seen_calls.get(_normalize_call(tool_name, tool_args))

    def classify_turn(self, new_items: int, step_num: int) -> str:
        """Classify a completed turn.  Returns classification constant."""
        self._new_items_history.append(new_items)

        if new_items > 0:
            cls = self.PRODUCTIVE
        else:
            cls = self.EMPTY

        self._classifications.append(cls)
        if cls == self.PRODUCTIVE:
            self._consecutive_unproductive = 0
        else:
            self._consecutive_unproductive += 1

        return cls

    def mark_redundant(self, tool_name: str, tool_args: dict, step_num: int) -> None:
        """Mark a specific tool call as redundant (called by hook on dedup detection)."""
        if self._classifications:
            self._classifications[-1] = self.REDUNDANT

    @property
    def consecutive_unproductive(self) -> int:
        """Number of consecutive unproductive steps."""
        return self._consecutive_unproductive

    @property
    def is_stagnating(self) -> bool:
        """True when the last ``window`` steps all produced zero new items."""
        if len(self._new_items_history) < self._window:
            return False
        return sum(self._new_items_history[-self._window:]) <= DIMINISHING_RETURNS_THRESHOLD

    @property
    def redundant_count(self) -> int:
        """Total number of steps classified as redundant."""
        return sum(1 for c in self._classifications if c == self.REDUNDANT)

    @property
    def total_steps(self) -> int:
        """Total steps classified."""
        return len(self._classifications)

    # ── Backward-compat bridge for DiminishingReturnsTracker API ──

    @property
    def consecutive_empty(self) -> int:
        """Backward-compat: consecutive steps with no new info."""
        return self._consecutive_unproductive

    @property
    def is_diminishing(self) -> bool:
        """Backward-compat: alias for is_stagnating."""
        return self.is_stagnating

    def record_step(
        self,
        new_findings: int = 0,
        new_highlights: int = 0,
        new_entities: int = 0,
        *,
        new_items: int | None = None,
    ) -> None:
        """Backward-compat: record a step like DiminishingReturnsTracker."""
        total = new_items if new_items is not None else (new_findings + new_highlights + new_entities)
        self.classify_turn(total, self.total_steps + 1)

    def guidance_text(self) -> str:
        """Backward-compat: same interface as DiminishingReturnsTracker."""
        if self._consecutive_unproductive >= 5:
            return (
                "\n⚠ CRITICAL STAGNATION: %d consecutive steps with no new information. "
                "You MUST change your approach completely or call done()."
                % self._consecutive_unproductive
            )
        if self._consecutive_unproductive >= 3:
            return (
                "\n⚠ DIMINISHING RETURNS: Last %d steps found nothing new. "
                "Consider changing your approach or calling done()."
                % self._consecutive_unproductive
            )
        return ""


# ── Tool Result Truncation ───────────────────────────────────────


def truncate_tool_result(
    data: Any,
    max_chars: int = TOOL_RESULT_MAX_CHARS,
    max_rows: int = TOOL_RESULT_MAX_ROWS,
) -> Any:
    """Truncate tool result data to prevent context blow-up.

    When truncating, adds metadata so the LLM knows data was cut:
    total count, shown count, and guidance on how to get more.
    """
    if data is None:
        return data

    if isinstance(data, list):
        if len(data) > max_rows:
            return {
                "rows": data[:max_rows],
                "total": len(data),
                "showing": max_rows,
                "truncated": True,
                "note": f"Showing {max_rows} of {len(data)} items. Add filters to narrow results.",
            }
        return data

    if isinstance(data, dict):
        result = {}
        truncated = False
        for key, val in data.items():
            if key == "rows" and isinstance(val, list) and len(val) > max_rows:
                result[key] = val[:max_rows]
                result["showing"] = max_rows
                if "total" not in data:
                    result["total"] = len(val)
                truncated = True
            elif key == "per_table" and isinstance(val, list):
                new_per_table = []
                for entry in val:
                    if (
                        isinstance(entry, dict)
                        and isinstance(entry.get("rows"), list)
                        and len(entry["rows"]) > max_rows
                    ):
                        new_entry = {**entry, "rows": entry["rows"][:max_rows]}
                        new_entry["showing"] = max_rows
                        new_entry["total_in_table"] = entry.get("count", len(entry["rows"]))
                        truncated = True
                        new_per_table.append(new_entry)
                    else:
                        new_per_table.append(entry)
                result[key] = new_per_table
            else:
                result[key] = val
        if truncated and "note" not in result:
            result["note"] = "Some results truncated. Add filters to narrow results."
        return result

    if isinstance(data, str) and len(data) > max_chars:
        return data[:max_chars] + f"\n... [truncated, {len(data)} chars total]"

    return data


# ── Context Overflow Detection ───────────────────────────────────


def should_compress_context(
    prompt: str,
    context_window: int = 128_000,
    threshold: float = 0.75,
) -> bool:
    """True when the prompt exceeds ``threshold`` fraction of the context window."""
    return estimate_tokens(prompt) > context_window * threshold


def compress_session_log(
    session_log: Any,
    llm: Any = None,
    max_entries_to_keep: int = 5,
    must_preserve: Any = None,
) -> str | None:
    """Compact old session log entries.

    Two modes:
      1. Deterministic (llm=None): Uses SessionLog.compact() to build a
         structured summary from tracked data (entities, findings, tools,
         timeline). Fast, free, reliable.
      2. LLM-refined (llm provided): First builds the deterministic summary,
         then asks the LLM to refine it into a more coherent narrative.
         Costs one LLM call but produces better summaries for complex sessions.

    Args:
        session_log: A SessionLog instance (or compatible object with compact()).
        llm: Optional LLM backend for refinement.
        max_entries_to_keep: How many recent entries to keep verbatim.
        must_preserve: Optional callable(LogEntry) -> bool. Entries for which
            this returns True are kept verbatim even when compacted.

    Returns the summary text, or None if no compaction was needed.
    """
    if not hasattr(session_log, "compact"):
        return None
    if not session_log.compact(
        max_entries_to_keep=max_entries_to_keep,
        must_preserve=must_preserve,
    ):
        return None

    deterministic_summary = None
    for e in session_log.entries:
        if e.tool == "__summary__":
            deterministic_summary = "\n\n".join(e.findings) if e.findings else "compacted"
            break

    if deterministic_summary is None:
        return None

    if llm is not None:
        try:
            prompt = (
                "Refine this structured summary. Respond with ONLY text "
                "(no tool calls).\n\n"
                "First draft your thinking in <analysis> tags, then produce "
                "the final summary in <summary> tags.\n\n"
                "In the <summary>, preserve ALL entities, key findings, and "
                "highlights. Do not add information not in the input.\n\n"
                f"{deterministic_summary}"
            )
            refined = llm.generate(prompt, max_tokens=400, temperature=0.1)
            if refined and refined.strip():
                clean = re.sub(r"<analysis>.*?</analysis>", "", refined, flags=re.DOTALL)
                summary_match = re.search(r"<summary>(.*?)</summary>", clean, flags=re.DOTALL)
                if summary_match:
                    clean = summary_match.group(1).strip()
                else:
                    clean = clean.strip()
                if clean:
                    for e in session_log.entries:
                        if e.tool == "__summary__":
                            e.findings = [clean]
                            break
                    return clean
        except Exception as ex:
            logger.warning("LLM refinement failed, keeping deterministic summary: %s", ex)

    return deterministic_summary


# ── Result Budget Enforcement ────────────────────────────────────


def enforce_result_budget(
    steps: list,
    per_result_chars: int = RESULT_BUDGET_PER_RESULT,
    aggregate_chars: int = RESULT_BUDGET_AGGREGATE,
) -> None:
    """Enforce per-result and per-message aggregate budgets on step data.

    Modifies steps in place. Oversized results are replaced with
    compact summaries that preserve the recall key for retrieval.

    Skips results already compacted (marked with ``__compacted__`` key)
    to avoid progressive degradation.
    """
    def _is_compacted(data: Any) -> bool:
        return bool(isinstance(data, dict) and data.get("__compacted__"))

    # Phase 1: Per-result budget
    for step in steps:
        r = step.tool_result
        if r.data is None or r.error or _is_compacted(r.data):
            continue
        try:
            size = len(json.dumps(r.data, default=str))
        except (TypeError, ValueError):
            size = len(str(r.data))
        if size > per_result_chars:
            r.data = _compact_result(r.data, r.result_key, per_result_chars)

    # Phase 2: Aggregate budget
    total = 0
    sized: list[tuple[int, int]] = []
    for i, step in enumerate(steps):
        r = step.tool_result
        if r.data is None or r.error or _is_compacted(r.data):
            if r.data is not None and not r.error:
                try:
                    total += len(json.dumps(r.data, default=str))
                except (TypeError, ValueError):
                    total += len(str(r.data))
            continue
        try:
            size = len(json.dumps(r.data, default=str))
        except (TypeError, ValueError):
            size = len(str(r.data))
        sized.append((size, i))
        total += size

    if total > aggregate_chars:
        sized.sort(reverse=True)
        for size, idx in sized:
            if total <= aggregate_chars:
                break
            step = steps[idx]
            r = step.tool_result
            if r.data is None or _is_compacted(r.data):
                continue
            old_size = size
            r.data = _compact_result(r.data, r.result_key, 2000)
            try:
                new_size = len(json.dumps(r.data, default=str))
            except (TypeError, ValueError):
                new_size = len(str(r.data))
            total -= (old_size - new_size)


def _compact_result(data: Any, result_key: str | None, max_chars: int) -> Any:
    """Compact a result to max_chars, preserving key info.

    Always sets ``__compacted__: True`` so _is_compacted() recognises
    the result and skips it on subsequent budget passes (idempotent).
    """
    if isinstance(data, list):
        kept = data[:3]
        summary: dict[str, Any] = {
            "__compacted__": True,
            "original_count": len(data),
            "sample": kept,
        }
        if result_key:
            summary["recall_key"] = result_key
            summary["note"] = f"Full data available via result_key '{result_key}'"
        return summary

    if isinstance(data, dict):
        result: dict[str, Any] = {"__compacted__": True}
        budget = max_chars
        for key, val in data.items():
            if key == "__compacted__":
                continue  # skip old marker if present
            if key == "rows" and isinstance(val, list):
                result[key] = val[:3]
                result["total_rows"] = len(val)
                if result_key:
                    result["recall_key"] = result_key
            elif isinstance(val, (list, dict)):
                s = json.dumps(val, default=str)
                if len(s) > budget // 4:
                    result[key] = f"[{type(val).__name__}, {len(val) if isinstance(val, list) else len(str(val))} items — compacted]"
                else:
                    result[key] = val
            else:
                result[key] = val
        return result

    s = str(data)
    if len(s) > max_chars:
        return {"__compacted__": True, "summary": s[:max_chars], "total_chars": len(s)}
    return data


# ── Reactive Compaction ──────────────────────────────────────────


def reactive_compact(
    state: Any,
    session_log: Any,
    llm: Any = None,
    keep_recent: int = 3,
) -> str | None:
    """Emergency compaction when context is exhausted.

    More aggressive than compress_session_log: compresses ALL old entries
    (not just those above a threshold), ages old step results to None,
    and produces a single summary.

    Uses deterministic compaction — no LLM call. The ``llm`` parameter is
    kept for API compatibility but ignored.
    """
    if not hasattr(session_log, "entries") or len(session_log.entries) <= keep_recent:
        return None

    to_compress = [
        e for e in session_log.entries[:-keep_recent]
        if e.tool != "__summary__"
    ]
    if not to_compress:
        return None

    if hasattr(session_log, "compact"):
        if session_log.compact(max_entries_to_keep=keep_recent):
            if hasattr(state, "steps"):
                last_compressed_step = to_compress[-1].step
                for step in state.steps:
                    if step.number <= last_compressed_step:
                        step.tool_result.data = None
            for e in session_log.entries:
                if e.tool == "__summary__":
                    return "\n\n".join(e.findings) if e.findings else "compacted"
    return None
