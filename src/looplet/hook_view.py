"""Capability-scoped *views* of loop state for out-of-process hooks.

The portability boundary for a hook is not its code but its **declared
view** of loop state (HOOK_CARTRIDGE_DESIGN.md §4, hazard H2). A hook
that runs out-of-process — over the Loop Effect Protocol (LEP) — never
touches live ``state``/``session_log`` objects; the host serialises only
the *subset of fields the hook subscribed to* and ships that subset over
the wire.

``ViewSpec`` is that subscription (a field whitelist + a fidelity knob).
``extract_view`` is the pure projection from the live loop objects to a
JSON-safe dict containing exactly the subscribed fields and nothing else.
Because the projection is explicit and total, two properties hold:

* **Soundness (H2).** A hook can only observe what it declared; a hook
  that reads outside its view cannot exist over the wire, so the view is
  an enforceable upper bound on the hook's read set.
* **Determinism.** ``extract_view`` is a pure function of its inputs, so
  the same loop position yields the same wire payload on every runtime —
  the precondition for behavioural round-trip equivalence (§5.3).

Field vocabulary (the only keys a ``ViewSpec`` may name):

    tool          the pending/just-run tool name (str | None)
    args          the tool call arguments (dict)
    reasoning     the model's stated reason for the call (str)
    tool_result   {data, error, duration_ms, warnings} of the result
    step          the current step number (int)
    transcript    session-log entries — counts under ``digest`` fidelity,
                  full serialised entries under ``full`` fidelity
    usage         provider token usage / cost, when surfaced on state
    state_digest  a small, stable digest of agent state (counts only)

``fidelity`` is ``"digest"`` (default — cheap, counts/sizes only) or
``"full"`` (ship entry bodies; needed by hooks that legitimately read
the transcript, at the cost of bandwidth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ViewSpec", "extract_view", "KNOWN_VIEW_FIELDS"]

#: Every field name a ``ViewSpec`` is allowed to subscribe to. Naming an
#: unknown field is a configuration error (caught at load), not a silent
#: empty — that is what keeps the view an *enforceable* contract.
KNOWN_VIEW_FIELDS: frozenset[str] = frozenset(
    {
        "tool",
        "args",
        "reasoning",
        "tool_result",
        "step",
        "transcript",
        "usage",
        "state_digest",
    }
)


@dataclass(frozen=True)
class ViewSpec:
    """A hook's declared subscription to loop state (§4).

    Attributes:
        fields: The subset of :data:`KNOWN_VIEW_FIELDS` the hook may
            observe. The empty set means "no loop state at all" — valid
            for hooks that decide purely from the event slot.
        fidelity: ``"digest"`` (counts/sizes only — the default) or
            ``"full"`` (ship entry bodies for ``transcript``).
    """

    fields: frozenset[str] = field(default_factory=frozenset)
    fidelity: str = "digest"

    def __post_init__(self) -> None:
        unknown = set(self.fields) - KNOWN_VIEW_FIELDS
        if unknown:
            raise ValueError(
                f"ViewSpec names unknown field(s) {sorted(unknown)}; "
                f"allowed: {sorted(KNOWN_VIEW_FIELDS)}"
            )
        if self.fidelity not in ("digest", "full"):
            raise ValueError(f"ViewSpec.fidelity must be 'digest' or 'full', got {self.fidelity!r}")

    @classmethod
    def from_dict(cls, raw: Any) -> "ViewSpec":
        """Build a ``ViewSpec`` from a cartridge ``view:`` block.

        Accepts ``{"fields": [...], "fidelity": "digest"}`` or a bare
        list of field names. ``None`` yields the empty (no-state) view.
        """
        if raw is None:
            return cls()
        if isinstance(raw, (list, tuple, set, frozenset)):
            return cls(fields=frozenset(str(f) for f in raw))
        if isinstance(raw, dict):
            fields = raw.get("fields") or []
            if not isinstance(fields, (list, tuple, set, frozenset)):
                raise ValueError("view.fields must be a list of field names")
            return cls(
                fields=frozenset(str(f) for f in fields),
                fidelity=str(raw.get("fidelity", "digest")),
            )
        raise ValueError(f"unrecognised view spec: {raw!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a cartridge ``view:`` block (round-trip)."""
        return {"fields": sorted(self.fields), "fidelity": self.fidelity}


def _json_safe(value: Any) -> Any:
    """Best-effort coercion of arbitrary tool data to a JSON-safe value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _result_view(tool_result: Any) -> dict[str, Any] | None:
    if tool_result is None:
        return None
    return {
        "data": _json_safe(getattr(tool_result, "data", None)),
        "error": getattr(tool_result, "error", None),
        "duration_ms": getattr(tool_result, "duration_ms", 0.0),
        "warnings": list(getattr(tool_result, "warnings", []) or []),
    }


def _transcript_view(session_log: Any, *, full: bool) -> Any:
    entries = list(getattr(session_log, "entries", None) or [])
    if not full:
        return {"entry_count": len(entries)}
    out: list[dict[str, Any]] = []
    for entry in entries:
        to_dict = getattr(entry, "to_dict", None)
        if callable(to_dict):
            try:
                out.append(_json_safe(to_dict()))
                continue
            except Exception:  # pragma: no cover - defensive
                pass
        out.append({"repr": str(entry)})
    return out


def _state_digest(state: Any) -> dict[str, Any]:
    """A small, stable digest of agent state — counts only, never bodies."""
    digest: dict[str, Any] = {}
    entities = getattr(state, "entities", None)
    if entities is not None:
        try:
            digest["entity_count"] = len(entities)
        except TypeError:  # pragma: no cover - defensive
            pass
    return digest


def extract_view(
    spec: ViewSpec,
    *,
    state: Any = None,
    session_log: Any = None,
    tool_call: Any = None,
    tool_result: Any = None,
    step: int | None = None,
    usage: Any = None,
) -> dict[str, Any]:
    """Project the live loop objects onto the hook's declared ``spec``.

    Returns a JSON-safe dict whose keys are **exactly** the subscribed
    fields that are applicable at this call site. Fields the hook did not
    subscribe to are never included — even if the data is available — so
    the wire payload is a faithful witness of the hook's read set.
    """
    view: dict[str, Any] = {}
    full = spec.fidelity == "full"

    if "tool" in spec.fields:
        view["tool"] = getattr(tool_call, "tool", None) if tool_call is not None else None
    if "args" in spec.fields:
        view["args"] = (
            _json_safe(getattr(tool_call, "args", {}) or {}) if tool_call is not None else {}
        )
    if "reasoning" in spec.fields:
        view["reasoning"] = getattr(tool_call, "reasoning", "") if tool_call is not None else ""
    if "tool_result" in spec.fields:
        view["tool_result"] = _result_view(tool_result)
    if "step" in spec.fields:
        view["step"] = step
    if "transcript" in spec.fields:
        view["transcript"] = _transcript_view(session_log, full=full)
    if "usage" in spec.fields:
        if usage is None and state is not None:
            # Fall back to usage surfaced on state by the loop, so a hook
            # that only declares a `usage` view (e.g. a budget/cost hook)
            # still sees token usage at slots other than POST_LLM_RESPONSE.
            meta = getattr(state, "metadata", None)
            if isinstance(meta, dict):
                usage = meta.get("usage_total") or meta.get("last_usage")
        view["usage"] = _json_safe(usage)
    if "state_digest" in spec.fields:
        view["state_digest"] = _state_digest(state)

    return view
