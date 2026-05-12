"""Prompt-file hooks for Cartridge Spec v1.1.

Two extra prompt slots beyond ``prompts/system.md``:

* ``prompts/briefing.md`` — auto-prepended to every step's briefing
  section. Use for short, persistent reminders that should appear in
  every prompt.
* ``prompts/recovery.md`` — injected as ``InjectContext`` on the
  prompt that follows a tool error. Use for general remediation
  guidance the agent should consult when something goes wrong.

The cartridge loader reads these files at load time and instantiates
the appropriate hook; hosts and second runtimes can reuse the same
hooks against any cartridge-loaded text.
"""

from __future__ import annotations

from looplet import InjectContext


class StaticBriefingHook:
    """Auto-prepend a fixed briefing text to every step's prompt.

    Reads from ``prompts/briefing.md`` at cartridge load time. The
    text is appended to the briefing section via ``pre_prompt``. Other
    hooks may add their own briefing output; all are concatenated.

    Args:
        text: The briefing body to inject. Empty string disables the
            hook (it returns ``None`` so the briefing section stays
            unchanged).
    """

    def __init__(self, *, text: str = "") -> None:
        self.text = (text or "").strip()

    def pre_prompt(self, state, session_log, context, step_num):
        if not self.text:
            return None
        return self.text

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        return None

    def check_done(self, state, session_log, context, step_num):
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class RecoveryHintHook:
    """Inject a recovery hint into the next prompt after a tool error.

    Reads from ``prompts/recovery.md`` at cartridge load time. The
    hint fires once after each errored tool result via
    ``InjectContext`` from ``post_dispatch``. Successful dispatches
    return ``None`` so the hint doesn't pollute clean steps.

    Args:
        text: The recovery body to inject. Empty string disables.
    """

    def __init__(self, *, text: str = "") -> None:
        self.text = (text or "").strip()

    def pre_prompt(self, state, session_log, context, step_num):
        return None

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if not self.text:
            return None
        if tool_result.error is None:
            return None
        return InjectContext(self.text)

    def check_done(self, state, session_log, context, step_num):
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


__all__ = ["RecoveryHintHook", "StaticBriefingHook"]
