"""PolitenessGate hook — refuses done() until at least one greeting
has been recorded in the shared greeting log.

Demonstrates @ref shared-resource composition: this hook reads the
SAME log instance the greet tool wrote to.
"""


class PolitenessGate:
    def __init__(self, *, log) -> None:
        # ``log`` arrives from the @ref-resolved resource registry —
        # the same instance the greet tool will mutate at runtime.
        self.log = log

    def to_config(self) -> dict:
        # Round-trip kwargs for preset_to_cartridge. The runtime log
        # instance becomes a @ref string so two hooks can share it.
        return {"log": "@greeting_log"}

    def check_done(self, state, session_log, context, step_num):
        if not self.log.entries:
            return "Politeness check failed: greet at least one person before calling done()."
        return None
