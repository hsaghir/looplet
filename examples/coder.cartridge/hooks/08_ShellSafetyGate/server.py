"""ShellSafetyGate - a portable ``kind: lep`` permission policy.

A catastrophic-command guardrail for the coder agent that runs *out of
process* over the Loop Effect Protocol (LEP). The host ships only the
declared view (``tool`` + ``args``); this server inspects the proposed
``bash`` command and denies a small, conservative set of irreversibly
destructive patterns (root/home wipes, raw-disk writes, filesystem
formats, fork bombs, world-writable recursion on ``/``). Everything else
is allowed - ordinary coding shell usage is untouched.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a self-contained
``kind: lep`` block alongside the cartridge's in-process hooks
(TestGuard, FileCache, StaleFile, Linter, Eval) - demonstrating both the
out-of-process and in-process authoring styles in one cartridge.
"""

import re

from looplet.lep import LEPServerBase

# Conservative, high-confidence catastrophic patterns. Each is something
# a coding agent should never need and that is effectively irreversible.
_DANGEROUS = [
    re.compile(r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf][a-z]*\s+(-[a-z]*\s+)*(/|~|\$HOME)\s*($|\s)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:"),  # classic fork bomb
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b[^\n]*\bof=/dev/"),
    re.compile(r">\s*/dev/sd[a-z]\b"),
    re.compile(r"\bchmod\s+(-[a-z]*\s+)*-R\s+0*777\s+/\s*($|\s)"),
]


class ShellSafetyServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "bash":
            command = (view.get("args") or {}).get("command")
            if isinstance(command, str):
                for pattern in _DANGEROUS:
                    if pattern.search(command):
                        return {
                            "kind": "Deny",
                            "block": (
                                "shell-safety policy: refusing a catastrophic "
                                f"command ({pattern.pattern!r})"
                            ),
                        }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(ShellSafetyServer().serve())
