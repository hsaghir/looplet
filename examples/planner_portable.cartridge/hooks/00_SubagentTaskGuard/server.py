"""SubagentTaskGuard — a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP).
The host ships only the declared view (``tool`` + ``args``) over
line-delimited JSON-RPC; this server returns a permission decision.

Policy: refuse any ``subagent`` call whose ``task`` argument is empty or
whitespace. Spawning a child loop with no task wastes an entire nested
run and its token budget, so the guard blocks it before dispatch.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a declarative
``kind: lep`` block — no Python source needs to be vendored beyond this
self-contained server.
"""

from looplet.lep import LEPServerBase


class SubagentTaskGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "subagent":
            task = (view.get("args") or {}).get("task")
            if not isinstance(task, str) or not task.strip():
                return {
                    "kind": "Deny",
                    "block": "refusing to spawn a subagent with an empty task",
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(SubagentTaskGuardServer().serve())
