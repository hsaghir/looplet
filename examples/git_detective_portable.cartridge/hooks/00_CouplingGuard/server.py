"""CouplingGuard — a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP).
The host ships only the declared view (``tool`` + ``args``) over
line-delimited JSON-RPC; this server returns a permission decision.

Policy: refuse any ``coupled_files`` call whose ``min_coupling``
argument does not parse to a positive integer. The tool treats a
non-positive or non-numeric threshold as "no filter", which floods the
agent with every co-changed pair and burns its read budget — so the
guard rejects the call and asks for a sane threshold instead.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a declarative
``kind: lep`` block — no Python source needs to be vendored beyond this
self-contained server.
"""

from looplet.lep import LEPServerBase


class CouplingGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "coupled_files":
            raw = (view.get("args") or {}).get("min_coupling", "3")
            try:
                threshold = int(str(raw).strip())
            except (TypeError, ValueError):
                threshold = 0
            if threshold < 1:
                return {
                    "kind": "Deny",
                    "block": 'min_coupling must be a positive integer (e.g. "3")',
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(CouplingGuardServer().serve())
