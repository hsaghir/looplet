"""NameGuard - a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP).
The host never hands it live loop state; it ships only the declared
view (``tool`` + ``args``) over line-delimited JSON-RPC, and this
server returns a permission decision.

Policy: refuse any ``greet`` call whose ``name`` argument is empty or
whitespace. Everything else is allowed. Because the decision is a pure
function of the declared view, the hook is classified ``portable`` and
round-trips losslessly as a declarative ``kind: lep`` block - no Python
source needs to be vendored beyond this self-contained server.

The complementary ``00_PolitenessGate`` hook stays in-process: it reads
a shared greeting-log instance (live state) that cannot cross the wire.
Together they demonstrate both authoring styles in one cartridge.
"""

from looplet.lep import LEPServerBase


class NameGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "greet":
            name = (view.get("args") or {}).get("name")
            if not isinstance(name, str) or not name.strip():
                return {"kind": "Deny", "block": "refusing to greet an empty name"}
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(NameGuardServer().serve())
