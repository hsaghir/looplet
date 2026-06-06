"""CalcGuard — a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP) and
guards the ``add`` tool that is itself served by a *separate* MCP stdio
process (see ``_server/calc.py``). It demonstrates that LEP permission
policies and MCP transport tools compose cleanly: the host ships the
declared view (``tool`` + ``args``) to this server, which vets the call
before it is ever dispatched to the MCP server.

Policy: refuse any ``add`` call whose ``a`` or ``b`` operand is missing
or not a number. The calculator server's schema requires two integers,
so a non-numeric operand would fault the MCP process — the guard turns
that into a clean, early permission denial instead.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a declarative
``kind: lep`` block.
"""

from looplet.lep import LEPServerBase


def _is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
            return True
        except (TypeError, ValueError):
            return False
    return False


class CalcGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "add":
            args = view.get("args") or {}
            if not (_is_number(args.get("a")) and _is_number(args.get("b"))):
                return {
                    "kind": "Deny",
                    "block": "add requires two numeric operands 'a' and 'b'",
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(CalcGuardServer().serve())
