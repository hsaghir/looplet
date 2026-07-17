"""RegistryGuard - a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP).
The host ships only the declared view (``tool`` + ``args``) over
line-delimited JSON-RPC; this server returns a permission decision.

Policy: refuse any ``check_package`` or ``find_alternatives`` call whose
``package_name`` argument is empty or whitespace. Querying a package
registry with a blank name wastes a tool budget step and never returns
useful data, so the guard short-circuits it before dispatch.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a declarative
``kind: lep`` block - no Python source needs to be vendored beyond this
self-contained server.
"""

from looplet.lep import LEPServerBase

_GUARDED = {"check_package", "find_alternatives"}


class RegistryGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") in _GUARDED:
            name = (view.get("args") or {}).get("package_name")
            if not isinstance(name, str) or not name.strip():
                return {
                    "kind": "Deny",
                    "block": "refusing a registry lookup with an empty package_name",
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(RegistryGuardServer().serve())
