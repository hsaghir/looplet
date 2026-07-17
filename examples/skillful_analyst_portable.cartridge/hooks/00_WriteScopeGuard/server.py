"""WriteScopeGuard - a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP).
The host ships only the declared view (``tool`` + ``args``) over
line-delimited JSON-RPC; this server returns a permission decision.

Policy: refuse any ``write_text`` call whose ``path`` is empty or
escapes the project root via a ``..`` parent-directory traversal. The
analyst writes its findings inside the workspace; a blank or traversing
path is almost always a mistake (or an attempt to clobber a file
outside the sandbox), so the guard blocks it before the write happens.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a declarative
``kind: lep`` block - no Python source needs to be vendored beyond this
self-contained server.
"""

from looplet.lep import LEPServerBase


class WriteScopeGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "write_text":
            path = (view.get("args") or {}).get("path")
            if not isinstance(path, str) or not path.strip():
                return {
                    "kind": "Deny",
                    "block": "refusing write_text with an empty path",
                }
            parts = path.replace("\\", "/").split("/")
            if ".." in parts:
                return {
                    "kind": "Deny",
                    "block": "refusing write_text with a '..' path escaping the project root",
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(WriteScopeGuardServer().serve())
