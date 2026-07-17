"""PolitenessGate - ported from an in-process Python hook CLASS to a
portable ``kind: lep`` hook.

The original ``hooks/00_PolitenessGate/hook.py`` read the live, shared
``@greeting_log`` instance directly from the resource registry - that
shared-address-space dependency is what pinned it to a Python host. Here
the hook runs OUT OF PROCESS over the Loop Effect Protocol (LEP) and
reaches the same greeting log through the **State Service**: it connects
to the socket the loader exported as ``LOOPLET_STATE_GREETING_LOG`` and
asks the service for its current count.

Because the greet MCP tool writes to that very same state service, this
hook sees the greetings the tool recorded - cross-process composition
that exactly reproduces the in-process ``@ref`` sharing of the original,
with zero shared Python objects.

Policy (``check_done`` slot): refuse ``done()`` until at least one
greeting has been recorded.
"""

import os

from looplet.lep import LEPServerBase
from looplet.state_service import StateServiceClient

_LOG_CLIENT: StateServiceClient | None = None


def _log():
    global _LOG_CLIENT
    if _LOG_CLIENT is not None:
        return _LOG_CLIENT
    socket_path = os.environ.get("LOOPLET_STATE_GREETING_LOG")
    if not socket_path:
        return None
    try:
        _LOG_CLIENT = StateServiceClient(socket_path)
    except Exception:  # noqa: BLE001
        _LOG_CLIENT = None
    return _LOG_CLIENT


class PolitenessGateServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_done":
            client = _log()
            recorded = 0
            if client is not None:
                try:
                    recorded = int(client.count())
                except Exception:  # noqa: BLE001
                    recorded = 0
            if recorded == 0:
                return {
                    "kind": "Block",
                    "block": (
                        "Politeness check failed: greet at least one person before calling done()."
                    ),
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(PolitenessGateServer().serve())
