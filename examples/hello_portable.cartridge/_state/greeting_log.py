"""Greeting log — ported from the in-process ``@ref`` resource to an
out-of-process **State Service** (State Service Protocol).

In the original ``hello`` cartridge this lived at
``resources/greeting_log.py`` as a plain ``GreetingLog`` class whose
single instance was shared (via ``"@greeting_log"`` refs) between the
greet tool and the PolitenessGate hook — both running in the SAME Python
process. That shared-address-space requirement is exactly what pinned
the cartridge to a Python host.

Here the same state lives in ITS OWN process behind a Unix socket. The
loader spawns this server, injects a :class:`StateServiceClient` proxy
under the name ``greeting_log`` (so ``requires: [greeting_log]`` and
``@greeting_log`` resolve to it unchanged), and exports the socket path
as ``LOOPLET_STATE_GREETING_LOG`` so the MCP greet tool and the LEP
PolitenessGate hook — each a separate process — can connect to and
mutate the SAME log. All method calls are serialized under one lock, so
concurrent readers/writers see a consistent record.

Note one mechanical difference from the in-process original: ``entries``
is a METHOD here (``entries()``), not an attribute, because callers reach
it through the client proxy which forwards method calls over the wire.
"""

from looplet.state_service import StateServiceBase


class GreetingLogService(StateServiceBase):
    """Append-only log of ``(name, greeting_text)`` pairs, served over SSP."""

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[list[str]] = []

    def record(self, name: str, text: str) -> None:
        """Append a greeting to the shared log."""
        self._entries.append([name, text])

    def names(self) -> list[str]:
        """Return the names greeted so far, in order."""
        return [n for n, _ in self._entries]

    def entries(self) -> list[list[str]]:
        """Return all ``[name, text]`` pairs recorded so far."""
        return [list(e) for e in self._entries]

    def count(self) -> int:
        """Return how many greetings have been recorded."""
        return len(self._entries)


if __name__ == "__main__":
    raise SystemExit(GreetingLogService().serve())
