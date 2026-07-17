"""Shared greeting log - every greeting is appended here so hooks
that audit politeness can inspect the same record the tool wrote.

Demonstrates the @ref shared-resource pattern: the greet tool and
the PolitenessGate hook both reference ``"@greeting_log"`` in their
config.yaml and get the SAME list instance - without this, they'd
each get an independent empty list and audit logic would silently
break on workspace reload.
"""


class GreetingLog:
    """Append-only log of (name, greeting_text) tuples."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def record(self, name: str, text: str) -> None:
        self.entries.append((name, text))

    def names(self) -> list[str]:
        return [n for n, _ in self.entries]


def build():
    return GreetingLog()
