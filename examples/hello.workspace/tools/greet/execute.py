"""Greet tool — top-level function, no closures.

Mutates the shared ``greeting_log`` resource (passed by setup.py)
so other components can audit greetings later.
"""


def execute(*, name: str) -> dict:
    text = f"Hello, {name}!"
    # Resource access happens through the module global ``GREETING_LOG``
    # which setup.py wires from the shared registry.
    if GREETING_LOG is not None:
        GREETING_LOG.record(name, text)
    return {"greeting": text}


# Set by setup.py at workspace load time. Importing this module before
# load (e.g. in tests) leaves it None — the tool still works, just
# without recording.
GREETING_LOG = None
