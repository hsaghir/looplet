"""Coder example — a production-grade looplet coding agent.

Pure-Python building blocks split across three modules so the same
code can be read by a developer, run as a CLI script, or shipped as
a runnable cartridge:

* :mod:`examples.coder.tools` — what the agent can do
  (``FileCache`` + the ``bash``/``read_file``/``write_file``/...
  tool definitions).
* :mod:`examples.coder.hooks` — how we observe and steer it
  (``TestGuardHook``, ``FileCacheHook``, ``StaleFileHook``,
  ``LinterHook``).
* :mod:`examples.coder.wiring` — how the pieces compose
  (``SYSTEM_PROMPT``, ``build_default_hooks``,
  ``build_default_memory_sources``, ``build_eval_hook``,
  ``scripted_responses``).

The library entrypoint :mod:`examples.coder.agent` and the cartridge
``examples/coder/skill/looplet.py`` both consume these modules
directly — that is the parity guarantee. No magic, just imports.
"""

from __future__ import annotations
