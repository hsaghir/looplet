"""Wire the shared greeting log into the greet tool's module global.

Demonstrates the setup.py escape hatch: declarative loading covers
99% of cases, and setup.py handles the last 1% — runtime mutations
that need real Python (here: injecting a shared resource into a
top-level tool function's module-global slot).

Most workspaces don't need a setup.py. This one uses it solely so
the greet tool can mutate the same GreetingLog the PolitenessGate
hook reads — without setup.py the tool would fall back to its
``GREETING_LOG = None`` default and politeness checking would be
disabled.
"""


def setup(preset, resources, tool_modules, hook_modules):
    log = resources.get("greeting_log")
    if log is None:
        return preset
    greet_module = tool_modules.get("greet")
    if greet_module is not None:
        greet_module.GREETING_LOG = log
    return preset
