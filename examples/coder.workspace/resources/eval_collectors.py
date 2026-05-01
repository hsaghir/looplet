"""Collectors for the coder workspace's EvalHook.

Re-uses the v1 cartridge's ``make_test_collector`` so the v2
workspace surfaces the same outcome-grounded artifacts. Reads
``runtime['workspace']`` for the project root.
"""

from examples.coder.wiring import make_test_collector


def build(runtime=None):
    runtime = runtime or {}
    workspace = str(runtime.get("workspace", "."))
    return [make_test_collector(workspace)]
