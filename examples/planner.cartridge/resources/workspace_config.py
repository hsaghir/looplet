"""Resolve the planner's host-owned project directory."""

from looplet.cartridge.runtime_helpers import default_workspace_config


def build(runtime=None):
    return default_workspace_config(runtime)
