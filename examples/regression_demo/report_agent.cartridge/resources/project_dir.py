"""Expose the eval runner's fresh project root to the write tool."""


def build(runtime=None):
    return (runtime or {}).get("project_root", ".")
