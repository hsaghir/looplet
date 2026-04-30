"""Wire shared resources into every tool module that needs them.

Most workspaces don't need a setup.py. The coder workspace uses one
to inject the shared ``workspace_config`` and ``file_cache`` resources
into the module globals of the top-level tool functions — the
declarative @ref pattern only resolves into hook constructor kwargs,
and tools accept their kwargs from the LLM's tool call, not from a
constructor.

After this runs:
  * Every tool's ``WORKSPACE_CONFIG`` global points at the shared
    workspace_config resource → bash/list_dir/read_file/write_file/
    edit_file/glob/grep all see the same workspace path.
  * Every tool's ``FILE_CACHE`` global points at the shared file_cache
    resource → read_file/write_file/edit_file all coordinate with
    StaleFileHook + FileCacheHook through one cache instance.
"""


def setup(preset, resources, tool_modules, hook_modules):
    workspace_config = resources.get("workspace_config")
    file_cache = resources.get("file_cache")

    for tool_name, module in tool_modules.items():
        if workspace_config is not None and hasattr(module, "WORKSPACE_CONFIG"):
            module.WORKSPACE_CONFIG = workspace_config
        if file_cache is not None and hasattr(module, "FILE_CACHE"):
            module.FILE_CACHE = file_cache
    return preset
