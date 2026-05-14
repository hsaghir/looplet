"""setup.py is rejected by spec v2.

Spec v1.x accepted this with a DeprecationWarning. Spec v2 hard-fails:
the loader raises CartridgeSerializationError naming the file. This
file exists deliberately to verify that rejection.
"""


def setup(preset, resources, **kwargs):  # pragma: no cover - never invoked
    return preset
