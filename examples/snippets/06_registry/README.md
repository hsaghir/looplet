# 06 - Cartridge registry

Once an agent is a directory, *registries are filesystem operations*.
This snippet implements two registry verbs in pure stdlib Python:
`list` and `pull` (`git clone` of a cartridge from a git URL into a
target dir).

```bash
# List cartridges in the local repo.
uv run python examples/snippets/06_registry/registry.py list .

# "Pull" a cartridge by cloning a git repo's subdirectory.
uv run python examples/snippets/06_registry/registry.py pull \
    https://github.com/hsaghir/looplet.git \
    examples/threat_intel.cartridge \
    /tmp/threat_intel.cartridge
```

The point is not the script. It is that the script *can be 60 lines*
because the artifact has a known shape. Standardising the registry
manifest, signature scheme, and admission rules is then a community
project - exactly the path OCI took.
