# Looplet Enterprise Host Reference

This is a deliberately small host-side reference integration for enterprise deployments.
It owns request identity, policy, checkpoint/provenance directories, and process-level
lifecycle. Looplet remains the execution kernel: cartridge loading, tool dispatch,
lifecycle events, checkpoints, and provenance stay in Looplet.

Run a deterministic cartridge:

```bash
python -m examples.enterprise_host --cartridge examples/coder.cartridge \
  --workspace /tmp/looplet-enterprise-workspace \
  --task "Inspect the workspace and finish" --scripted
```

The host emits one JSON summary containing the run envelope, terminal status, step
count, policy decisions, and checkpoint/provenance locations.
