# Looplet Enterprise Host Reference

This is a deliberately small host-side reference integration for enterprise deployments.
It owns request identity, policy, checkpoint/provenance directories, and process-level
lifecycle. Looplet remains the execution kernel: cartridge loading, tool dispatch,
lifecycle events, checkpoints, and provenance stay in Looplet.

From the repository, run the deterministic Python-factory bundle fixture:

```bash
python -m examples.enterprise_host --cartridge tests/fixtures/coder_skill_bundle \
  --workspace /tmp/looplet-enterprise-workspace \
  --task "Inspect the workspace and finish" --scripted
```

The host emits one JSON summary containing the run envelope, terminal status, step
count, policy decisions, and checkpoint/provenance locations. It also includes
the shared content-redacted `diagnostics()` view, redacted `config.explain()`
values, and resolved terminal names. Unknown provider cost remains `null`.
This host reference accepts Python-factory bundles; cartridge declaration
inspection is available separately through `looplet describe <path> --json`.
