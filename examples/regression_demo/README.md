# Failure → regression contract

This network-free demo holds the model's decisions constant while one
reviewable harness line changes:

1. A scripted backend calls `publish_report(revenue=120, cost=80)` and
   then `done()`.
2. The v1 tool writes the wrong profit (`revenue + cost`).
3. A host-side collector reads the resulting `report.json`; a required
   grader fails.
4. Looplet replays the **captured responses** through the fixed v2 tool
   in a fresh workspace.
5. The same collector and grader pass.

Run it from the repository root:

```bash
uv run python examples/regression_demo/run_demo.py
```

Expected core output:

```text
1. CAPTURE v1 with fixed model responses
   model decisions: publish_report -> done
   collected profit: 200
   required eval: FAIL (0.00)

2. CHANGE one reviewable harness line
   - "profit": revenue + cost,
   + "profit": revenue - cost,

3. REPLAY captured responses with fresh v2 tool execution
   same decisions: true
   collected profit: 40
   required eval: PASS (1.00)
```

The generated evidence directory contains both cartridge versions,
fresh workspaces, the v1 model-call cassette, trajectories, independent
outcome artifacts, grader results, and grader-only expected data.

## What this proves

- A model call can be captured once and inspected as ordinary files.
- Captured model responses can be re-executed against changed tools,
  hooks, state, and permissions without another model call.
- Collectors can grade world state rather than trusting the agent's
  claimed result or preferred trajectory.
- Required graders can turn the observed failure into a CI contract.

## What this does not prove

Captured-response replay is **not deterministic replay**. It fixes the
recorded model responses, but tools execute again. Clocks, networks,
randomness, and other side effects remain fresh unless you isolate or
mock them. It also does not measure whether a prompt change would cause
the model to choose better actions; that requires new sampled runs.
