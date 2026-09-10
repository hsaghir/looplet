# Agent release-gate pilot

This pilot tests one product hypothesis before Looplet commits to an
agent-release-engineering direction:

> Does an independent outcome gate catch harness regressions that a
> trace-only gate would accept, without rejecting correct changes?

The pilot uses one captured report-agent run and replays the same model
responses through seven tool implementations:

- `control_correct`: the shipped implementation;
- `benign_formatting`: a behavior-preserving serialization change;
- five seeded regressions: wrong arithmetic, ignored cost, reversed sign,
  wrong output field, and a successful call that skips the artifact write.

The trace-only gate checks only what an ordinary trace/reliability check can
see: the loop completed, the expected tools were called, and no tool returned
an error. The outcome gate uses the host-side collector and required grader to
inspect `report.json` independently.

## Run

```bash
uv run python benchmarks/agent_release_gate_pilot/run_pilot.py
```

Use `--out /tmp/looplet-agent-release-gate-pilot` to choose an output
directory. The run is network-free and uses a scripted backend only to hold
model decisions constant. It writes:

```text
pilot.json                 # machine-readable per-variant results
REPORT.md                  # rendered result and interpretation
runs/<variant>/            # versioned Looplet evidence bundles
workspaces/<variant>/      # fresh side-effect directories
cartridges/<variant>/      # exact harness variant under test
```

## Interpretation

The pilot supports the release-gate hypothesis if:

1. `control_correct` and `benign_formatting` pass both gates;
2. every seeded regression passes the trace-only gate;
3. every seeded regression fails the independent outcome gate;
4. each failed outcome has inspectable evidence identifying the observed
   world-state mismatch.

This is a capability pilot, not evidence of general model quality. It does
not measure fresh sampling, user pain, diagnosis time with humans, or the
rate of real regressions in production. Those are the next experiments if
this controlled result is positive.