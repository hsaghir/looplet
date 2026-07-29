# Private tool loop → outcome-grounded regression contract

A team that already owns a model/tool loop can adopt Looplet one boundary
at a time. This example evolves one tiny on-call handoff agent through
four stages with no provider, no network call, and no cartridge.

The agent has a bug worth catching: it hands resolved incidents to the
next owner as if they were still open work.

Run it from the repository root:

```bash
uv run python -m examples.private_loop_migration
```

Expected core output:

```text
1. REPLACE the loop, keep the private tools
   raw loop:        list_incidents -> publish_handoff -> done
   composable_loop: list_incidents -> publish_handoff -> done
   private callables that ran: list_incidents -> publish_handoff
   same tools reused:      true
   same handoff written:   true

2. CAPTURE the failing run as ordinary files
   model calls recorded: 3

3. GRADE the outcome the host observed, not the route
   collected open_count: 4 (expected 2)
   required eval: FAIL (0.00)

4. FIX one tool implementation and replay the recorded decisions
   - return list(incidents)
   + return [incident for incident in incidents if incident["status"] != "resolved"]
   replayed:                list_incidents -> publish_handoff -> done
   same recorded decisions: true
   collected open_count: 2 (expected 2)
   required eval: PASS (1.00)
```

The evidence directory holds a separate seeded workspace per stage, the
captured prompts and responses, both trajectories, the collected
artifacts, the grader-only expected data, and the grader results.

## Files

- `handoff_tools.py`: the tool callables the private harness already
  owned, plus the one buggy implementation the last stage replaces.
- `handoff_contract.py`: the case, the collector that reads the
  workspace, and the required grader.
- `run_recipe.py`: the four stages, the summary, and the CLI.

## What this proves

- A loop swap can be attributed: both loops dispatch the same callables
  through one `tools_from(...)` registry and write the same file.
- A failing run becomes ordinary files that a person can read.
- A required grader can read world state instead of a preferred tool
  sequence, so a model that takes another valid route still passes.
- Captured-response replay isolates a tool change from model variation.

## What this does not prove

Replay fixes the recorded model responses; the tools execute again, so
clocks, networks, and other side effects stay live unless you isolate
them. It says nothing about whether a prompt change would produce better
decisions, which needs new sampled runs.

The guide that walks each stage is [`docs/migrate.md`](../../docs/migrate.md),
and [`tests/test_private_loop_migration.py`](../../tests/test_private_loop_migration.py)
is the gate that keeps this example honest.
