# Relaunch checklist

No launch date overrides a failed integrity or evidence gate.

## 1. Product and claim gates

- [ ] The canonical category is identical across README, docs home, PyPI
      description, release copy, and repository description.
- [ ] The primary CTA runs the network-free regression proof.
- [ ] Every outward factual claim appears in `claims-ledger.md`.
- [ ] No public copy says deterministic replay, automatic optimization,
      production-safe, or universal framework replacement.
- [ ] Raw loop, graph runtime, hosted eval platform, and turnkey agent
      alternatives are described honestly.
- [ ] Factory-generated cartridges are described as drafts, not validated
      production agents.

## 2. Executable proof gates

- [ ] Fresh checkout/install instructions succeed.
- [ ] `uv run python examples/regression_demo/run_demo.py` exits 0.
- [ ] Stable output shows 200/FAIL → one-line diff → 40/PASS.
- [ ] `tests/test_regression_demo.py` passes independently.
- [ ] Evidence contains v1/v2 cartridges and workspaces, v1 cassette,
      trajectories, artifacts, expected data, and eval results.
- [ ] The expected oracle is absent from the agent-visible task.
- [ ] Demo cleanup refuses to delete unrelated non-demo directories.
- [ ] Docs say tools and side effects execute again during replay.

## 3. Repository validation

- [ ] Ruff check and format check pass.
- [ ] Pyright passes.
- [ ] Full pytest suite passes on supported local interpreter.
- [ ] `make check` passes from a clean worktree.
- [ ] No runtime dependency was added.
- [ ] Only relaunch proof code was added; runtime core stayed unchanged unless a
      separately evidenced defect required it.
- [ ] Git diff contains no generated evidence, bytecode, secrets, or local
      environment paths.

## 4. Documentation validation

- [ ] `mkdocs build --strict` succeeds.
- [ ] Internal links and anchors resolve.
- [ ] README links resolve from GitHub and PyPI contexts.
- [ ] No stale 0.1.x positioning, test counts, "one dependency," or workspace
      class examples remain.
- [ ] Cartridge docs distinguish colocated self-tests from protected holdouts.
- [ ] Provenance docs consistently say captured-response replay.
- [ ] Roadmap keeps search, statistics, and optimization outside core.
- [ ] Code samples use public imports and canonical CLI names.

## 5. Visual validation

- [ ] Homepage inspected at desktop and narrow mobile widths.
- [ ] Light and dark color schemes preserve contrast and hierarchy.
- [ ] Terminal proof does not overflow its container.
- [ ] Workflow and fit panels stack cleanly on mobile.
- [ ] Focus indicators and buttons are keyboard-visible.
- [ ] Mermaid diagrams render in both schemes.
- [ ] Existing launch images are legible but are not used as evidence for the
      red-to-green claim.

## 6. GitHub/release setup

- [ ] Relaunch PR has a concise positioning summary and validation evidence.
- [ ] CI is green on every supported Python version and docs job.
- [ ] Hosted docs deploy and canonical proof URL is verified after merge.
- [ ] Repository About description and website URL are updated.
- [ ] Topic set reflects `agent-harness`, `agent-evals`, `python`, and
      `tool-calling` without keyword stuffing.
- [ ] Behavioral-regression issue form renders correctly.
- [ ] Planned issues were searched for duplicates and only launch-ready items
      were opened.
- [ ] Release/discussion copy uses live canonical URLs.

## 7. Publication

- [ ] Show HN post uses the recommended technical title and caveat.
- [ ] First comment links source and explains trace/replay/eval separation.
- [ ] Social copy is adapted per audience rather than pasted everywhere.
- [ ] No vote solicitation or coordinated engagement.
- [ ] Maintainer availability is reserved for technical questions and fixes.

## 8. Follow-up

- [ ] Repeated questions are captured in FAQ changes.
- [ ] Reproduced defects become cases/collectors/graders before feature work.
- [ ] Metrics record qualified proof/contract adoption, not only impressions.
- [ ] Claims ledger is updated when evidence changes.
- [ ] 7-day and 30-day reviews decide what to stop, not only what to add.
