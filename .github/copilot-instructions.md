# looplet Design Principles

Preserve looplet's center of gravity: a minimal, simple, powerful, familiar
Python library for owning and regression-testing an LLM tool-calling harness.

## The Simple Story

Use this mental model when explaining or changing looplet:

1. The LLM proposes a tool call.
2. The registry validates and dispatches it.
3. Hooks observe or steer the loop.
4. State records each step.
5. The loop yields a `Step` back to the caller.

Everything else should compile into that story.

## Design Guardrails

- **Minimal core:** keep `composable_loop(...)` small and domain-neutral. New capabilities should live in tools, hooks, presets, skills, bundles, CLI helpers, or docs unless the loop itself truly needs to know.
- **Simple defaults:** make the first path teachable in minutes. Prefer progressive disclosure over adding required concepts to the beginner path.
- **Powerful composition:** grow power by composing plain tools, hooks, config, state, memory sources, provenance, and cartridges. Avoid closed monoliths or hidden orchestration.
- **Familiar Python:** prefer normal functions, classes, dataclasses, protocols, iterators, and importable factories. Avoid DSLs, magic globals, mandatory inheritance, and dependency-heavy plugin systems.
- **Layered cartridges:** skills and cartridges are packaging/distribution layers over looplet primitives, not a second agent runtime. A cartridge should build or wrap an `AgentPreset` whenever possible.
- **Honest conversion:** do not pretend to decompile arbitrary Python. Exact wrappers and blueprint comparisons are reliable; expanded source generation should depend on explicit recorded recipes.
- **Observable behavior:** preserve inspectable `Step` streams, deterministic tests, and provenance paths. Debug output should remain close to eval artifacts.
- **Outcome-grounded evidence:** prefer collectors that inspect world state
	over graders that require one historical tool sequence.
- **Protected promotion oracles:** cartridge evals are self-tests. Keep
	holdouts host-owned whenever a candidate can edit its cartridge.
- **Captured-response replay:** recorded model responses are held constant; fresh tools
	and side effects still execute. Never call this deterministic replay.
- **Zero-runtime-dependency bias:** keep the core package dependency-free unless there is a compelling, user-visible reason to change that.
