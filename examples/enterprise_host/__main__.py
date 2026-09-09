"""Run one Looplet cartridge through the reference enterprise host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from looplet import (
    MockLLMBackend,
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    RunEnvelope,
    SkillRuntime,
    load_skill_bundle,
    run_skill_bundle,
    validate_skill_bundle,
)


class _PolicyHook(PermissionHook):
    """Reference host policy: allow reads, deny destructive shell commands."""

    def __init__(self) -> None:
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny(
            "bash",
            arg_matcher=lambda args: any(
                token in str(args.get("command", "")) for token in ("rm -rf", "shutdown", "mkfs")
            ),
            reason="destructive command blocked by enterprise policy",
        )
        super().__init__(engine)


def _scripted_responses(bundle: Any) -> list[str]:
    responses = getattr(bundle.module, "scripted_responses", None)
    if not callable(responses):
        raise ValueError("--scripted requires a cartridge module with scripted_responses()")
    return list(responses())


def run_host(
    *,
    cartridge: Path,
    workspace: Path,
    task: str,
    run_id: str,
    tenant_id: str,
    actor_id: str,
    scripted: bool,
) -> dict[str, Any]:
    root = workspace / ".looplet" / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    runtime = SkillRuntime(
        workspace=workspace,
        max_steps=20,
        output_dir=root / "output",
        options={"eval_hook": False, "require_tests": False, "use_native_tools": False},
    )
    bundle = load_skill_bundle(cartridge)
    validation = validate_skill_bundle(bundle, runtime)
    if not validation.ok:
        raise ValueError("invalid cartridge: " + "; ".join(validation.errors))
    validation.close()
    envelope = RunEnvelope(
        run_id=run_id,
        request_id=f"request-{run_id}",
        tenant_id=tenant_id,
        actor_id=actor_id,
        deployment="looplet-enterprise-host-reference",
        cartridge_version=str(bundle.skill.metadata.get("version") or bundle.skill.name),
        policy_version="reference-v1",
        trace_id=f"trace-{run_id}",
    )
    preset = bundle.build_preset(runtime)
    preset.config.run_envelope = envelope
    preset.config.checkpoint_dir = str(root / "checkpoints")
    llm = MockLLMBackend(responses=_scripted_responses(bundle)) if scripted else None
    if llm is None:
        raise ValueError("the reference host currently requires --scripted")
    try:
        steps = list(
            run_skill_bundle(
                bundle,
                llm=llm,
                task=task,
                runtime=runtime,
                extra_hooks=[_PolicyHook()],
                provenance=True,
                trace_dir=root / "traces",
                preset=preset,
            )
        )
        return {
            "run_envelope": envelope.to_dict(),
            "status": getattr(preset.state, "run_status", None),
            "termination_reason": getattr(preset.state, "termination_reason", None),
            "steps": len(steps),
            "policy_decisions": list(
                getattr(preset.state, "metadata", {}).get("policy_decisions", [])
            ),
            "checkpoint_dir": str(root / "checkpoints"),
            "trace_dir": str(root / "traces"),
        }
    finally:
        preset.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cartridge", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", default="reference-run-1")
    parser.add_argument("--tenant-id", default="reference-tenant")
    parser.add_argument("--actor-id", default="reference-actor")
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args(argv)
    args.workspace.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_host(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
