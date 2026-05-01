"""``python -m looplet`` — CLI entry point.

Subcommands:
    show <trace-dir>              One-page summary of a captured trace directory.
    doctor                        Check local looplet/backend configuration.
    run <bundle> <task>           Run a runnable skill bundle.
    blueprint <bundle>            Print a bundle blueprint as JSON.
    export-code <bundle> <file>   Export a bundle as Python wrapper code.
    package <factory> <dir>       Package an importable factory as a bundle.
    wrap-claude-skill <src> <dir> Wrap a Claude Skill as a looplet bundle.
    list-bundles <roots...>       List runnable bundles under one or more roots.
    eval <args...>                Run evals or browse cases (see `looplet eval -h`).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

from looplet import __version__


def _fmt_ms(ms: float | int | None) -> str:
    if ms is None:
        return "   -  "
    return f"{int(ms):>5}ms"


def _render_show(trace_dir: Path) -> int:
    if not trace_dir.exists():
        print(f"error: {trace_dir} does not exist", file=sys.stderr)
        return 1

    # ── trajectory.json (optional; short-circuit if missing) ─────
    traj_path = trace_dir / "trajectory.json"
    manifest_path = trace_dir / "manifest.jsonl"
    traj: dict[str, Any] = {}
    if traj_path.exists():
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"error: could not parse {traj_path}: {exc}", file=sys.stderr)
            return 1

    # ── manifest.jsonl (optional) ────────────────────────────────
    calls: list[dict[str, Any]] = []
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except Exception:
                continue

    if not traj and not calls:
        print(
            f"error: {trace_dir} contains no trajectory.json or "
            "manifest.jsonl — not a trace directory",
            file=sys.stderr,
        )
        return 1

    # ── Header ──────────────────────────────────────────────────
    run_id = traj.get("run_id") or trace_dir.name
    term = traj.get("termination_reason", "?")
    term_glyph = {"done": "✓", "error": "✗"}.get(term, "·")
    step_count = traj.get("step_count", len(traj.get("steps", [])))
    llm_count = traj.get("llm_call_count", len(calls))
    # Total duration: sum step durations if available, else call durations.
    total_ms = sum(s.get("duration_ms", 0) for s in traj.get("steps", []))
    if total_ms == 0 and calls:
        total_ms = sum(c.get("duration_ms", 0) for c in calls)

    print(
        f"{run_id}  {term_glyph} {term}  "
        f"{step_count} steps  {llm_count} LLM calls  {int(total_ms)}ms"
    )
    print()

    # ── Steps ───────────────────────────────────────────────────
    for s in traj.get("steps", []):
        num = s.get("step_num", "?")
        tc = s.get("tool_call", {}) or {}
        tr = s.get("tool_result", {}) or {}
        tool = tc.get("tool") or tc.get("action") or "?"
        args = s.get("args_summary") or tr.get("args") or ""
        err = tr.get("error") or s.get("error")
        ok = "✗" if err else "✓"
        data = tr.get("data")
        if err:
            tail = f"ERROR: {str(err)[:40]}"
        elif isinstance(data, list):
            tail = f"{len(data)} items"
        elif isinstance(data, dict):
            tail = f"{tr.get('total_items') or len(data)} keys"
        elif data is None:
            tail = ""
        else:
            snippet = str(data)
            tail = snippet if len(snippet) <= 30 else snippet[:27] + "..."
        dur = _fmt_ms(s.get("duration_ms"))
        linked = s.get("llm_call_indices") or []
        link_str = f"call {linked[0]}" if linked else ""
        print(f"#{num}  {ok} {tool}({str(args)[:30]:<30}) → {tail:<20} [{dur}] {link_str}")

    # ── LLM summary ─────────────────────────────────────────────
    if calls:
        total_prompt = sum(c.get("prompt_chars") or 0 for c in calls)
        total_resp = sum(c.get("response_chars") or 0 for c in calls)
        errors = sum(1 for c in calls if c.get("error"))
        print()
        print(
            f"LLM: {len(calls)} calls, "
            f"{total_prompt:,} in / {total_resp:,} out chars, "
            f"{errors} errors"
        )

    # ── Failure modes (if present) ──────────────────────────────
    modes = traj.get("failure_modes") or []
    if modes:
        print()
        for fm in modes:
            print(f"!! {fm}")

    return 0


def _status_line(status: str, name: str, detail: str) -> str:
    marks = {"ok": "OK", "warn": "WARN", "error": "ERROR"}
    return f"{marks.get(status, '?')} {name}: {detail}"


def _doctor_checks(*, probe_backend: bool) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    py_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "name": "python",
            "status": "ok" if py_ok else "error",
            "detail": platform.python_version() + (" (>=3.11)" if py_ok else " (<3.11)"),
        }
    )
    checks.append({"name": "looplet", "status": "ok", "detail": f"version {__version__}"})

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("OPENAI_MODEL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        checks.append(
            {
                "name": "OPENAI_BASE_URL",
                "status": "warn",
                "detail": "not set; set it to probe an OpenAI-compatible backend",
            }
        )
    else:
        checks.append({"name": "OPENAI_BASE_URL", "status": "ok", "detail": base_url})
    checks.append(
        {
            "name": "OPENAI_MODEL",
            "status": "ok" if model else "warn",
            "detail": model or "not set",
        }
    )
    checks.append(
        {
            "name": "OPENAI_API_KEY",
            "status": "ok" if api_key else "warn",
            "detail": "set" if api_key else "not set (local endpoints often accept 'x')",
        }
    )

    if not probe_backend:
        checks.append(
            {"name": "backend_probe", "status": "ok", "detail": "skipped by --no-backend"}
        )
        return checks
    if not base_url or not model:
        checks.append(
            {
                "name": "backend_probe",
                "status": "warn",
                "detail": "skipped; OPENAI_BASE_URL and OPENAI_MODEL are required",
            }
        )
        return checks

    try:
        from looplet.backends import OpenAIBackend  # noqa: PLC0415
        from looplet.native_tools import probe_native_tool_support  # noqa: PLC0415

        llm = OpenAIBackend(base_url=base_url, api_key=api_key or "x", model=model)
        probe = probe_native_tool_support(llm)
        checks.append(
            {
                "name": "native_tools",
                "status": "ok" if probe.supported else "warn",
                "detail": probe.reason,
            }
        )
        if not probe.supported:
            checks.append(
                {
                    "name": "tool_protocol",
                    "status": "ok",
                    "detail": "use LoopConfig(use_native_tools=False) or probe before enabling native tools",
                }
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            {
                "name": "backend_probe",
                "status": "warn",
                "detail": f"could not probe backend: {type(exc).__name__}: {exc}",
            }
        )
    return checks


def _render_doctor(*, probe_backend: bool, json_output: bool, strict: bool) -> int:
    checks = _doctor_checks(probe_backend=probe_backend)
    if json_output:
        print(json.dumps({"checks": checks}, indent=2))
    else:
        print("looplet doctor")
        print()
        for check in checks:
            print(_status_line(check["status"], check["name"], check["detail"]))
    bad = [
        check
        for check in checks
        if check["status"] == "error" or (strict and check["status"] == "warn")
    ]
    return 1 if bad else 0


def _render_run(
    *,
    bundle_path: Path,
    task: str,
    workspace: Path,
    max_steps: int,
    scripted: bool,
    scripted_responses: list[str],
    require_tests: bool,
    trace_dir: Path | None,
    no_trace: bool,
) -> int:
    from looplet.backends import OpenAIBackend  # noqa: PLC0415
    from looplet.bundles import (  # noqa: PLC0415
        BundleValidation,
        SkillRuntime,
        load_skill_bundle,
        run_skill_bundle,
        validate_skill_bundle,
    )
    from looplet.native_tools import probe_native_tool_support  # noqa: PLC0415
    from looplet.resilient import ResilientBackend  # noqa: PLC0415
    from looplet.testing import MockLLMBackend  # noqa: PLC0415

    try:
        bundle = load_skill_bundle(bundle_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: invalid bundle {bundle_path}", file=sys.stderr)
        print(f"  - load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    effective_trace_dir = None
    if not no_trace:
        effective_trace_dir = trace_dir or (
            workspace.resolve()
            / ".looplet"
            / "traces"
            / f"{bundle.skill.name}-{uuid.uuid4().hex[:12]}"
        )

    def _validation_runtime(output_dir: Path | None) -> SkillRuntime:
        return SkillRuntime(
            workspace=workspace,
            max_steps=max_steps,
            options={"require_tests": require_tests, "use_native_tools": False},
            output_dir=output_dir,
        )

    def _report_invalid_bundle(validation: BundleValidation) -> None:
        print(f"error: invalid bundle {bundle_path}", file=sys.stderr)
        for error in validation.errors:
            print(f"  - {error}", file=sys.stderr)

    bundle_run = getattr(bundle.module, "run", None)
    scripted_mode = scripted or bool(scripted_responses)
    for index, response in enumerate(scripted_responses, start=1):
        if not response.strip():
            print(
                f"error: --scripted-response {index} must not be empty",
                file=sys.stderr,
            )
            return 1
    provider_validation: BundleValidation | None = None
    if scripted and not scripted_responses:
        provider = getattr(bundle.module, "scripted_responses", None)
        if callable(provider):
            provider_validation = validate_skill_bundle(
                bundle,
                _validation_runtime(
                    (None if no_trace else trace_dir)
                    if callable(bundle_run)
                    else effective_trace_dir,
                ),
            )
            if not provider_validation.ok:
                _report_invalid_bundle(provider_validation)
                return 1
            provider = cast(Callable[[], Iterable[str]], provider)
            provider_returned_string = False
            try:
                with bundle.import_context():
                    provided_responses = provider()
                    if isinstance(provided_responses, str):
                        provider_returned_string = True
                    else:
                        scripted_responses = list(provided_responses)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"error: bundle {bundle.skill.name!r} failed while loading scripted responses",
                    file=sys.stderr,
                )
                print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            if provider_returned_string:
                print(
                    f"error: bundle {bundle.skill.name!r} scripted_responses() must return "
                    "an iterable of response strings, got str",
                    file=sys.stderr,
                )
                return 1
            if not scripted_responses:
                print(
                    f"error: bundle {bundle.skill.name!r} scripted_responses() returned no responses",
                    file=sys.stderr,
                )
                return 1
            for index, response in enumerate(scripted_responses, start=1):
                if not isinstance(response, str):
                    print(
                        f"error: bundle {bundle.skill.name!r} scripted_responses() item "
                        f"{index} must be str, got {type(response).__name__}",
                        file=sys.stderr,
                    )
                    return 1
                if not response.strip():
                    print(
                        f"error: bundle {bundle.skill.name!r} scripted_responses() item "
                        f"{index} must not be empty",
                        file=sys.stderr,
                    )
                    return 1
        elif not callable(bundle_run):
            print(
                f"error: bundle {bundle.skill.name!r} does not provide scripted_responses()",
                file=sys.stderr,
            )
            return 1

    if callable(bundle_run):
        validation = provider_validation or validate_skill_bundle(
            bundle,
            _validation_runtime(None if no_trace else trace_dir),
        )
        if not validation.ok:
            _report_invalid_bundle(validation)
            return 1
        for warning in validation.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        bundle_run = cast(Callable[..., int], bundle_run)
        try:
            with bundle.import_context():
                result = bundle_run(
                    task=task,
                    workspace=workspace,
                    max_steps=max_steps,
                    scripted=scripted_mode,
                    scripted_responses=scripted_responses,
                    require_tests=require_tests,
                    trace_dir=trace_dir,
                    provenance=not no_trace,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"error: bundle {bundle.skill.name!r} failed while running", file=sys.stderr)
            print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if isinstance(result, bool) or not isinstance(result, int):
            print(f"error: bundle {bundle.skill.name!r} returned invalid status", file=sys.stderr)
            print(f"  - expected int exit code, got {type(result).__name__}", file=sys.stderr)
            return 1
        return result

    class _NativeToolFlag:
        enabled: bool
        used_as_bool: bool

        def __init__(self, enabled: bool = False) -> None:
            self.enabled = enabled
            self.used_as_bool = False

        def __bool__(self) -> bool:
            self.used_as_bool = True
            return self.enabled

    class _TrackingRuntime(SkillRuntime):
        accessed_options: set[str]
        native_tool_flag: _NativeToolFlag

        def __init__(
            self,
            *,
            workspace: Path,
            max_steps: int,
            options: dict[str, Any],
            output_dir: Path | None,
        ) -> None:
            super().__init__(
                workspace=workspace,
                max_steps=max_steps,
                options=options,
                output_dir=output_dir,
            )
            object.__setattr__(self, "accessed_options", set())
            object.__setattr__(self, "native_tool_flag", _NativeToolFlag(False))

        def option(self, name: str, default: Any = None) -> Any:
            self.accessed_options.add(name)
            if name == "use_native_tools":
                return self.native_tool_flag
            return super().option(name, default)

    runtime = _TrackingRuntime(
        workspace=workspace,
        max_steps=max_steps,
        options={"require_tests": require_tests, "use_native_tools": False},
        output_dir=effective_trace_dir,
    )
    validation = provider_validation or validate_skill_bundle(bundle, runtime)
    if not validation.ok:
        _report_invalid_bundle(validation)
        return 1

    if scripted_mode:
        llm = MockLLMBackend(responses=scripted_responses)
        model_label = "scripted MockLLMBackend"
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "x")
        model = os.environ.get("OPENAI_MODEL", "llama3.1")
        llm = ResilientBackend(
            OpenAIBackend(base_url=base_url, api_key=api_key, model=model),
            retries=2,
            timeout_s=120,
        )
        model_label = model

    protocol_probe = probe_native_tool_support(llm)
    if protocol_probe.supported and "use_native_tools" in runtime.accessed_options:
        runtime.native_tool_flag.enabled = True
        if validation.preset is not None:
            if validation.preset.config.use_native_tools is False:
                if runtime.native_tool_flag.used_as_bool:
                    validation.preset.config.use_native_tools = True
    for warning in validation.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    uses_native_protocol = bool(
        protocol_probe.supported
        and validation.preset is not None
        and validation.preset.config.use_native_tools
    )

    print(f"looplet run {bundle.skill.name}")
    print(f"  Task: {task}")
    print(f"  Workspace: {workspace}")
    print(f"  Model: {model_label} | Budget: {max_steps} steps")
    print(f"  Tool protocol: {'native' if uses_native_protocol else 'json-text'}")
    print(f"  Probe: {protocol_probe.reason}")
    print()

    render_step = getattr(bundle.module, "render_step", None)
    try:
        for step in run_skill_bundle(
            bundle,
            llm=llm,
            task=task,
            runtime=runtime,
            provenance=not no_trace,
            trace_dir=effective_trace_dir,
            preset=validation.preset,
        ):
            if callable(render_step):
                with bundle.import_context():
                    rendered = render_step(step)
                if rendered:
                    print(rendered)
            else:
                status = "ERROR" if step.tool_result.error else "ok"
                print(f"#{step.number} {step.tool_call.tool} {status}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: bundle {bundle.skill.name!r} failed while running", file=sys.stderr)
        print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if effective_trace_dir is not None:
        print(f"\n  Trace: {effective_trace_dir}")
    return 0


def _render_blueprint(*, bundle_path: Path, workspace: Path, max_steps: int) -> int:
    from looplet.blueprints import blueprint_from_bundle  # noqa: PLC0415
    from looplet.bundles import SkillRuntime  # noqa: PLC0415

    try:
        blueprint = blueprint_from_bundle(
            bundle_path,
            SkillRuntime(workspace=workspace, max_steps=max_steps),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not inspect bundle {bundle_path}", file=sys.stderr)
        print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(blueprint.to_dict(), indent=2, sort_keys=True))
    return 0


def _render_export_code(*, bundle_path: Path, out_file: Path, function_name: str) -> int:
    from looplet.blueprints import export_bundle_to_library_code  # noqa: PLC0415

    try:
        written = export_bundle_to_library_code(
            bundle_path,
            out_file,
            function_name=function_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not export bundle {bundle_path}", file=sys.stderr)
        print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"exported {bundle_path} -> {written}")
    return 0


def _render_package(
    *,
    factory_ref: str,
    out_dir: Path,
    name: str,
    description: str,
    tags: list[str],
) -> int:
    from looplet.blueprints import package_agent_factory_as_bundle  # noqa: PLC0415

    try:
        written = package_agent_factory_as_bundle(
            factory_ref,
            out_dir,
            name=name,
            description=description,
            tags=tags,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not package factory {factory_ref!r}", file=sys.stderr)
        print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"packaged {factory_ref} -> {written}")
    return 0


def _render_wrap_claude_skill(*, skill_path: Path, out_dir: Path) -> int:
    from looplet.blueprints import (  # noqa: PLC0415
        claude_skill_compatibility,
        wrap_claude_skill_as_bundle,
    )

    try:
        report = claude_skill_compatibility(skill_path)
        written = wrap_claude_skill_as_bundle(skill_path, out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not wrap Claude Skill {skill_path}", file=sys.stderr)
        print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"wrapped {skill_path} -> {written}")
    print(f"compatibility: {report.level}")
    if report.warnings:
        for warning in report.warnings:
            print(f"warning: {warning}")
    return 0


def _render_list_bundles(
    *,
    roots: list[Path],
    json_output: bool,
    include_invalid: bool,
) -> int:
    from looplet.bundles import discover_skill_bundles  # noqa: PLC0415

    try:
        cards = discover_skill_bundles(
            roots,
            include_invalid=include_invalid,
            on_duplicate="warn",
        )
    except Exception as exc:  # noqa: BLE001
        print("error: could not list bundles", file=sys.stderr)
        print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps([card.to_dict() for card in cards], indent=2, sort_keys=True))
        return 0
    for card in cards:
        status = "ok" if card.ok else "invalid"
        print(f"{card.name}\t{status}\t{card.path}\t{card.description}")
        for error in card.errors:
            print(f"  - {error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Short-circuit "eval" to its own CLI so flags like --help/-h reach
    # the eval parser instead of being captured by the top-level parser.
    # argparse.REMAINDER on a subparser does not protect option-like
    # tokens from the parent parser, so we route eval before parsing.
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] == "eval":
        from looplet.evals import eval_cli  # noqa: PLC0415

        return eval_cli(raw[1:])

    parser = argparse.ArgumentParser(
        prog="python -m looplet",
        description="looplet — run, inspect, and package observable agent loops",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser(
        "show",
        help="Show a one-page summary of a captured trace directory",
    )
    show.add_argument("trace_dir", type=Path, help="Path to a trace directory")

    doctor = sub.add_parser(
        "doctor",
        help="Check local looplet configuration and optional backend tool protocol",
    )
    doctor.add_argument(
        "--no-backend",
        action="store_true",
        help="Skip network/backend probing and only check local configuration",
    )
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero for warnings as well as errors",
    )

    run = sub.add_parser(
        "run",
        help="Run a runnable skill bundle",
    )
    run.add_argument("bundle", type=Path, help="Path to a runnable skill bundle")
    run.add_argument("task", help="Task description to pass to the bundle")
    run.add_argument("--workspace", "-w", type=Path, default=Path.cwd(), help="Workspace path")
    run.add_argument("--max-steps", type=int, default=20, help="Maximum tool calls")
    run.add_argument(
        "--scripted",
        action="store_true",
        help="Use bundle-provided deterministic scripted responses",
    )
    run.add_argument(
        "--scripted-response",
        action="append",
        default=[],
        help="Mock LLM response; pass multiple times for deterministic runs",
    )
    run.add_argument(
        "--no-tests",
        action="store_true",
        help="Pass require_tests=False to bundles that support it",
    )
    run.add_argument("--trace-dir", type=Path, help="Write provenance trace output here")
    run.add_argument("--no-trace", action="store_true", help="Disable default provenance capture")

    blueprint = sub.add_parser(
        "blueprint",
        help="Print a runnable skill bundle blueprint as JSON",
    )
    blueprint.add_argument("bundle", type=Path, help="Path to a runnable skill bundle")
    blueprint.add_argument(
        "--workspace", "-w", type=Path, default=Path.cwd(), help="Workspace path"
    )
    blueprint.add_argument("--max-steps", type=int, default=20, help="Maximum tool calls")

    export_code = sub.add_parser(
        "export-code",
        help="Export a bundle as exact Python library wrapper code",
    )
    export_code.add_argument("bundle", type=Path, help="Path to a runnable skill bundle")
    export_code.add_argument("out_file", type=Path, help="Python file to write")
    export_code.add_argument(
        "--function-name",
        default="build",
        help="Generated factory function name",
    )

    package = sub.add_parser(
        "package",
        help="Package an importable AgentPreset factory as a runnable skill bundle",
    )
    package.add_argument("factory_ref", help="Factory reference, e.g. my_agent:build")
    package.add_argument("out_dir", type=Path, help="Bundle directory to write")
    package.add_argument("--name", required=True, help="Skill name")
    package.add_argument("--description", required=True, help="Skill description")
    package.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Skill tag; pass multiple times for multiple tags",
    )

    wrap_claude = sub.add_parser(
        "wrap-claude-skill",
        help="Wrap a Claude/Agent Skills folder as a runnable looplet bundle",
    )
    wrap_claude.add_argument("skill", type=Path, help="Claude Skill directory or SKILL.md")
    wrap_claude.add_argument("out_dir", type=Path, help="Bundle directory to write")

    list_bundles = sub.add_parser(
        "list-bundles",
        help="List runnable skill bundles under one or more roots",
    )
    list_bundles.add_argument("roots", nargs="+", type=Path, help="Bundle roots to scan")
    list_bundles.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    list_bundles.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include bundle-like folders with missing or invalid entrypoints",
    )

    eval_cmd = sub.add_parser(
        "eval",
        help="Run evals or browse cases (see `looplet eval -h`).",
        add_help=False,
    )
    # Pre-routed in main() so --help/-h reach eval_cli; this argument is
    # only declared so help output mentions a positional payload.
    eval_cmd.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to looplet.evals.eval_cli",
    )

    args = parser.parse_args(argv)

    if args.command == "show":
        return _render_show(args.trace_dir)
    if args.command == "doctor":
        return _render_doctor(
            probe_backend=not args.no_backend,
            json_output=args.json,
            strict=args.strict,
        )
    if args.command == "run":
        return _render_run(
            bundle_path=args.bundle,
            task=args.task,
            workspace=args.workspace,
            max_steps=args.max_steps,
            scripted=args.scripted,
            scripted_responses=args.scripted_response,
            require_tests=not args.no_tests,
            trace_dir=args.trace_dir,
            no_trace=args.no_trace,
        )
    if args.command == "blueprint":
        return _render_blueprint(
            bundle_path=args.bundle,
            workspace=args.workspace,
            max_steps=args.max_steps,
        )
    if args.command == "export-code":
        return _render_export_code(
            bundle_path=args.bundle,
            out_file=args.out_file,
            function_name=args.function_name,
        )
    if args.command == "package":
        return _render_package(
            factory_ref=args.factory_ref,
            out_dir=args.out_dir,
            name=args.name,
            description=args.description,
            tags=args.tag,
        )
    if args.command == "wrap-claude-skill":
        return _render_wrap_claude_skill(skill_path=args.skill, out_dir=args.out_dir)
    if args.command == "list-bundles":
        return _render_list_bundles(
            roots=args.roots,
            json_output=args.json,
            include_invalid=args.include_invalid,
        )
    if args.command == "eval":
        # Pre-routed in main(); kept here only as a defensive fallback.
        from looplet.evals import eval_cli  # noqa: PLC0415

        return eval_cli(args.eval_args)
    # Unreachable — argparse rejects unknown commands.
    return 2


if __name__ == "__main__":
    sys.exit(main())
