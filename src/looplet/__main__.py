"""``python -m looplet`` - CLI entry point.

Subcommands:
    show <trace-dir>              One-page summary of a captured trace directory.
    replay <trace-dir>            Replay captured responses with a trusted cartridge.
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
import hashlib
import json
import math
import platform
import shlex
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO, cast

from looplet import __version__
from looplet.artifact_compat import read_artifact_descriptor
from looplet.cli import add_run_options, completion_exit_code, completion_payload, execution_output


def _fmt_ms(ms: float | int | None) -> str:
    if ms is None:
        return "   -  "
    return f"{int(ms):>5}ms"


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _trace_shape_error(
    trajectory: dict[str, Any],
    calls: list[dict[str, Any]],
) -> str | None:
    for field in ("run_id", "termination_reason"):
        value = trajectory.get(field)
        if value is not None and not isinstance(value, str):
            return f"trajectory.{field} must be a string"
    for field in ("step_count", "llm_call_count"):
        value = trajectory.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            return f"trajectory.{field} must be a non-negative integer"
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        return "trajectory.steps must be an array"
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"trajectory.steps[{index}] must be an object"
        for field in ("tool_call", "tool_result"):
            value = step.get(field)
            if value is not None and not isinstance(value, dict):
                return f"trajectory.steps[{index}].{field} must be an object"
        linked = step.get("llm_call_indices")
        if linked is not None and not isinstance(linked, list):
            return f"trajectory.steps[{index}].llm_call_indices must be an array"
        duration = step.get("duration_ms")
        if duration is not None and not _is_finite_number(duration):
            return f"trajectory.steps[{index}].duration_ms must be a finite number"
    failure_modes = trajectory.get("failure_modes")
    if failure_modes is not None and not isinstance(failure_modes, list):
        return "trajectory.failure_modes must be an array"
    for index, call in enumerate(calls):
        for field in ("duration_ms", "prompt_chars", "response_chars"):
            value = call.get(field)
            if value is not None and not _is_finite_number(value):
                return f"manifest[{index}].{field} must be a finite number"
    return None


def _render_show(trace_dir: Path, *, json_output: bool = False) -> int:
    if not trace_dir.exists():
        print(f"error: {trace_dir} does not exist", file=sys.stderr)
        return 1
    try:
        descriptor = read_artifact_descriptor(
            trace_dir,
            expected_kinds=("eval_run", "provenance"),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # ── trajectory.json (optional; short-circuit if missing) ─────
    traj_path = trace_dir / "trajectory.json"
    manifest_path = trace_dir / "manifest.jsonl"
    declared = set(descriptor["components"]) if descriptor is not None else None
    traj: dict[str, Any] = {}
    if (declared is None or "trajectory" in declared) and traj_path.exists():
        try:
            parsed_traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"error: could not parse {traj_path}: {exc}", file=sys.stderr)
            return 1
        if not isinstance(parsed_traj, dict):
            print(f"error: expected a JSON object in {traj_path}", file=sys.stderr)
            return 1
        traj = parsed_traj

    # ── manifest.jsonl (optional) ────────────────────────────────
    calls: list[dict[str, Any]] = []
    if (declared is None or "model_calls" in declared) and manifest_path.exists():
        try:
            manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            print(f"error: could not read {manifest_path}: {exc}", file=sys.stderr)
            return 1
        for line_number, line in enumerate(manifest_lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                call = json.loads(line)
            except Exception as exc:
                print(
                    f"error: could not parse {manifest_path} line {line_number}: {exc}",
                    file=sys.stderr,
                )
                return 1
            if not isinstance(call, dict):
                print(
                    f"error: expected a JSON object in {manifest_path} line {line_number}",
                    file=sys.stderr,
                )
                return 1
            calls.append(call)

    if not traj and not calls:
        print(
            f"error: {trace_dir} contains no trajectory.json or "
            "manifest.jsonl - not a trace directory",
            file=sys.stderr,
        )
        return 1

    shape_error = _trace_shape_error(traj, calls)
    if shape_error is not None:
        print(f"error: invalid trace shape: {shape_error}", file=sys.stderr)
        return 1

    eval_record = None
    saved_evals = None
    if (descriptor is not None and descriptor["kind"] == "eval_run") or (
        descriptor is None
        and any((trace_dir / name).exists() for name in ("evals.json", "artifacts.json"))
    ):
        from looplet.evals import load_eval_run

        try:
            eval_record = load_eval_run(trace_dir)
            evals_path = trace_dir / "evals.json"
            saved_evals = (
                json.loads(evals_path.read_text(encoding="utf-8")) if evals_path.exists() else []
            )
        except Exception as exc:
            print(
                f"error: could not read saved eval evidence in {trace_dir}: {exc}", file=sys.stderr
            )
            return 1

    if json_output:
        payload = {"trajectory": traj or None, "manifest": calls}
        if eval_record is not None:
            payload["evals"] = saved_evals
            payload["artifacts"] = eval_record.context.artifacts
        print(json.dumps(payload, indent=2))
        return 0

    # ── Header ──────────────────────────────────────────────────
    run_id = traj.get("run_id") or trace_dir.name
    term = traj.get("termination_reason", "?")
    term_glyph = {"done": "✓", "error": "✗"}.get(term, "·")
    step_count = traj.get("step_count", len(traj.get("steps", [])))
    llm_count = traj.get("llm_call_count", len(calls))
    # Total duration: sum step durations if available, else call durations.
    total_ms = sum(s.get("duration_ms") or 0 for s in traj.get("steps", []))
    if total_ms == 0 and calls:
        total_ms = sum(c.get("duration_ms") or 0 for c in calls)

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

    if term == "done" and traj.get("steps"):
        terminal_step = traj["steps"][-1]
        terminal_result = terminal_step.get("tool_result") or {}
        final_output = terminal_result.get("data")
        rejected = isinstance(final_output, dict) and final_output.get("rejected")
        if (
            final_output is not None
            and not terminal_step.get("error")
            and not terminal_result.get("error")
            and not rejected
        ):
            print("\nresult:")
            print(
                final_output
                if isinstance(final_output, str)
                else json.dumps(final_output, indent=2)
            )

    if eval_record is not None:
        print("\nsaved evals:")
        for result in eval_record.results:
            print(f"  {result.pretty()}")
        if not eval_record.results:
            print("  (none)")
        print("\nartifacts:")
        print(json.dumps(eval_record.context.artifacts, indent=2))
        print(
            "\nCompletion is separate from correctness. Apply your gate threshold to these saved scores."
        )

    return 0


def _status_line(status: str, name: str, detail: str) -> str:
    marks = {"ok": "OK", "warn": "WARN", "error": "ERROR"}
    return f"{marks.get(status, '?')} {name}: {detail}"


def _endpoint_check(backend: Any) -> dict[str, str] | None:
    import os
    from urllib.parse import urlsplit

    from looplet.backends import AnthropicBackend, OpenAIBackend

    endpoint = getattr(getattr(backend, "_client", None), "base_url", None)
    if endpoint is None:
        return None
    try:
        parsed = urlsplit(str(endpoint))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("unsupported endpoint")
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        origin = f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return {
            "name": "endpoint",
            "status": "warn",
            "detail": "endpoint could not be safely displayed",
        }
    source = "provider client default"
    normalized = parsed._replace(path=parsed.path.rstrip("/"), query=parsed.query.rstrip("/"))
    variables: tuple[str, ...] = ()
    if isinstance(backend, AnthropicBackend):
        variables = ("ANTHROPIC_BASE_URL",)
    elif isinstance(backend, OpenAIBackend):
        provider = os.environ.get("LOOPLET_PROVIDER", "").lower()
        if provider in {"openai", "oai"} or (
            not provider and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))
        ):
            variables = ("OPENAI_BASE_URL",)
        else:
            variables = ("COPILOT_PROXY_URL", "LOOPLET_TAX_LLM_BASE_URL", "OPENAI_BASE_URL")
    for variable in variables:
        configured = os.environ.get(variable)
        if not configured:
            continue
        try:
            candidate = urlsplit(configured)
        except ValueError:
            continue
        if (
            candidate._replace(path=candidate.path.rstrip("/"), query=candidate.query.rstrip("/"))
            == normalized
        ):
            source = variable
            break
    return {
        "name": "endpoint",
        "status": "ok",
        "detail": f"{origin} (origin only; source: {source})",
    }


def _build_check(package_root: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    try:
        for source in sorted(package_root.rglob("*.py")):
            relative_path = source.relative_to(package_root)
            if relative_path.parts[0] == "_bundled":
                continue
            digest.update(relative_path.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        return {
            "name": "build",
            "status": "warn",
            "detail": f"Source fingerprint unavailable ({type(exc).__name__})",
        }
    return {"name": "build", "status": "ok", "detail": f"sha256:{digest.hexdigest()}"}


def _doctor_checks(*, probe_backend: bool) -> list[dict[str, str]]:
    import warnings

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
    checks.append(_build_check(Path(__file__).parent))

    from looplet.backends import ProviderSelectionWarning, make_backend  # noqa: PLC0415

    llm = None
    with warnings.catch_warnings(record=True) as backend_warnings:
        warnings.simplefilter("always", ProviderSelectionWarning)
        try:
            llm = make_backend()
            checks.append({"name": "provider", "status": "ok", "detail": type(llm).__name__})
            checks.append(
                {
                    "name": "model",
                    "status": "ok",
                    "detail": str(getattr(llm, "_model", "backend default")),
                }
            )
            endpoint = _endpoint_check(llm)
            if endpoint is not None:
                checks.append(endpoint)
        except Exception as exc:
            checks.append({"name": "backend_config", "status": "warn", "detail": str(exc)})
    for warning in backend_warnings:
        if issubclass(warning.category, ProviderSelectionWarning):
            checks.append(
                {"name": "provider_selection", "status": "warn", "detail": str(warning.message)}
            )
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )

    if not probe_backend:
        checks.append(
            {"name": "backend_probe", "status": "ok", "detail": "skipped by --no-backend"}
        )
        return checks
    if llm is None:
        checks.append(
            {
                "name": "backend_probe",
                "status": "warn",
                "detail": "skipped; configure a provider first",
            }
        )
        return checks

    try:
        from looplet.native_tools import probe_native_tool_support  # noqa: PLC0415

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
                    "status": "warn",
                    "detail": "native tools unavailable; set use_native_tools=False for JSON-text mode",
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
    json_output: bool = False,
    require_completed: bool = False,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    output = sys.stdout
    with execution_output(json_output):
        return _run_bundle(
            bundle_path=bundle_path,
            task=task,
            workspace=workspace,
            max_steps=max_steps,
            scripted=scripted,
            scripted_responses=scripted_responses,
            require_tests=require_tests,
            trace_dir=trace_dir,
            no_trace=no_trace,
            json_output=json_output,
            output=output,
            require_completed=require_completed,
            model=model,
            base_url=base_url,
        )


def _run_bundle(
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
    json_output: bool,
    output: TextIO,
    require_completed: bool = False,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    from looplet.bundles import (  # noqa: PLC0415
        BundleValidation,
        SkillRuntime,
        load_skill_bundle,
        run_skill_bundle,
        validate_skill_bundle,
    )
    from looplet.cli import build_cli_backend  # noqa: PLC0415
    from looplet.resilient import ResilientBackend  # noqa: PLC0415
    from looplet.testing import MockLLMBackend  # noqa: PLC0415
    from looplet.types import RunResult  # noqa: PLC0415

    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"error: project root is not a directory: {workspace}", file=sys.stderr)
        return 1
    if task == "-":
        task = sys.stdin.read()
        if not task.strip():
            print("error: task read from stdin is empty", file=sys.stderr)
            return 1
    if max_steps < 1:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 1

    def _close_validation_preset(validation: BundleValidation | None) -> None:
        if validation is not None and validation.preset is not None:
            validation.preset.close()

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
            options={"require_tests": require_tests, "use_native_tools": True},
            output_dir=output_dir,
        )

    def _report_invalid_bundle(validation: BundleValidation) -> None:
        print(f"error: invalid bundle {bundle_path}", file=sys.stderr)
        for error in validation.errors:
            print(f"  - {error}", file=sys.stderr)

    bundle_run = getattr(bundle.module, "run", None)
    legacy_run = (
        callable(bundle_run)
        and not json_output
        and not require_completed
        and model is None
        and base_url is None
    )
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
                    (None if no_trace else trace_dir) if legacy_run else effective_trace_dir,
                ),
            )
            if not provider_validation.ok:
                _report_invalid_bundle(provider_validation)
                _close_validation_preset(provider_validation)
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
                _close_validation_preset(provider_validation)
                return 1
            if provider_returned_string:
                print(
                    f"error: bundle {bundle.skill.name!r} scripted_responses() must return "
                    "an iterable of response strings, got str",
                    file=sys.stderr,
                )
                _close_validation_preset(provider_validation)
                return 1
            if not scripted_responses:
                print(
                    f"error: bundle {bundle.skill.name!r} scripted_responses() returned no responses",
                    file=sys.stderr,
                )
                _close_validation_preset(provider_validation)
                return 1
            for index, response in enumerate(scripted_responses, start=1):
                if not isinstance(response, str):
                    print(
                        f"error: bundle {bundle.skill.name!r} scripted_responses() item "
                        f"{index} must be str, got {type(response).__name__}",
                        file=sys.stderr,
                    )
                    _close_validation_preset(provider_validation)
                    return 1
                if not response.strip():
                    print(
                        f"error: bundle {bundle.skill.name!r} scripted_responses() item "
                        f"{index} must not be empty",
                        file=sys.stderr,
                    )
                    _close_validation_preset(provider_validation)
                    return 1
        elif not legacy_run:
            print(
                f"error: bundle {bundle.skill.name!r} does not provide scripted_responses()",
                file=sys.stderr,
            )
            return 1

    if legacy_run:
        validation = provider_validation or validate_skill_bundle(
            bundle,
            _validation_runtime(None if no_trace else trace_dir),
        )
        if not validation.ok:
            _report_invalid_bundle(validation)
            _close_validation_preset(validation)
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
        finally:
            _close_validation_preset(validation)
        if isinstance(result, bool) or not isinstance(result, int):
            print(f"error: bundle {bundle.skill.name!r} returned invalid status", file=sys.stderr)
            print(f"  - expected int exit code, got {type(result).__name__}", file=sys.stderr)
            return 1
        return result

    runtime = SkillRuntime(
        workspace=workspace,
        max_steps=max_steps,
        options={"require_tests": require_tests, "use_native_tools": True},
        output_dir=effective_trace_dir,
    )
    validation = provider_validation or validate_skill_bundle(bundle, runtime)
    if not validation.ok:
        _report_invalid_bundle(validation)
        _close_validation_preset(validation)
        return 1

    preset = validation.preset
    if preset is None:
        print("error: bundle did not build an AgentPreset", file=sys.stderr)
        return 1
    preset.config.max_steps = max_steps
    preset.state.max_steps = max_steps

    if scripted_mode:
        llm = MockLLMBackend(responses=scripted_responses)
        model_label = "scripted MockLLMBackend"
    else:
        try:
            backend = build_cli_backend(model=model, base_url=base_url)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            _close_validation_preset(validation)
            return 1
        llm = ResilientBackend(backend, retries=2, timeout_s=120)
        model_label = str(getattr(backend, "_model", type(backend).__name__))

    for warning in validation.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    uses_native_protocol = bool(
        validation.preset is not None and validation.preset.config.use_native_tools
    )

    if not json_output:
        print(f"looplet run {bundle.skill.name}")
        print(f"  Task: {task}")
        print(f"  Workspace: {workspace}")
        print(f"  Model: {model_label} | Budget: {max_steps} steps")
        print(
            "  Tool protocol: native (use_native_tools=False for JSON-text)"
            if uses_native_protocol
            else "  Tool protocol: json-text"
        )
        print()

    render_step = getattr(bundle.module, "render_step", None)
    started = time.perf_counter()
    n_steps = 0
    try:
        try:
            for step in run_skill_bundle(
                bundle,
                llm=llm,
                task=task,
                runtime=runtime,
                provenance=not no_trace,
                trace_dir=effective_trace_dir,
                preset=preset,
            ):
                n_steps += 1
                if json_output:
                    continue
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
    finally:
        _close_validation_preset(validation)
    result = RunResult.from_state(preset.state)
    payload = completion_payload(
        result,
        steps=n_steps,
        duration_ms=(time.perf_counter() - started) * 1000,
        trace_dir=effective_trace_dir,
    )
    if json_output:
        print(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ),
            file=output,
        )
    else:
        reason = result.termination_reason or "unknown"
        status = "done" if result.completed else f"stopped ({reason})"
        if result.failed:
            status = f"failed ({reason})"
        print(f"\n  {status} - {n_steps} steps")
        if payload["error"] is not None:
            print(f"  error: {payload['error']['message']}")
        if effective_trace_dir is not None:
            print(f"  Trace: {effective_trace_dir}")
    return completion_exit_code(result, require_completed=require_completed)


def _render_inspect(
    path: Path,
    *,
    json_output: bool,
    mode: str = "declaration",
    workspace: Path | None = None,
    max_steps: int = 20,
    include_sensitive: bool = False,
) -> int:
    if path.name in {"cartridge.json", "workspace.json"}:
        path = path.parent
    if (path / "cartridge.json").is_file():
        from looplet.cli.spec_commands import cmd_describe  # noqa: PLC0415

        return cmd_describe(
            argparse.Namespace(
                cartridge=path,
                json=json_output,
                instantiate=mode == "runtime",
                include_sensitive=include_sensitive,
            )
        )
    if (path / "SKILL.md").is_file() or path.name == "SKILL.md":
        if mode == "runtime":
            return _render_blueprint(
                bundle_path=path,
                workspace=workspace or Path.cwd(),
                max_steps=max_steps,
                instantiate=True,
                report_mode=mode,
            )
        from looplet.bundles import discover_skill_bundles  # noqa: PLC0415

        try:
            skill_file = path if path.name == "SKILL.md" else path / "SKILL.md"
            cards = discover_skill_bundles(skill_file, include_invalid=True)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not cards:
            print(f"error: no skill bundle found at {path}", file=sys.stderr)
            return 1
        card = cards[0]
        if json_output:
            print(
                json.dumps(
                    {**card.to_dict(), "mode": "declaration-only", "runtime_validated": False},
                    indent=2,
                )
            )
        else:
            print(f"{card.name}\t{'ok' if card.ok else 'invalid'}\t{card.description}")
            print("  declaration-only; executable wiring has not been validated")
            for error in card.errors:
                print(f"  - {error}")
        return 0 if card.ok else 1
    if mode == "runtime":
        print("error: runtime inspection requires a cartridge or bundle", file=sys.stderr)
        return 1
    return _render_show(path, json_output=json_output)


def _render_blueprint(
    *,
    bundle_path: Path,
    workspace: Path,
    max_steps: int,
    instantiate: bool = True,
    report_mode: str | None = None,
) -> int:
    from looplet.blueprints import blueprint_from_bundle  # noqa: PLC0415
    from looplet.bundles import SkillRuntime  # noqa: PLC0415

    with execution_output(True):
        try:
            blueprint = blueprint_from_bundle(
                bundle_path,
                SkillRuntime(workspace=workspace, max_steps=max_steps),
                instantiate=instantiate,
                validate=instantiate and report_mode is not None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not inspect bundle {bundle_path}", file=sys.stderr)
            print(f"  - {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    report = blueprint.to_dict()
    validated = bool(blueprint.metadata.get("runtime_validated", False))
    if report_mode is not None:
        report.update(
            mode="runtime" if instantiate else "declaration-only", runtime_validated=validated
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report_mode is not None and instantiate and not validated else 0


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
        description="Looplet - build, run, inspect, and test Python agents.",
    )
    parser.add_argument("--version", action="version", version=f"looplet {__version__}")
    parser.add_argument(
        "--help-all", action="help", help="Show advanced commands and compatibility aliases"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    if raw in (["--help"], ["-h"]):
        print(parser.format_usage().rstrip())
        print(f"\n{parser.description}\n\nWorkflow:")
        print("  Python: run your agent with `python your_agent.py`; packaging is optional.\n")
        for command, description in (
            ("new", "Create an agent cartridge"),
            ("run", "Run a cartridge or skill bundle"),
            ("inspect", "Inspect declarations, live wiring, or captured traces"),
            ("eval", "Test agent behavior against cases"),
            ("replay", "Replay captured responses in a fresh workspace"),
        ):
            print(f"  {command:10s} {description}")
        print("\nUse COMMAND --help for options; --help-all lists every command and alias.")
        parser.exit()

    inspect = sub.add_parser(
        "inspect", help="Inspect declarations or trace evidence; explicitly opt in to live wiring"
    )
    inspect.add_argument("path", type=Path, help="Path to a cartridge, bundle, or trace")
    inspect.add_argument("--json", action="store_true", help="Emit machine-readable inspection")
    inspect.add_argument(
        "--mode",
        choices=["declaration", "runtime"],
        default="declaration",
        help="Declaration mode never imports authored code; runtime mode builds and closes it",
    )
    inspect.add_argument("--workspace", type=Path, help="Workspace for live bundle inspection")
    inspect.add_argument("--max-steps", type=int, default=20, help="Live bundle step budget")
    inspect.add_argument(
        "--include-sensitive", action="store_true", help="Include authored content"
    )

    show = sub.add_parser(
        "show",
        help="Show a one-page summary of a captured trace directory",
    )
    show.add_argument("trace_dir", type=Path, help="Path to a trace directory")
    show.add_argument("--json", action="store_true", help="Emit the parsed trace artifacts as JSON")

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
        aliases=["run-bundle"],
        help="Run a cartridge or runnable skill bundle",
    )
    run.add_argument("bundle", type=Path, help="Path to a cartridge or runnable skill bundle")
    run.add_argument("task", help="Task description, or '-' to read it from stdin")
    run.add_argument(
        "--project-root",
        "--workspace",
        "-w",
        dest="workspace",
        type=Path,
        help="Existing directory the agent operates on (defaults to the current project)",
    )
    run.add_argument(
        "--max-steps",
        type=int,
        help="Charged-step budget; successful free calls do not count (bundle default: 20)",
    )
    bundle_options = run.add_argument_group("Skill-bundle-only options")
    bundle_options.add_argument(
        "--scripted",
        action="store_true",
        help="Use bundle-provided deterministic scripted responses",
    )
    bundle_options.add_argument(
        "--no-tests",
        action="store_true",
        help="Pass require_tests=False to bundles that support it",
    )
    add_run_options(run)

    blueprint = sub.add_parser(
        "blueprint",
        help="Print a runnable skill bundle blueprint as JSON",
    )
    blueprint.add_argument("bundle", type=Path, help="Path to a runnable skill bundle")
    blueprint.add_argument(
        "--workspace",
        "-w",
        type=Path,
        default=Path.cwd(),
        help="Directory supplied to the live factory",
    )
    blueprint.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Charged-step budget; successful free calls do not count",
    )
    blueprint_mode = blueprint.add_mutually_exclusive_group()
    blueprint_mode.add_argument(
        "--instantiate", action="store_true", help="Build and close live resources (the default)"
    )
    blueprint_mode.add_argument(
        "--declaration-only",
        dest="instantiate",
        action="store_false",
        help="Inspect declarations without importing or building authored code",
    )
    blueprint.set_defaults(instantiate=None)
    blueprint.add_argument(
        "--mode", choices=["declaration", "runtime"], help="Explicit inspection mode"
    )

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

    # ── factory subcommands (``new``, ``run-cartridge``) ──
    # Lives in a dedicated module so the two factory-facing commands
    # can evolve independently of the bundle / trace / eval CLI.
    from looplet.cli.factory_commands import add_subparsers as _add_factory  # noqa: PLC0415

    _add_factory(sub)

    # ── spec subcommands (``conform``, ``describe``, ``diff``) ──
    # Cartridge Spec v2.0 tooling. Operationalises the conformance,
    # legibility, and semantic-diff promises into runnable commands.
    from looplet.cli.spec_commands import add_subparsers as _add_spec  # noqa: PLC0415

    _add_spec(sub)

    args = parser.parse_args(argv)

    if args.command in {"describe", "blueprint"}:
        args.report_mode = args.mode
        default_mode = "runtime" if args.command == "blueprint" else "declaration"
        legacy_mode = (
            ("runtime" if args.instantiate else "declaration")
            if args.instantiate is not None
            else None
        )
        if args.mode is not None and legacy_mode is not None and args.mode != legacy_mode:
            parser.error("--mode conflicts with the legacy inspection flag")
        args.mode = args.mode or legacy_mode or default_mode
        args.instantiate = args.mode == "runtime"

    if args.command == "inspect":
        return _render_inspect(
            args.path,
            json_output=args.json,
            mode=args.mode,
            workspace=args.workspace,
            max_steps=args.max_steps,
            include_sensitive=args.include_sensitive,
        )
    if args.command == "show":
        return _render_show(args.trace_dir, json_output=args.json)
    if args.command == "doctor":
        return _render_doctor(
            probe_backend=not args.no_backend,
            json_output=args.json,
            strict=args.strict,
        )
    if args.command in {"run", "run-bundle"}:
        if args.command == "run" and args.bundle.suffix == ".py" and not args.bundle.is_dir():
            print(
                f"error: run Python agents with `python {shlex.quote(str(args.bundle))}`. "
                "`looplet run` accepts cartridge or skill-bundle directories.",
                file=sys.stderr,
            )
            return 1
        from looplet.cartridge.runtime_helpers import resolve_project_root  # noqa: PLC0415
        from looplet.cli.factory_commands import cmd_run_workspace  # noqa: PLC0415

        runtime = {"project_root": str(args.workspace)} if args.workspace is not None else None
        project_root = Path(resolve_project_root(runtime))
        is_cartridge = (args.bundle / "cartridge.json").is_file()
        is_bundle = (args.bundle / "SKILL.md").is_file() or args.bundle.name == "SKILL.md"
        if args.command == "run" and is_cartridge:
            if is_bundle:
                print(
                    "error: both cartridge.json and SKILL.md are present; "
                    "use run-cartridge or run-bundle explicitly",
                    file=sys.stderr,
                )
                return 1
            if args.scripted:
                print("error: use --scripted-response for cartridges", file=sys.stderr)
                return 1
            if args.no_tests:
                print("error: --no-tests is only for skill bundles", file=sys.stderr)
                return 1
            return cmd_run_workspace(
                argparse.Namespace(
                    workspace=args.bundle,
                    task=args.task,
                    project_root=project_root,
                    max_steps=args.max_steps,
                    trace_dir=args.trace_dir,
                    no_trace=args.no_trace,
                    json=args.json,
                    require_completed=args.require_completed,
                    quiet=args.quiet,
                    pretty=args.pretty,
                    parent_trace=args.parent_trace,
                    scripted_response=args.scripted_response,
                    model=args.model,
                    base_url=args.base_url,
                )
            )
        for flag, enabled in (
            ("--quiet", args.quiet),
            ("--pretty", args.pretty),
            ("--parent-trace", args.parent_trace is not None),
        ):
            if enabled:
                print(f"error: {flag} is only for cartridges", file=sys.stderr)
                return 1
        return _render_run(
            bundle_path=args.bundle,
            task=args.task,
            workspace=project_root,
            max_steps=args.max_steps if args.max_steps is not None else 20,
            scripted=args.scripted,
            scripted_responses=args.scripted_response,
            require_tests=not args.no_tests,
            trace_dir=args.trace_dir,
            no_trace=args.no_trace,
            json_output=args.json,
            require_completed=args.require_completed,
            model=args.model,
            base_url=args.base_url,
        )
    if args.command == "blueprint":
        return _render_blueprint(
            bundle_path=args.bundle,
            workspace=args.workspace,
            max_steps=args.max_steps,
            instantiate=args.instantiate,
            report_mode=args.report_mode,
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
    # Factory subcommands (``new``, ``run-cartridge``) attach their
    # handler via ``set_defaults(_handler=...)`` so we dispatch
    # generically here. Keeps ``__main__`` from growing per-command
    # branches as more factory commands land.
    handler = getattr(args, "_handler", None)
    if handler is not None:
        return int(handler(args))
    # Unreachable - argparse rejects unknown commands.
    return 2


if __name__ == "__main__":
    sys.exit(main())
