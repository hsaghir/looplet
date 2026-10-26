"""Qualify an installed Core candidate with Looplet Analytics and a live proxy.

Run in an Analytics environment with the candidate Core wheel installed. Set
OPENAI_BASE_URL and OPENAI_API_KEY, then pass a fresh --output directory.
Use --self-check to validate fixtures and resolved profiles without a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import looplet_analytics
from jsonschema import Draft202012Validator
from looplet_analytics.agent import load_analytics_cartridge, run_agent
from looplet_analytics.llm import build_openai_backend

import looplet

INVOICES = [
    (1, "C1", "2026-01-01", 100, "paid"),
    (2, "C2", "2026-01-02", 200, "paid"),
    (3, "C1", "2026-01-03", 300, "cancelled"),
    (4, "C3", "2026-01-03", 50, "paid"),
    (5, "C2", "2026-01-04", 75, "paid"),
    (6, "C4", "2026-02-01", 1000, "paid"),
    (7, "C1", "2026-01-20", -20, "paid"),
    (8, "C3", "2025-12-31", 999, "paid"),
]
CUSTOMERS = {"C1": "enterprise", "C2": "retail", "C3": "enterprise", "C4": "retail"}
HISTORY = [
    ("C1", "2025-01-01", "2026-01-10", "enterprise"),
    ("C1", "2026-01-10", "", "retail"),
    ("C2", "2025-01-01", "", "retail"),
    ("C3", "2025-01-01", "", "enterprise"),
    ("C4", "2025-01-01", "", "retail"),
]


def write_csv(path: Path, columns: list[str], rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def fixtures(root: Path) -> dict[str, int]:
    root.mkdir()
    write_csv(
        root / "invoices.csv", ["invoice", "customer", "issued", "amount", "status"], INVOICES
    )
    write_csv(root / "customers.csv", ["customer", "segment"], list(CUSTOMERS.items()))
    write_csv(
        root / "customer_history.csv", ["customer", "valid_from", "valid_to", "segment"], HISTORY
    )
    (root / "manual.md").write_text(
        "# Invoice policy\n\n"
        "Only paid invoices contribute to paid net amount. Include negative reversals.\n"
        "Dates use UTC. January 2026 starts on January 1 and excludes February 1.\n"
        "customers.csv records current segments, not historical segments.\n"
        "History periods include valid_from and exclude valid_to; blank valid_to is unbounded.\n"
        "Standard invoices become overdue after 30 days. Premium invoices allow 45 days.\n",
        encoding="utf-8",
    )
    paid = [row for row in INVOICES if row[4] == "paid" and "2026-01-01" <= row[2] < "2026-02-01"]
    temporal_total = sum(
        row[3]
        for row in paid
        for customer, start, end, segment in HISTORY
        if customer == row[1]
        and start <= row[2]
        and (not end or row[2] < end)
        and segment == "enterprise"
    )
    return {
        "paid": sum(row[3] for row in paid),
        "join": sum(row[3] for row in paid if CUSTOMERS[row[1]] == "enterprise"),
        "temporal": temporal_total,
        "policy": 30,
    }


CASES = [
    ("paid", "direct", "chat", "What is the paid net invoice amount for January 2026?"),
    (
        "join",
        "grounded",
        "responses",
        "What is the January 2026 paid net invoice amount "
        "for customers currently classified as enterprise in customers.csv?",
    ),
    (
        "temporal",
        "direct",
        "responses",
        "What is the January 2026 paid net invoice amount "
        "for customers classified as enterprise on each invoice's issued date, using customer_history.csv?",
    ),
    (
        "policy",
        "standard",
        "chat",
        "According to manual.md, after how many days do standard "
        "invoices become overdue? Do not use the premium policy.",
    ),
]


def numeric_grade(answer: str, expected: int) -> bool:
    try:
        return Decimal(answer.strip()) == Decimal(expected)
    except InvalidOperation:
        return False


def inspect_profiles(context: Path) -> dict[str, Any]:
    profiles = {}
    for profile in ("direct", "standard", "grounded"):
        preset, _, _ = load_analytics_cartridge(context, profile=profile, max_steps=16)
        with preset:
            schemas = preset.tools.tool_schemas()
            for schema in schemas:
                Draft202012Validator.check_schema(schema["input_schema"])
            profiles[profile] = {
                "harness": preset.resolve_harness().describe(),
                "tool_schema_count": len(schemas),
            }
    return profiles


def run_trial(
    output: Path, context: Path, model: str, case: tuple[str, str, str, str], expected: int
) -> dict[str, Any]:
    name, profile, wire, question = case
    trial = output / f"{model}-{name}-{wire}-{profile}"
    trial.mkdir()
    os.environ["OPENAI_WIRE_API"] = wire
    backend = build_openai_backend(model=model, max_tokens=6000, timeout=90, max_retries=0)
    report: dict[str, Any] = {"model": model, "case": name, "wire": wire, "profile": profile}
    try:
        run = run_agent(
            {
                "question": question
                + " Return only the final numeric value, without units or prose."
            },
            llm=backend,
            context_dir=context,
            trace_dir=trial / "trace",
            profile=profile,
            max_steps=16,
            max_model_calls=16,
            max_tokens=6000,
            case_timeout_s=180,
        )
        report.update(
            answer=run.answer,
            completed=run.completed,
            stop_reason=run.stop_reason,
            error=run.error,
            reported_llm_calls=run.llm_calls,
            step_count=run.step_count,
            duration_ms=run.duration_ms,
            evidence_verified=run.evidence_verified,
            gate_rejections=run.gate_rejections,
            expected=expected,
            grade=numeric_grade(run.answer, expected),
            configuration=run.configuration,
        )
    except Exception as error:
        report.update(completed=False, error=f"{type(error).__name__}: {error}", grade=False)
    finally:
        backend._client.close()
    report["passed"] = bool(report["completed"] and report["grade"] and not report["error"])
    (trial / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "model",
                    "case",
                    "wire",
                    "profile",
                    "passed",
                    "answer",
                    "error",
                    "reported_llm_calls",
                )
            }
        ),
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-6-luna"])
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    context = args.output / "context"
    expected = fixtures(context)
    assert expected == {"paid": 405, "join": 130, "temporal": 150, "policy": 30}
    assert numeric_grade("405.0", 405) and not numeric_grade("404", 405)
    assert not numeric_grade("I completed the task", 405)
    provenance = {
        "looplet": looplet.__file__,
        "analytics": looplet_analytics.__file__,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in context.iterdir()
        },
        "expected": expected,
    }
    profiles = inspect_profiles(context)
    if args.self_check:
        print(json.dumps({"provenance": provenance, "profiles": profiles}, indent=2))
        return 0
    reports = [
        run_trial(args.output, context, model, case, expected[case[0]])
        for model in args.models
        for case in CASES
    ]
    summary = {
        "provenance": provenance,
        "profiles": profiles,
        "passed": all(row["passed"] for row in reports),
        "trials": reports,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
