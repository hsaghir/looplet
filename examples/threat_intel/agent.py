#!/usr/bin/env python3
"""Threat Intel Briefing Agent — local-first security news analyst.

Fetches security advisories, extracts IOCs/CVEs, classifies severity,
and produces a structured daily threat intelligence briefing.

Runs 100% local with any OpenAI-compatible endpoint (Ollama, llama-server, vLLM).
No data leaves your machine.

Usage:
    # With llama-server running Qwen3.6-27B on port 8080:
    python examples/threat_intel/agent.py

    # With Ollama:
    OPENAI_BASE_URL=http://localhost:11434/v1 python examples/threat_intel/agent.py

    # With a cloud API:
    OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... python examples/threat_intel/agent.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from looplet import (
    BaseToolRegistry,
    Conversation,
    DefaultState,
    LoopConfig,
    OpenAIBackend,
    StaticMemorySource,
    StreamingHook,
    ToolSpec,
    TrajectoryRecorder,
    composable_loop,
    register_done_tool,
)
from looplet.compact import PruneToolResults, TruncateCompact, compact_chain
from looplet.limits import PerToolLimitHook
from looplet.provenance import RecordingLLMBackend
from looplet.resilient import ResilientBackend
from looplet.session import SessionLog
from looplet.stagnation import StagnationHook, tool_call_fingerprint
from looplet.streaming import CallbackEmitter
from looplet.tools import register_think_tool
from looplet.types import ToolContext

# ═══════════════════════════════════════════════════════════════════
# SIMULATED THREAT FEEDS (in production, these would be real RSS/API)
# ═══════════════════════════════════════════════════════════════════

THREAT_FEEDS = {
    "cisa_alerts": [
        {
            "id": "AA26-113A",
            "title": "Critical Vulnerability in Bitwarden CLI Supply Chain",
            "date": "2026-04-23",
            "source": "CISA",
            "summary": (
                "CISA is aware of an ongoing supply chain compromise affecting the "
                "Bitwarden CLI package distributed via npm. The malicious version "
                "(2026.4.1) exfiltrates vault credentials to an attacker-controlled "
                "endpoint at collect.checkmarx-analytics[.]com. Organizations using "
                "Bitwarden CLI should immediately verify package integrity against "
                "known-good hashes published at bitwarden.com/checksums. "
                "CVE-2026-3891 has been assigned with a CVSS score of 9.8."
            ),
            "cves": ["CVE-2026-3891"],
            "iocs": [
                "collect.checkmarx-analytics[.]com",
                "npm package bitwarden-cli@2026.4.1",
                "SHA256:a1b2c3d4e5f6...malicious_hash",
            ],
            "severity": "CRITICAL",
        },
        {
            "id": "AA26-112B",
            "title": "French Government Agency Data Breach via API Misconfiguration",
            "date": "2026-04-22",
            "source": "CISA",
            "summary": (
                "A French government employment agency confirmed a data breach "
                "affecting 43 million records. The breach was caused by an "
                "unauthenticated API endpoint that exposed personal information "
                "including names, social security numbers, and employment history. "
                "The exposed API was api.emploi-gouv[.]fr/v2/citizens. "
                "No CVE assigned. Organizations should audit their public-facing APIs."
            ),
            "cves": [],
            "iocs": ["api.emploi-gouv[.]fr/v2/citizens"],
            "severity": "HIGH",
        },
    ],
    "nvd_recent": [
        {
            "cve_id": "CVE-2026-3891",
            "description": "Bitwarden CLI npm package supply chain compromise allowing credential exfiltration",
            "cvss_v3": 9.8,
            "vendor": "Bitwarden",
            "product": "CLI",
            "published": "2026-04-23",
        },
        {
            "cve_id": "CVE-2026-3847",
            "description": "GitHub Actions runner token exposure via crafted workflow in public repositories",
            "cvss_v3": 8.1,
            "vendor": "GitHub",
            "product": "Actions",
            "published": "2026-04-22",
        },
        {
            "cve_id": "CVE-2026-3802",
            "description": "MeshCore mesh networking firmware buffer overflow allowing remote code execution",
            "cvss_v3": 7.5,
            "vendor": "MeshCore",
            "product": "Firmware",
            "published": "2026-04-21",
        },
    ],
    "osint_reports": [
        {
            "title": "Telecom Surveillance Campaign Targeting European Carriers",
            "source": "TechCrunch / Citizen Lab",
            "date": "2026-04-23",
            "summary": (
                "Researchers have uncovered two sophisticated surveillance campaigns "
                "targeting European telecom providers. The campaigns use custom "
                "implants delivered via SS7 protocol exploitation and compromise of "
                "lawful intercept systems. Attribution points to state-sponsored "
                "actors. Affected infrastructure includes Diameter signaling nodes "
                "and GTP tunneling endpoints. IOCs include C2 domains "
                "update.telecom-infra[.]net and ssl-verify.carrier-mgmt[.]com, "
                "and implant hashes SHA256:deadbeef1234...implant_a and "
                "SHA256:cafebabe5678...implant_b."
            ),
            "iocs": [
                "update.telecom-infra[.]net",
                "ssl-verify.carrier-mgmt[.]com",
                "SHA256:deadbeef1234...implant_a",
                "SHA256:cafebabe5678...implant_b",
            ],
            "ttp_ids": ["T1557", "T1040", "T1132"],
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════


def fetch_feed(*, feed_name: str) -> dict:
    """Fetch a threat intelligence feed by name."""
    available = list(THREAT_FEEDS.keys())
    if feed_name not in THREAT_FEEDS:
        return {"error": f"Unknown feed '{feed_name}'. Available: {available}"}
    items = THREAT_FEEDS[feed_name]
    return {"feed": feed_name, "item_count": len(items), "items": items}


def search_cve(*, cve_id: str) -> dict:
    """Look up details for a specific CVE."""
    for item in THREAT_FEEDS.get("nvd_recent", []):
        if item["cve_id"] == cve_id:
            return item
    return {"cve_id": cve_id, "error": "CVE not found in recent data"}


def extract_iocs(*, text: str, ctx: ToolContext) -> dict:
    """Extract Indicators of Compromise from text using pattern matching + LLM."""
    # Pattern-based extraction
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    domain_pattern = r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\[?\.\]?[a-zA-Z]{2,}(?:\[?\.\]?[a-zA-Z]{2,})*\b"
    cve_pattern = r"CVE-\d{4}-\d{4,}"
    sha256_pattern = r"SHA256:[a-fA-F0-9]+"
    hash_pattern = r"\b[a-fA-F0-9]{64}\b"

    ips = re.findall(ip_pattern, text)
    domains = [
        d
        for d in re.findall(domain_pattern, text)
        if "[.]" in d or (len(d) > 5 and "." in d and not d[0].isdigit())
    ]
    cves = re.findall(cve_pattern, text)
    hashes = re.findall(sha256_pattern, text) + re.findall(hash_pattern, text)

    result = {
        "ips": list(set(ips)),
        "domains": list(set(domains)),
        "cves": list(set(cves)),
        "hashes": list(set(hashes)),
        "total_iocs": len(set(ips)) + len(set(domains)) + len(set(cves)) + len(set(hashes)),
    }

    # Use ctx.llm to classify severity if available
    if ctx.llm is not None and result["total_iocs"] > 0:
        try:
            classification = ctx.llm.generate(
                f"Given these IOCs extracted from a threat report, classify the overall "
                f"threat severity as CRITICAL, HIGH, MEDIUM, or LOW. Respond with just "
                f"the severity level, nothing else.\n\nIOCs: {json.dumps(result, indent=2)}",
                max_tokens=20,
            )
            result["llm_severity"] = classification.strip().upper()
            ctx.warn(f"Used LLM to classify severity: {result['llm_severity']}")
        except Exception:
            pass

    return result


def map_mitre(*, technique_ids: str) -> dict:
    """Map MITRE ATT&CK technique IDs to descriptions."""
    mitre_db = {
        "T1557": {"name": "Adversary-in-the-Middle", "tactic": "Credential Access, Collection"},
        "T1040": {"name": "Network Sniffing", "tactic": "Credential Access, Discovery"},
        "T1132": {"name": "Data Encoding", "tactic": "Command and Control"},
        "T1195": {"name": "Supply Chain Compromise", "tactic": "Initial Access"},
        "T1059": {"name": "Command and Scripting Interpreter", "tactic": "Execution"},
    }
    ids = [t.strip() for t in technique_ids.split(",")]
    results = {}
    for tid in ids:
        if tid in mitre_db:
            results[tid] = mitre_db[tid]
        else:
            results[tid] = {"name": "Unknown", "tactic": "Unknown"}
    return {"techniques": results, "count": len(results)}


def assess_risk(*, title: str, severity: str, affected_products: str, ctx: ToolContext) -> dict:
    """Assess organizational risk for a specific threat using LLM reasoning."""
    if ctx.llm is not None:
        try:
            assessment = ctx.llm.generate(
                f"You are a threat intelligence analyst. Assess the organizational risk "
                f"for this threat in 2-3 sentences. Consider likelihood of exploitation, "
                f"blast radius, and recommended priority.\n\n"
                f"Threat: {title}\nSeverity: {severity}\n"
                f"Affected: {affected_products}\n\n"
                f"Respond with: PRIORITY: [IMMEDIATE/HIGH/MEDIUM/LOW] followed by "
                f"a brief justification.",
                max_tokens=150,
            )
            ctx.warn("Used LLM for risk assessment")
            return {
                "title": title,
                "severity": severity,
                "assessment": assessment.strip(),
            }
        except Exception as e:
            return {"title": title, "severity": severity, "assessment": f"LLM error: {e}"}
    return {
        "title": title,
        "severity": severity,
        "assessment": f"Risk assessment unavailable (no LLM). Severity: {severity}",
    }


# ═══════════════════════════════════════════════════════════════════
# AGENT SETUP
# ═══════════════════════════════════════════════════════════════════


def build_agent():
    """Build the threat intel briefing agent."""
    # LLM
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "local")
    model = os.environ.get("OPENAI_MODEL", "Qwen3.6-27B")

    base_llm = OpenAIBackend(
        base_url=base_url,
        api_key=api_key,
        model=model,
        tool_choice="required",
    )
    llm = ResilientBackend(base_llm, retries=2, timeout_s=120)
    recording = RecordingLLMBackend(llm)

    # Tools
    tools = BaseToolRegistry()
    register_done_tool(
        tools,
        parameters={
            "briefing": "The complete threat intelligence briefing in structured text",
        },
    )
    register_think_tool(tools)

    tools.register(
        ToolSpec(
            name="fetch_feed",
            description="Fetch a threat intelligence feed. Available feeds: cisa_alerts, nvd_recent, osint_reports",
            parameters={"feed_name": "str"},
            execute=fetch_feed,
        )
    )
    tools.register(
        ToolSpec(
            name="search_cve",
            description="Look up details for a specific CVE ID (e.g., CVE-2026-3891)",
            parameters={"cve_id": "str"},
            execute=search_cve,
        )
    )
    tools.register(
        ToolSpec(
            name="extract_iocs",
            description="Extract IOCs (IPs, domains, CVEs, hashes) from text and classify severity",
            parameters={"text": "str"},
            execute=extract_iocs,
        )
    )
    tools.register(
        ToolSpec(
            name="map_mitre",
            description="Map MITRE ATT&CK technique IDs to names and tactics. Pass comma-separated IDs.",
            parameters={"technique_ids": "str"},
            execute=map_mitre,
        )
    )
    tools.register(
        ToolSpec(
            name="assess_risk",
            description="Assess organizational risk for a specific threat",
            parameters={"title": "str", "severity": "str", "affected_products": "str"},
            execute=assess_risk,
        )
    )

    # Hooks
    stag_hook = StagnationHook(
        fingerprint=tool_call_fingerprint,
        threshold=3,
        nudge="[stagnation] You're repeating yourself. Synthesize what you have and produce the briefing.",
    )
    limit_hook = PerToolLimitHook(
        default_limit=8,
        limits={"fetch_feed": 4, "think": 3, "assess_risk": 4},
    )

    events: list = []
    stream_hook = StreamingHook(CallbackEmitter(events.append))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    config = LoopConfig(
        max_steps=15,
        max_tokens=1000,
        temperature=0.3,
        use_native_tools=True,
        system_prompt=(
            f"You are a senior threat intelligence analyst producing a daily briefing "
            f"for {today}. Your task:\n\n"
            f"1. Fetch all three feeds: cisa_alerts, nvd_recent, osint_reports\n"
            f"2. For each critical/high severity item, extract IOCs and assess risk\n"
            f"3. Map any MITRE ATT&CK technique IDs\n"
            f"4. Call done() with a structured briefing that includes:\n"
            f"   - Executive Summary (3-4 sentences)\n"
            f"   - Critical Findings (with CVEs, IOCs, and risk assessment)\n"
            f"   - Recommended Actions\n\n"
            f"Work systematically through the feeds. Use tools, don't guess."
        ),
        compact_service=compact_chain(
            PruneToolResults(keep_recent=6),
            TruncateCompact(keep_recent=3),
        ),
        memory_sources=[
            StaticMemorySource(
                "## Briefing Standards\n"
                "- Always include CVE IDs with CVSS scores\n"
                "- Defang all IOCs (use [.] instead of .)\n"
                "- Prioritize supply chain and zero-day threats\n"
                "- Include MITRE ATT&CK mapping when available\n"
            ),
        ],
    )

    state = DefaultState(max_steps=15)
    session_log = SessionLog()
    conv = Conversation()

    return {
        "llm": recording,
        "tools": tools,
        "config": config,
        "state": state,
        "session_log": session_log,
        "conversation": conv,
        "hooks": [stag_hook, limit_hook, stream_hook],
        "events": events,
        "recording": recording,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    agent = build_agent()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        THREAT INTELLIGENCE BRIEFING AGENT                   ║")
    print("║        Powered by looplet • 100% local                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    with tempfile.TemporaryDirectory() as traj_dir:
        recorder = TrajectoryRecorder(
            recording_llm=agent["recording"],
            output_dir=traj_dir,
        )

        all_hooks = agent["hooks"] + [recorder]
        task = {"description": "Produce today's threat intelligence briefing."}

        print("Agent working...\n")
        steps = []
        briefing = None

        for step in composable_loop(
            llm=agent["llm"],
            task=task,
            tools=agent["tools"],
            state=agent["state"],
            config=agent["config"],
            hooks=all_hooks,
            session_log=agent["session_log"],
            conversation=agent["conversation"],
        ):
            steps.append(step)
            tool = step.tool_call.tool
            err = step.tool_result.error
            warns = step.tool_result.warnings

            # Progress indicator
            if tool == "done":
                briefing = step.tool_result.data
                print(f"  ✓ Step {step.number}: Briefing complete!")
            elif tool == "think":
                analysis = step.tool_call.args.get("analysis", "")[:80]
                print(f"  💭 Step {step.number}: thinking — {analysis}...")
            elif err:
                print(f"  ✗ Step {step.number}: {tool} — {str(err)[:60]}")
            else:
                data = step.tool_result.data or {}
                preview = ""
                if isinstance(data, dict):
                    if "item_count" in data:
                        preview = f"{data['item_count']} items"
                    elif "total_iocs" in data:
                        preview = f"{data['total_iocs']} IOCs"
                    elif "assessment" in data:
                        preview = data["assessment"][:60]
                    elif "techniques" in data:
                        preview = f"{data['count']} techniques"
                    else:
                        preview = str(data)[:60]
                print(f"  → Step {step.number}: {tool} — {preview}")
                if warns:
                    for w in warns:
                        print(f"    ⚠ {w}")

        # ── Output ──────────────────────────────────────────────
        print()
        print("═" * 64)

        if briefing and isinstance(briefing, dict):
            briefing_text = briefing.get("briefing", briefing.get("summary", str(briefing)))
            print()
            print(briefing_text)
        else:
            # Fallback: show last step's data
            print("\n⚠ Agent did not produce a structured briefing.")
            if steps:
                last = steps[-1]
                print(f"Last step: {last.tool_call.tool}")
                if last.tool_result.data:
                    print(json.dumps(last.tool_result.data, indent=2)[:2000])

        print()
        print("═" * 64)

        # ── Stats ───────────────────────────────────────────────
        print("\n📊 Agent Statistics:")
        print(f"  Steps: {len(steps)}")
        print(f"  LLM calls: {len(agent['recording'].calls)}")
        scoped = [c for c in agent["recording"].calls if c.scope]
        print(f"  Tool-internal LLM calls: {len(scoped)}")
        print(f"  Session log entries: {len(agent['session_log'].entries)}")

        event_types = {type(e).__name__ for e in agent["events"]}
        print(f"  Streaming events: {len(agent['events'])} ({len(event_types)} types)")

        # Check trajectory saved
        traj_path = Path(traj_dir) / "trajectory.json"
        if traj_path.exists():
            traj = json.loads(traj_path.read_text())
            print(f"  Trajectory: saved ({len(traj.get('steps', []))} steps)")
        print()


if __name__ == "__main__":
    main()
