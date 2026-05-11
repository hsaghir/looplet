"""Inlined v1 ``threat_intel`` agent code — the underlying data and
tool functions previously hosted at ``examples/threat_intel/agent.py``.

Each plain ``def`` here is wired to the workspace by a thin
``tools/<name>/execute.py`` shim that does
``from lib import <name> as execute``; the workspace ``tool.yaml``
carries name / description / parameters metadata so no ``@tool``
decorator is needed at this layer.
"""

from __future__ import annotations

import json
import re

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
