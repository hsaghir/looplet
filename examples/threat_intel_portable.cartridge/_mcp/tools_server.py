"""Stdio MCP server for the threat_intel_portable cartridge.

Serves all 7 tools that were in-process ``tools/*/execute.py`` bodies in
the original ``threat_intel`` cartridge — ``fetch_feed``, ``search_cve``,
``extract_iocs``, ``map_mitre``, ``assess_risk``, ``think``, ``done`` —
over the MCP stdio transport. Moving them out of process is what makes
the twin fully portable: no Python tool body is required by the host.

The simulated threat feeds (``THREAT_FEEDS``) and the MITRE technique
table are vendored here verbatim from the original ``threat_intel_lib``.

LLM-backed behaviour reaches the host model through the Model Gateway
(MGP): the loader exports ``LOOPLET_LLM_SOCKET`` and this server connects
to it lazily, so ``extract_iocs`` can add its ``llm_severity`` field and
``assess_risk`` can return a real analyst assessment — full parity with
the in-process original. When no gateway is present (or no backend is
bound yet) the calls degrade to exactly the branches the originals take
when ``ctx.llm is None``.

Standard-library only.
Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import os
import re
import socket
import sys


class _HostLLM:
    """Minimal stdlib-only client to the host Model Gateway (MGP).

    Connects to ``$LOOPLET_LLM_SOCKET`` (set by the loader) and forwards
    ``generate`` to the host's live LLM backend. ``generate`` raises if no
    gateway/backend is reachable, so callers degrade exactly like the
    in-process original's ``ctx.llm is None`` branch.
    """

    def __init__(self):
        self._sock = None
        self._buf = b""
        self._id = 0
        path = os.environ.get("LOOPLET_LLM_SOCKET")
        if not path or not hasattr(socket, "AF_UNIX"):
            return
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30.0)
            sock.connect(path)
            self._sock = sock
            self._rpc("llm/initialize", {})
        except OSError:
            self._sock = None

    def _readline(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                line, self._buf = self._buf, b""
                return line
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _rpc(self, method, params):
        self._id += 1
        rid = self._id
        self._sock.sendall(
            (json.dumps({"id": rid, "method": method, "params": params}) + "\n").encode("utf-8")
        )
        line = self._readline()
        if not line:
            raise OSError("model gateway closed the connection")
        msg = json.loads(line.decode("utf-8"))
        if msg.get("error"):
            raise RuntimeError(msg["error"].get("message", "model gateway error"))
        return msg.get("result") or {}

    def available(self):
        """True iff the gateway has a live LLM backend bound *right now*.

        Re-checks per call because the host binds the backend lazily at
        run time (``AgentPreset.run(llm)``), which may happen after this
        client connected. Maps to the original's ``ctx.llm is not None``
        guard: when False, callers take their no-LLM degradation branch
        instead of treating absence as an error.
        """
        if self._sock is None:
            return False
        try:
            return bool(self._rpc("llm/initialize", {}).get("ready"))
        except (OSError, RuntimeError):
            return False

    def generate(self, prompt, **kwargs):
        if self._sock is None:
            raise RuntimeError("no host LLM gateway is reachable")
        return str(self._rpc("llm/generate", {"prompt": prompt, "kwargs": kwargs}).get("text", ""))


_HOST_LLM = None
_HOST_LLM_TRIED = False


def _host_llm():
    """Return the host Model Gateway client only when a backend is bound.

    Returns ``None`` when there is no reachable gateway *or* no backend is
    currently bound — i.e. exactly the cases where the in-process original
    sees ``ctx.llm is None`` and degrades.
    """
    global _HOST_LLM, _HOST_LLM_TRIED
    if not _HOST_LLM_TRIED:
        _HOST_LLM_TRIED = True
        client = _HostLLM()
        if client._sock is not None:
            _HOST_LLM = client
    if _HOST_LLM is not None and _HOST_LLM.available():
        return _HOST_LLM
    return None


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
            "description": "Bitwarden CLI npm package supply chain compromise allowing "
            "credential exfiltration",
            "cvss_v3": 9.8,
            "vendor": "Bitwarden",
            "product": "CLI",
            "published": "2026-04-23",
        },
        {
            "cve_id": "CVE-2026-3847",
            "description": "GitHub Actions runner token exposure via crafted workflow in "
            "public repositories",
            "cvss_v3": 8.1,
            "vendor": "GitHub",
            "product": "Actions",
            "published": "2026-04-22",
        },
        {
            "cve_id": "CVE-2026-3802",
            "description": "MeshCore mesh networking firmware buffer overflow allowing "
            "remote code execution",
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

_MITRE_DB = {
    "T1557": {"name": "Adversary-in-the-Middle", "tactic": "Credential Access, Collection"},
    "T1040": {"name": "Network Sniffing", "tactic": "Credential Access, Discovery"},
    "T1132": {"name": "Data Encoding", "tactic": "Command and Control"},
    "T1195": {"name": "Supply Chain Compromise", "tactic": "Initial Access"},
    "T1059": {"name": "Command and Scripting Interpreter", "tactic": "Execution"},
}


def fetch_feed(feed_name):
    available = list(THREAT_FEEDS.keys())
    if feed_name not in THREAT_FEEDS:
        return {"error": f"Unknown feed '{feed_name}'. Available: {available}"}
    items = THREAT_FEEDS[feed_name]
    return {"feed": feed_name, "item_count": len(items), "items": items}


def search_cve(cve_id):
    for item in THREAT_FEEDS.get("nvd_recent", []):
        if item["cve_id"] == cve_id:
            return item
    return {"cve_id": cve_id, "error": "CVE not found in recent data"}


def extract_iocs(text):
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

    # Use the host LLM (via the Model Gateway) to classify severity if
    # reachable — same branch the in-process original takes with ctx.llm.
    llm = _host_llm()
    if llm is not None and result["total_iocs"] > 0:
        try:
            classification = llm.generate(
                f"Given these IOCs extracted from a threat report, classify the overall "
                f"threat severity as CRITICAL, HIGH, MEDIUM, or LOW. Respond with just "
                f"the severity level, nothing else.\n\nIOCs: {json.dumps(result, indent=2)}",
                max_tokens=20,
            )
            result["llm_severity"] = classification.strip().upper()
        except Exception:
            pass

    return result


def map_mitre(technique_ids):
    ids = [t.strip() for t in technique_ids.split(",")]
    results = {}
    for tid in ids:
        results[tid] = _MITRE_DB.get(tid, {"name": "Unknown", "tactic": "Unknown"})
    return {"techniques": results, "count": len(results)}


def assess_risk(title, severity, affected_products):
    # Reach the host LLM through the Model Gateway when available; else
    # degrade to the original's no-LLM fallback message.
    llm = _host_llm()
    if llm is not None:
        try:
            assessment = llm.generate(
                f"You are a threat intelligence analyst. Assess the organizational risk "
                f"for this threat in 2-3 sentences. Consider likelihood of exploitation, "
                f"blast radius, and recommended priority.\n\n"
                f"Threat: {title}\nSeverity: {severity}\n"
                f"Affected: {affected_products}\n\n"
                f"Respond with: PRIORITY: [IMMEDIATE/HIGH/MEDIUM/LOW] followed by "
                f"a brief justification.",
                max_tokens=150,
            )
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


TOOLS = [
    {
        "name": "fetch_feed",
        "description": "Fetch a specific threat feed by name (cisa_alerts, nvd_recent, "
        "osint_reports).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "feed_name": {
                    "type": "string",
                    "description": "One of cisa_alerts, nvd_recent, osint_reports.",
                }
            },
            "required": ["feed_name"],
        },
    },
    {
        "name": "search_cve",
        "description": "Look up details for a specific CVE ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "CVE identifier (e.g. CVE-2025-12345).",
                }
            },
            "required": ["cve_id"],
        },
    },
    {
        "name": "extract_iocs",
        "description": "Extract IOCs (IPs, domains, CVEs, hashes) from text via pattern "
        "matching, then add an LLM severity classification via the host Model Gateway "
        "(degrades to pattern-only when no host LLM is reachable).",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Raw text to scan for IOCs."}},
            "required": ["text"],
        },
    },
    {
        "name": "map_mitre",
        "description": "Map comma-separated MITRE ATT&CK technique IDs to names and tactics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "technique_ids": {
                    "type": "string",
                    "description": "Comma-separated technique IDs (e.g. T1059,T1071).",
                }
            },
            "required": ["technique_ids"],
        },
    },
    {
        "name": "assess_risk",
        "description": "Assess organizational risk for a specific threat using the host "
        "LLM via the Model Gateway (degrades to a severity-only summary when no host "
        "LLM is reachable).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Threat title."},
                "severity": {
                    "type": "string",
                    "description": "critical | high | medium | low.",
                },
                "affected_products": {
                    "type": "string",
                    "description": "Comma-separated product list.",
                },
            },
            "required": ["title", "severity", "affected_products"],
        },
    },
    {
        "name": "think",
        "description": "Record an analytical step or hypothesis. No side effects.",
        "inputSchema": {
            "type": "object",
            "properties": {"analysis": {"type": "string", "description": "Brief analytical note."}},
            "required": ["analysis"],
        },
    },
    {
        "name": "done",
        "description": "Signal that the briefing is complete.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The structured threat-intel briefing.",
                }
            },
            "required": ["summary"],
        },
    },
]


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _content(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


def _dispatch(name, args):
    if name == "fetch_feed":
        return fetch_feed(args.get("feed_name", ""))
    if name == "search_cve":
        return search_cve(args.get("cve_id", ""))
    if name == "extract_iocs":
        return extract_iocs(args.get("text", ""))
    if name == "map_mitre":
        return map_mitre(args.get("technique_ids", ""))
    if name == "assess_risk":
        return assess_risk(
            args.get("title", ""),
            args.get("severity", ""),
            args.get("affected_products", ""),
        )
    if name == "think":
        return {"analysis": args.get("analysis"), "noted": True}
    if name == "done":
        return {"status": "completed", "summary": args.get("summary")}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        msg_id = req.get("id")
        if method == "initialize":
            respond(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "threat-intel-tools", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            result = _dispatch(name, args)
            if result is None:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
            else:
                respond(msg_id, _content(result))
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
