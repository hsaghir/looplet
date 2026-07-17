"""LLM parity: portable twins (via the Model Gateway) match the originals.

The three portable twins - ``dep_doctor_portable``, ``threat_intel_portable``,
``git_detective_portable`` - serve their tools out of process over MCP. The
LLM-backed tools (``find_alternatives``, ``extract_iocs`` / ``assess_risk``,
``commit_patterns``) used to *degrade* because an out-of-process server had no
handle on the host's ``ctx.llm``. The **Model Gateway (MGP)** closes that gap:
the loader starts a host-resident 1:N socket server, exports
``LOOPLET_LLM_SOCKET``, and binds the live backend at run time; the MCP server
connects to it and calls back into the *same* model.

Each test here proves two things against the *same scripted backend*:

1. **Functional + qualitative parity** - with a backend bound to the twin's
   gateway, the portable tool returns the identical LLM-derived field/value
   that the in-process original returns when handed the backend via
   ``ctx.llm``.
2. **Graceful degradation** - with no backend bound (the load-time default,
   e.g. a headless dispatch before any run), the portable tool falls back to
   exactly the ``ctx.llm is None`` branch.

This is the end-to-end dogfood of the new primitive: a real LLM call crosses
the process boundary (host gateway ← MCP tool subprocess) and comes back.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from looplet.cartridge import cartridge_to_preset
from looplet.types import ToolCall, ToolContext

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="model gateway requires AF_UNIX sockets",
)


class _ScriptedLLM:
    """Deterministic backend: returns canned replies in order, records calls.

    Shared by the portable preset (via the gateway) and the in-process
    original (via ``ctx.llm``) so any divergence is a parity bug, not a
    backend difference.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self._replies.pop(0) if self._replies else "DONE"


def _original_ctx(backend):
    return ToolContext(llm=backend)


# ── dep_doctor: find_alternatives ─────────────────────────────


@pytest.mark.timeout(60)
def test_dep_doctor_find_alternatives_parity():
    reply = '[{"name": "httpx", "reason": "actively maintained async client"}]'
    portable = cartridge_to_preset(_EXAMPLES / "dep_doctor_portable.cartridge")
    original = cartridge_to_preset(_EXAMPLES / "dep_doctor.cartridge")
    try:
        # The loader started a gateway for the portable twin (it has
        # mcp_servers); bind a backend exactly as AgentPreset.run would.
        assert portable.model_gateway is not None, "gateway should auto-start"
        portable.model_gateway.set_backend(_ScriptedLLM([reply]))

        p = portable.tools.dispatch(
            ToolCall(tool="find_alternatives", args={"package_name": "requests"})
        )
        o = original.tools.dispatch(
            ToolCall(tool="find_alternatives", args={"package_name": "requests"}),
            ctx=_original_ctx(_ScriptedLLM([reply])),
        )
        assert p.error is None and o.error is None
        # Identical LLM-derived structure.
        assert p.data == o.data
        assert p.data["alternatives"] == [
            {"name": "httpx", "reason": "actively maintained async client"}
        ]
    finally:
        portable.close()
        original.close()


@pytest.mark.timeout(60)
def test_dep_doctor_find_alternatives_degrades_without_backend():
    portable = cartridge_to_preset(_EXAMPLES / "dep_doctor_portable.cartridge")
    try:
        # No backend bound (load-time default) → degrade to empty list,
        # the original's ctx.llm-None branch.
        p = portable.tools.dispatch(
            ToolCall(tool="find_alternatives", args={"package_name": "requests"})
        )
        assert p.error is None
        assert p.data == {"package": "requests", "alternatives": []}
    finally:
        portable.close()


# ── threat_intel: extract_iocs + assess_risk ──────────────────


@pytest.mark.timeout(60)
def test_threat_intel_extract_iocs_severity_parity():
    text = "C2 at update.telecom-infra[.]net and CVE-2026-3891, hash SHA256:deadbeef1234"
    portable = cartridge_to_preset(_EXAMPLES / "threat_intel_portable.cartridge")
    original = cartridge_to_preset(_EXAMPLES / "threat_intel.cartridge")
    try:
        assert portable.model_gateway is not None
        portable.model_gateway.set_backend(_ScriptedLLM(["  high  "]))

        p = portable.tools.dispatch(ToolCall(tool="extract_iocs", args={"text": text}))
        o = original.tools.dispatch(
            ToolCall(tool="extract_iocs", args={"text": text}),
            ctx=_original_ctx(_ScriptedLLM(["  high  "])),
        )
        assert p.error is None and o.error is None
        # The LLM severity field is now present in BOTH, normalised the
        # same way (.strip().upper()).
        assert p.data["llm_severity"] == "HIGH"
        assert o.data["llm_severity"] == "HIGH"
        # The deterministic IOC payload is identical too.
        assert p.data["cves"] == o.data["cves"]
        assert p.data["total_iocs"] == o.data["total_iocs"]
    finally:
        portable.close()
        original.close()


@pytest.mark.timeout(60)
def test_threat_intel_assess_risk_parity():
    assessment = "PRIORITY: IMMEDIATE. Patch Bitwarden CLI now; credential theft is active."
    portable = cartridge_to_preset(_EXAMPLES / "threat_intel_portable.cartridge")
    original = cartridge_to_preset(_EXAMPLES / "threat_intel.cartridge")
    try:
        assert portable.model_gateway is not None
        portable.model_gateway.set_backend(_ScriptedLLM([assessment]))

        args = {
            "title": "Bitwarden supply chain",
            "severity": "CRITICAL",
            "affected_products": "Bitwarden CLI",
        }
        p = portable.tools.dispatch(ToolCall(tool="assess_risk", args=args))
        o = original.tools.dispatch(
            ToolCall(tool="assess_risk", args=args),
            ctx=_original_ctx(_ScriptedLLM([assessment])),
        )
        assert p.error is None and o.error is None
        assert p.data == o.data
        assert p.data["assessment"] == assessment
        assert "no LLM" not in p.data["assessment"]
    finally:
        portable.close()
        original.close()


@pytest.mark.timeout(60)
def test_threat_intel_degrades_without_backend():
    portable = cartridge_to_preset(_EXAMPLES / "threat_intel_portable.cartridge")
    try:
        ar = portable.tools.dispatch(
            ToolCall(
                tool="assess_risk",
                args={
                    "title": "x",
                    "severity": "HIGH",
                    "affected_products": "y",
                },
            )
        )
        assert ar.error is None
        assert "no LLM" in ar.data["assessment"]

        ei = portable.tools.dispatch(ToolCall(tool="extract_iocs", args={"text": "CVE-2026-3891"}))
        assert ei.error is None
        assert "llm_severity" not in ei.data
    finally:
        portable.close()


# ── git_detective: commit_patterns ────────────────────────────


def _make_repo(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "Test")
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("y = 2\n")
    git("add", "-A")
    git("commit", "-q", "-m", "feat: initial")
    (repo / "a.py").write_text("x = 2\n")
    git("add", "-A")
    git("commit", "-q", "-m", "fix: bump a")
    return repo


@pytest.mark.timeout(60)
def test_git_detective_commit_patterns_parity(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    reply = "SCORE: 9/10 Disciplined conventional commits with clear, scoped messages."

    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(repo))
    portable = cartridge_to_preset(_EXAMPLES / "git_detective_portable.cartridge")
    original = cartridge_to_preset(
        _EXAMPLES / "git_detective.cartridge", runtime={"repo": str(repo)}
    )
    try:
        assert portable.model_gateway is not None
        portable.model_gateway.set_backend(_ScriptedLLM([reply]))

        p = portable.tools.dispatch(ToolCall(tool="commit_patterns", args={}))
        o = original.tools.dispatch(
            ToolCall(tool="commit_patterns", args={}),
            ctx=_original_ctx(_ScriptedLLM([reply])),
        )
        assert p.error is None and o.error is None
        # The LLM assessment is present in both and identical.
        assert p.data["commit_quality_assessment"] == reply
        assert o.data["commit_quality_assessment"] == reply
        # Deterministic stats match too.
        assert p.data["conventional_commits_pct"] == o.data["conventional_commits_pct"]
        assert p.data["total_analyzed"] == o.data["total_analyzed"]
    finally:
        portable.close()
        original.close()


@pytest.mark.timeout(60)
def test_git_detective_commit_patterns_degrades_without_backend(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(repo))
    portable = cartridge_to_preset(_EXAMPLES / "git_detective_portable.cartridge")
    try:
        p = portable.tools.dispatch(ToolCall(tool="commit_patterns", args={}))
        assert p.error is None
        assert "commit_quality_assessment" not in p.data
        assert p.data["conventional_commits_pct"] == 100.0
    finally:
        portable.close()
