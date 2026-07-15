#!/usr/bin/env python
"""Non-coding ("soft") benchmark tasks: system design, conceptual
explanation, debugging reasoning, and professional writing.

These have no single correct answer, so the per-task verifier here only
checks that the agent produced a substantive answer.md artifact. Quality
is scored separately by judge.py (a blind LLM-as-judge). Same schema as
tasks.py so bench.py can run it via BENCH_TASKS=soft_tasks.
"""

from __future__ import annotations

from pathlib import Path


def _answer_ok(ws: Path, py: str):
    for name in ("answer.md", "ANSWER.md", "answer.txt"):
        p = ws / name
        if p.exists():
            txt = p.read_text(errors="ignore").strip()
            if len(txt) >= 400:
                return True, f"{name}: {len(txt)} chars"
            return False, f"{name} too short ({len(txt)} chars)"
    return False, "no answer.md produced"


_SUFFIX = " Write your answer as Markdown to a file named answer.md, then stop."

TASKS = [
    {
        "id": "s1_url_shortener",
        "kind": "design",
        "title": "System design: URL shortener at scale",
        "prompt": (
            "Design a URL shortener service (like bit.ly) that must handle "
            "roughly 10,000 new links per second and 100 million redirects per "
            "day. Cover: the high-level architecture, the data model, how you "
            "generate short codes (and why), read/write paths, caching, how you "
            "store and serve click analytics, and the main scaling and "
            "reliability tradeoffs." + _SUFFIX
        ),
        "seed": {},
        "verify": _answer_ok,
    },
    {
        "id": "s2_rate_limiter",
        "kind": "design",
        "title": "Design + compare rate-limiting algorithms",
        "prompt": (
            "Design a rate limiter for a public HTTP API. Compare at least two "
            "algorithms (for example token bucket, leaky bucket, fixed window, "
            "sliding-window log/counter), with their tradeoffs. Explain how you "
            "would implement it in a distributed, multi-instance deployment "
            "(e.g. using Redis), how you handle bursts, clock skew, and what "
            "response/headers a client should receive when throttled." + _SUFFIX
        ),
        "seed": {},
        "verify": _answer_ok,
    },
    {
        "id": "s3_tls_explain",
        "kind": "explain",
        "title": "Explain the TLS/HTTPS handshake",
        "prompt": (
            "Explain, accurately and clearly for a mid-level software engineer, "
            "how HTTPS/TLS establishes a secure connection. Cover the handshake "
            "steps, the role of certificates and certificate authorities, why "
            "both asymmetric and symmetric cryptography are used (and where "
            "each is used), and what protects against man-in-the-middle "
            "attacks. Note anything that differs in TLS 1.3." + _SUFFIX
        ),
        "seed": {},
        "verify": _answer_ok,
    },
    {
        "id": "s4_cap_theorem",
        "kind": "explain",
        "title": "Explain CAP theorem with examples",
        "prompt": (
            "Explain the CAP theorem precisely: what C, A, and P actually mean, "
            "why you can only fully guarantee two of the three during a network "
            "partition, and the common misconceptions about it. Give one "
            "concrete real-world example of a CP system and one of an AP "
            "system, and explain exactly what tradeoff each makes and when you "
            "would choose it." + _SUFFIX
        ),
        "seed": {},
        "verify": _answer_ok,
    },
    {
        "id": "s5_tail_latency",
        "kind": "reasoning",
        "title": "Diagnose p99 tail latency",
        "prompt": (
            "A web service has p50 latency of 200 ms but p99 latency of 4 "
            "seconds under normal load. Walk through how you would diagnose and "
            "fix this tail-latency problem. List the most likely hypotheses "
            "(with reasoning), how you would test or measure each one, and the "
            "likely fixes — prioritized by expected impact and effort. Be "
            "specific rather than generic." + _SUFFIX
        ),
        "seed": {},
        "verify": _answer_ok,
    },
    {
        "id": "s6_postmortem",
        "kind": "writing",
        "title": "Write an incident postmortem",
        "prompt": (
            "Write a clear, professional, blameless incident postmortem for the "
            "following outage: after a routine deploy at 14:05, a web app began "
            "returning 500 errors because the database connection pool was "
            "exhausted; the on-call engineer rolled back at 14:50 and service "
            "recovered. About 40% of requests failed for 45 minutes. Include "
            "the standard sections (summary, impact, timeline, root cause, "
            "detection, resolution, and at least three concrete, actionable "
            "follow-up items with owners/areas). Use a professional tone." + _SUFFIX
        ),
        "seed": {},
        "verify": _answer_ok,
    },
]
