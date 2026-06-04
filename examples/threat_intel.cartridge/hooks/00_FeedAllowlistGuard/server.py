"""FeedAllowlistGuard — a portable ``kind: lep`` permission policy.

This hook runs *out of process* over the Loop Effect Protocol (LEP).
The host ships only the declared view (``tool`` + ``args``) over
line-delimited JSON-RPC; this server returns a permission decision.

Policy: refuse any ``fetch_feed`` call whose ``feed_name`` is not on the
trusted allowlist. A threat-intel agent should only pull from vetted
sources (CISA, NVD, curated OSINT); an unknown feed name is either a
hallucination or an attempt to reach an unapproved source, so the guard
denies it. This is the classic out-of-process "egress allowlist"
pattern expressed as a portable cartridge hook.

Because the decision is a pure function of the declared view, the hook
is classified ``portable`` and round-trips losslessly as a declarative
``kind: lep`` block — no Python source needs to be vendored beyond this
self-contained server.
"""

from looplet.lep import LEPServerBase

_ALLOWED_FEEDS = {"cisa_alerts", "nvd_recent", "osint_reports"}


class FeedAllowlistGuardServer(LEPServerBase):
    def decide(self, slot, view):
        if slot == "check_permission" and view.get("tool") == "fetch_feed":
            feed = (view.get("args") or {}).get("feed_name")
            if not isinstance(feed, str) or feed.strip() not in _ALLOWED_FEEDS:
                allowed = ", ".join(sorted(_ALLOWED_FEEDS))
                return {
                    "kind": "Deny",
                    "block": f"feed_name must be one of the allowlisted sources: {allowed}",
                }
        return {"kind": "Continue"}


if __name__ == "__main__":
    raise SystemExit(FeedAllowlistGuardServer().serve())
