"""Smoke tests for the deterministic pretty GIF demo."""

from __future__ import annotations

import pytest

from looplet.examples import pretty_demo

pytestmark = pytest.mark.smoke


class TestPrettyDemo:
    def test_main_runs_without_delays(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(pretty_demo.time, "sleep", lambda _seconds: None)

        rc = pretty_demo.main(["--fast"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "looplet new" in out
        assert "looplet run-cartridge" in out
        assert "URL summarizer" in out
        assert "fetch_url" in out
        assert "url_summarizer.cartridge draft is structurally valid" in out
        assert "Example Domain: placeholder documentation page" in out
