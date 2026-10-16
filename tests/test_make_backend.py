"""Tests for the turnkey ``make_backend`` factory (RPC ``set_backend`` wire).

Monkeypatches the provider ``from_env`` classmethods so no SDK or API key is
needed - we only assert provider resolution/routing and the error path.
"""

from __future__ import annotations

import warnings

import pytest

from looplet import backends
from looplet.backends import make_backend


def test_make_backend_explicit_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backends.AnthropicBackend,
        "from_env",
        classmethod(lambda cls, *, model=None: ("anthropic", model)),
    )
    monkeypatch.setattr(
        backends.OpenAIBackend,
        "from_env",
        classmethod(lambda cls, *, model=None: ("openai", model)),
    )
    assert make_backend(provider="anthropic", model="m1") == ("anthropic", "m1")
    assert make_backend(provider="openai") == ("openai", None)
    # aliases
    assert make_backend(provider="claude")[0] == "anthropic"


def test_make_backend_autodetects_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backends.AnthropicBackend, "from_env", classmethod(lambda cls, *, model=None: "anthropic")
    )
    monkeypatch.setattr(
        backends.OpenAIBackend, "from_env", classmethod(lambda cls, *, model=None: "openai")
    )
    monkeypatch.delenv("LOOPLET_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert make_backend() == "anthropic"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert make_backend() == "openai"


def test_make_backend_no_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LOOPLET_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="could not resolve a provider"):
        make_backend()


@pytest.mark.parametrize("openai_setting", ["OPENAI_API_KEY", "OPENAI_BASE_URL"])
@pytest.mark.parametrize("selection", ["automatic", "environment", "argument"])
def test_make_backend_warns_only_for_ambiguous_automatic_selection(
    monkeypatch: pytest.MonkeyPatch, openai_setting: str, selection: str
) -> None:
    monkeypatch.setattr(
        backends.AnthropicBackend, "from_env", classmethod(lambda cls, *, model=None: "anthropic")
    )
    monkeypatch.setattr(
        backends.OpenAIBackend, "from_env", classmethod(lambda cls, *, model=None: "openai")
    )
    for variable in ("LOOPLET_PROVIDER", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "review-placeholder")
    monkeypatch.setenv(openai_setting, "review-placeholder")

    if selection == "automatic":
        with pytest.warns(backends.ProviderSelectionWarning, match="LOOPLET_PROVIDER"):
            assert make_backend() == "anthropic"
    else:
        if selection == "environment":
            monkeypatch.setenv("LOOPLET_PROVIDER", "openai")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            assert make_backend(provider="openai" if selection == "argument" else None) == "openai"
        assert not captured
