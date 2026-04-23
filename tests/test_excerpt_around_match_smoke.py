"""Smoke tests for :func:`looplet.excerpt_around_match`.

Validates the pure-text witness-snippet helper used by any agent
builder that ships a search / grep / retrieval tool.
"""

import pytest

from looplet import excerpt_around_match

pytestmark = pytest.mark.smoke


def test_returns_empty_for_none_or_empty():
    assert excerpt_around_match(None, "x") == ""
    assert excerpt_around_match("", "x") == ""


def test_centers_window_around_match_with_ellipses():
    text = "a" * 100 + " NEEDLE " + "b" * 100
    out = excerpt_around_match(text, "NEEDLE", context=10)
    assert "NEEDLE" in out
    assert out.startswith("…") and out.endswith("…")
    assert len(out) <= 10 + len("NEEDLE") + 10 + 4  # window + space + ellipses


def test_no_leading_ellipsis_when_match_at_start():
    out = excerpt_around_match("NEEDLE tail here", "NEEDLE", context=40)
    assert not out.startswith("…")
    assert "NEEDLE" in out


def test_no_trailing_ellipsis_when_match_at_end():
    out = excerpt_around_match("head here NEEDLE", "NEEDLE", context=40)
    assert not out.endswith("…")


def test_case_insensitive_by_default():
    out = excerpt_around_match("XXX Needle YYY", "NEEDLE", context=5)
    assert "Needle" in out  # preserves original casing


def test_case_sensitive_when_requested():
    # Pattern not found in its exact case → falls back to head preview
    out = excerpt_around_match("xx needle yy", "NEEDLE", context=5, case_insensitive=False)
    assert "…" in out or "needle" in out  # degraded to plain head


def test_collapses_newlines_by_default():
    blob = "line1\nline2 NEEDLE line3\nline4"
    out = excerpt_around_match(blob, "NEEDLE", context=20)
    assert "\n" not in out
    assert "↵" in out  # newline marker


def test_collapse_newlines_can_be_disabled():
    blob = "line1\nNEEDLE\nline3"
    out = excerpt_around_match(blob, "NEEDLE", context=20, collapse_newlines=False)
    assert "\n" in out


def test_pattern_not_found_returns_head_preview():
    text = "x" * 200
    out = excerpt_around_match(text, "NEVER", context=10)
    # Still useful — a head preview — instead of empty
    assert out
    assert out.endswith("…")
    assert len(out) < len(text)


def test_empty_pattern_yields_head_preview():
    out = excerpt_around_match("abcdefghij" * 10, "", context=5)
    assert out  # non-empty head
    assert out.endswith("…")


def test_short_text_no_ellipses():
    out = excerpt_around_match("hello world", "world", context=40)
    assert out == "hello world"  # whole string fits, no ellipses


def test_exported_from_package_root():
    import looplet

    assert looplet.excerpt_around_match is excerpt_around_match
    assert "excerpt_around_match" in looplet.__all__
