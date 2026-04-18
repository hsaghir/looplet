"""Tests for per-backend context window adaptation.

Today, ``ContextPressureHook`` uses a static ``DEFAULT_CONTEXT_WINDOW =
128_000``. Real models range from 8K (old GPT-3.5) to 2M (Gemini 1.5).
Hard-coding a single value means:

* On a Haiku-like 200K model, we never trigger compaction until the
  session is ~60K over the ceiling — reactive-compact thrashes.
* On an 8K model, we happily build prompts that will 413 every turn.

Backends should be able to declare their effective window, optionally
including the output-token reservation
(``MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000``). ``ContextPressureHook``
picks the backend value up automatically unless the caller overrides.

Invariants:

1. When a backend exposes ``context_window``, the hook uses it.
2. When a backend exposes both ``context_window`` and
   ``reserved_output_tokens``, the effective ceiling is
   ``context_window - reserved_output_tokens``.
3. An explicit ``context_window`` kwarg on the hook overrides any
   backend value (user intent wins).
4. A backend lacking both attributes falls back to ``DEFAULT_CONTEXT_WINDOW``.
"""

from __future__ import annotations

from openharness.context import DEFAULT_CONTEXT_WINDOW, ContextPressureHook


class _BackendWithWindow:
    context_window = 8_000


class _BackendWithReserved:
    context_window = 200_000
    reserved_output_tokens = 20_000


class _PlainBackend:
    """No context_window attribute."""


class TestBackendContextWindowAdaptation:
    def test_uses_backend_context_window(self):
        backend = _BackendWithWindow()
        hook = ContextPressureHook(llm=backend)
        assert hook.context_window == 8_000

    def test_subtracts_reserved_output(self):
        backend = _BackendWithReserved()
        hook = ContextPressureHook(llm=backend)
        assert hook.context_window == 180_000  # 200K - 20K

    def test_plain_backend_falls_back_to_default(self):
        hook = ContextPressureHook(llm=_PlainBackend())
        assert hook.context_window == DEFAULT_CONTEXT_WINDOW

    def test_explicit_kwarg_wins_over_backend(self):
        backend = _BackendWithWindow()  # says 8K
        hook = ContextPressureHook(llm=backend, context_window=50_000)
        assert hook.context_window == 50_000

    def test_none_llm_uses_default(self):
        hook = ContextPressureHook(llm=None)
        assert hook.context_window == DEFAULT_CONTEXT_WINDOW

    def test_thresholds_recompute_against_effective_window(self):
        """Buffers are applied to the resolved window, so compact/warning/
        blocking thresholds shift automatically when the backend
        advertises a small window."""
        hook = ContextPressureHook(
            llm=_BackendWithWindow(),
            compact_buffer=2_000,
            warning_buffer=3_000,
            blocking_buffer=500,
        )
        # 8K - 2K = 6K compact threshold, etc.
        assert hook._compact_threshold == 6_000
        assert hook._warning_threshold == 5_000
        assert hook._blocking_threshold == 7_500
