"""Portability classifier — label a hook ``portable`` or ``inprocess``.

The honest version of "100% lossless cartridge⇄library" (see
HOOK_CARTRIDGE_DESIGN.md §9.0) is: *every* library hook has a faithful
cartridge representation, but only the **pure + declared-view** subset
is portable across runtimes; the rest are faithfully pinned to the
authoring runtime via ``kind: inprocess``. This module is the verifier
that draws that line, so the label is enforced rather than asserted.

It is deliberately conservative: static purity analysis of arbitrary
Python is undecidable, so anything not *positively* known to be portable
is labelled ``inprocess`` (faithful, Python-pinned). Positive evidence of
portability is one of:

* the hook is an :class:`looplet.lep.LEPHookAdapter` — already
  out-of-process by construction, hence portable by definition;
* the hook came from the stdlib archetype registry with a declared
  view (``builtin_hooks``/``use: stdlib/*``);
* the hook opts in explicitly via ``__lep_portable__ = True`` *and*
  exposes a :class:`~looplet.hook_view.ViewSpec` as ``view_spec``.

Negative evidence (forces ``inprocess`` even with an opt-in):

* a ``bind`` method — the hook subscribes to live loop ``ctx`` and thus
  reads outside any serialisable view (hazard H2/H3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from looplet.hook_view import ViewSpec

__all__ = ["Classification", "classify", "classify_preset_hooks"]

PORTABLE = "portable"
INPROCESS = "inprocess"


@dataclass(frozen=True)
class Classification:
    """Result of classifying one hook for cross-runtime portability."""

    kind: str  # PORTABLE | INPROCESS
    reasons: tuple[str, ...] = field(default_factory=tuple)
    view: ViewSpec | None = None

    @property
    def is_portable(self) -> bool:
        return self.kind == PORTABLE


def _has_bind(hook: Any) -> bool:
    return callable(getattr(hook, "bind", None))


def classify(
    hook: Any,
    *,
    view: ViewSpec | None = None,
    stdlib_name: str | None = None,
) -> Classification:
    """Classify ``hook`` as :data:`PORTABLE` or :data:`INPROCESS`.

    Args:
        hook: The instantiated hook object.
        view: An externally-declared view (e.g. from a cartridge
            ``view:`` block), if any.
        stdlib_name: The stdlib archetype name this hook was built from,
            if it came through ``use: stdlib/*`` / ``builtin_hooks``.
    """
    # Avoid an import cycle: looplet.lep imports nothing from here, but
    # importing it at module scope would still be fine — keep it local
    # for clarity.
    from looplet.lep import LEPHookAdapter

    declared_view = view or getattr(hook, "view_spec", None)

    if isinstance(hook, LEPHookAdapter):
        return Classification(
            PORTABLE,
            ("out-of-process LEP adapter",),
            declared_view or getattr(hook, "_view", None),
        )

    if _has_bind(hook):
        return Classification(
            INPROCESS,
            ("declares bind() — reads live loop ctx outside any serialisable view",),
            declared_view,
        )

    if stdlib_name is not None and declared_view is not None:
        return Classification(
            PORTABLE,
            (f"stdlib archetype '{stdlib_name}' with a declared view",),
            declared_view,
        )

    if getattr(hook, "__lep_portable__", False) and declared_view is not None:
        return Classification(
            PORTABLE,
            ("explicit __lep_portable__ opt-in with a declared view",),
            declared_view,
        )

    reasons = ["arbitrary Python hook — no positive portability evidence"]
    if declared_view is None:
        reasons.append("no declared view")
    return Classification(INPROCESS, tuple(reasons), declared_view)


def classify_preset_hooks(preset: Any) -> list[tuple[str, Classification]]:
    """Classify every hook on an :class:`~looplet.preset.AgentPreset`.

    Returns a list of ``(hook_type_name, Classification)`` pairs in hook
    order. Pure inspection — it does **not** mutate ``preset`` (so it is
    safe to call inside round-trip / serialise tests).
    """
    hooks = list(getattr(preset, "hooks", None) or [])
    return [(type(h).__name__, classify(h)) for h in hooks]
