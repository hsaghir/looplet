"""Built-in hooks any Looplet cartridge can opt into.

Symmetric to :mod:`looplet.builtin_tools`. A cartridge lists hooks it
wants in ``config.yaml``::

    builtin_hooks:
      - skill_activation                # zero-arg form
      - cost:                            # single-key dict form with kwargs
          model: claude-sonnet-4.5
      - steering: {}                     # explicit empty kwargs

The loader looks each name up in :data:`AVAILABLE`, resolves
``${ref:...}`` / ``${runtime.x}`` placeholders in kwargs against the
live resource registry, and instantiates the hook. The resulting hook
joins the cartridge's other hooks (those defined in ``hooks/<name>/``).

Built-ins live here (rather than in every cartridge's ``hooks/``) so
they evolve with looplet: a new release ships an improved hook and
every cartridge using ``builtin_hooks:`` picks it up immediately.

Currently shipped built-ins:

* ``skill_activation`` - auto-installs :class:`SkillActivationHook`
  bound to the ``skill_manager`` resource (the natural pair for the
  ``search_skills`` / ``activate_skill`` built-in tools).
* ``stagnation`` - :class:`looplet.stagnation.StagnationHook` (loop
  guard for repeating tool calls). Default kwargs:
  ``threshold=3``, ``ignore_tools=["think", "done"]``.
* ``per_tool_limit`` - :class:`looplet.limits.PerToolLimitHook`
  (caps each tool's call count). Tune per workspace via
  ``default_limit`` / ``limits``.
* ``threshold_compact`` - :class:`looplet.budget.ThresholdCompactHook`
  with an inline ``budget:`` dict (parsed into
  :class:`looplet.budget.ContextBudget`).
* ``static_briefing`` - :class:`looplet.cartridge.prompt_files.StaticBriefingHook`.
  Inline replacement for the v1.x magic ``prompts/briefing.md`` file:
  declare ``text:`` (inline body) xor ``path:`` (relative to cartridge
  root). Spec v2 prefers this declarative form so the briefing source
  is visible in ``config.yaml`` rather than auto-discovered by filename.
* ``recovery_hint`` - :class:`looplet.cartridge.prompt_files.RecoveryHintHook`.
  Inline replacement for the v1.x magic ``prompts/recovery.md`` file;
  same ``text:`` / ``path:`` kwargs as ``static_briefing``.

Adding a new built-in: write a small builder ``def build(*,
resources, **kwargs) -> LoopHook`` and register its name in
:data:`AVAILABLE`.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["AVAILABLE", "build_builtin_hook"]


# Builders are lazy so the cartridge loader can import this module
# without dragging unrelated subsystems on every load.
def _build_skill_activation(*, resources: dict[str, Any], **kwargs: Any) -> Any:
    from looplet.skills import SkillActivationHook  # noqa: PLC0415

    manager = kwargs.pop("manager", None) or resources.get("skill_manager")
    if manager is None:
        raise ValueError(
            "builtin hook 'skill_activation' needs a SkillManager - "
            "either add resources/skill_manager.py (the convention) "
            "or pass `manager: ${ref:my_manager}` in config.yaml."
        )
    return SkillActivationHook(manager, **kwargs)


def _build_stagnation(*, resources: dict[str, Any], **kwargs: Any) -> Any:  # noqa: ARG001
    from looplet.stagnation import StagnationHook  # noqa: PLC0415

    # Cartridge-author-friendly defaults. Any field is overridable.
    kwargs.setdefault("threshold", 3)
    kwargs.setdefault("ignore_tools", ["think", "done"])
    return StagnationHook(**kwargs)


def _build_per_tool_limit(*, resources: dict[str, Any], **kwargs: Any) -> Any:  # noqa: ARG001
    from looplet.limits import PerToolLimitHook  # noqa: PLC0415

    # PerToolLimitHook requires either default_limit or limits; supply
    # a sensible default for the zero-arg form.
    if not kwargs.get("limits") and "default_limit" not in kwargs:
        kwargs["default_limit"] = 10
    return PerToolLimitHook(**kwargs)


def _build_threshold_compact(*, resources: dict[str, Any], **kwargs: Any) -> Any:  # noqa: ARG001
    from looplet.budget import ContextBudget, ThresholdCompactHook  # noqa: PLC0415

    budget = kwargs.pop("budget", None)
    if budget is None:
        # Sensible default: 128k window, warn at 80k, error at 100k.
        budget = ContextBudget(context_window=128_000, warning_at=80_000, error_at=100_000)
    elif isinstance(budget, dict):
        budget = ContextBudget(**budget)
    return ThresholdCompactHook(budget=budget, **kwargs)


def _read_text_or_path(
    *,
    name: str,
    text: str | None,
    path: str | None,
    cartridge_root: Any,
) -> str:
    """Resolve a hook's body from inline ``text:`` or a ``path:`` on disk.

    ``path:`` is resolved relative to the cartridge root supplied via
    the ``cartridge_root`` resource (auto-injected by the loader).
    Exactly one of ``text:`` / ``path:`` must be set.
    """
    from pathlib import Path  # noqa: PLC0415

    if text is not None and path is not None:
        raise ValueError(f"builtin hook {name!r}: pass either ``text:`` or ``path:``, not both")
    if text is not None:
        return str(text)
    if path is None:
        raise ValueError(f"builtin hook {name!r}: requires either ``text:`` or ``path:``")
    if cartridge_root is None:
        raise ValueError(
            f"builtin hook {name!r}: ``path:`` resolution requires the loader to provide "
            f"a ``cartridge_root`` resource (this should be automatic)"
        )
    body = (Path(cartridge_root) / str(path)).read_text(encoding="utf-8")
    return body


def _build_static_briefing(*, resources: dict[str, Any], **kwargs: Any) -> Any:
    """Inline replacement for the magic ``prompts/briefing.md`` file.

    Declarative usage::

        builtin_hooks:
          - static_briefing:
              text: |-
                Be concise.
          # OR
          - static_briefing:
              path: prompts/briefing.md
    """
    from looplet.cartridge.prompt_files import StaticBriefingHook  # noqa: PLC0415

    body = _read_text_or_path(
        name="static_briefing",
        text=kwargs.pop("text", None),
        path=kwargs.pop("path", None),
        cartridge_root=resources.get("cartridge_root"),
    )
    return StaticBriefingHook(text=body, **kwargs)


def _build_recovery_hint(*, resources: dict[str, Any], **kwargs: Any) -> Any:
    """Inline replacement for the magic ``prompts/recovery.md`` file.

    Same kwargs as :func:`_build_static_briefing` (``text:`` xor ``path:``).
    """
    from looplet.cartridge.prompt_files import RecoveryHintHook  # noqa: PLC0415

    body = _read_text_or_path(
        name="recovery_hint",
        text=kwargs.pop("text", None),
        path=kwargs.pop("path", None),
        cartridge_root=resources.get("cartridge_root"),
    )
    return RecoveryHintHook(text=body, **kwargs)


AVAILABLE: dict[str, Callable[..., Any]] = {
    "skill_activation": _build_skill_activation,
    "stagnation": _build_stagnation,
    "per_tool_limit": _build_per_tool_limit,
    "threshold_compact": _build_threshold_compact,
    "static_briefing": _build_static_briefing,
    "recovery_hint": _build_recovery_hint,
}


def build_builtin_hook(
    name: str,
    *,
    resources: dict[str, Any],
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a built-in hook by name.

    Args:
        name: Entry in :data:`AVAILABLE`.
        resources: The active resource registry (from
            ``cartridge_to_preset``); built-in builders look up
            shared resources like ``skill_manager`` here.
        kwargs: Per-instance kwargs from ``config.yaml``.

    Raises:
        KeyError: If ``name`` is not in :data:`AVAILABLE`.
        ValueError: If a required resource is missing.
    """
    builder = AVAILABLE.get(name)
    if builder is None:
        raise KeyError(name)
    return builder(resources=resources, **(kwargs or {}))
