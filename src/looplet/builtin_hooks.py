"""Built-in hooks any looplet workspace can opt into.

Symmetric to :mod:`looplet.builtin_tools`. A workspace lists hooks it
wants in ``config.yaml``::

    builtin_hooks:
      - skill_activation                # zero-arg form
      - cost:                            # single-key dict form with kwargs
          model: claude-sonnet-4.5
      - steering: {}                     # explicit empty kwargs

The loader looks each name up in :data:`AVAILABLE`, resolves
``${ref:...}`` / ``${runtime.x}`` placeholders in kwargs against the
live resource registry, and instantiates the hook. The resulting hook
joins the workspace's other hooks (those defined in ``hooks/<name>/``).

Built-ins live here (rather than in every workspace's ``hooks/``) so
they evolve with looplet: a new release ships an improved hook and
every workspace using ``builtin_hooks:`` picks it up immediately.

Currently shipped built-ins:

* ``skill_activation`` — auto-installs :class:`SkillActivationHook`
  bound to the ``skill_manager`` resource (the natural pair for the
  ``search_skills`` / ``activate_skill`` built-in tools).
* ``stagnation`` — :class:`looplet.stagnation.StagnationHook` (loop
  guard for repeating tool calls). Default kwargs:
  ``threshold=3``, ``ignore_tools=["think", "done"]``.
* ``per_tool_limit`` — :class:`looplet.limits.PerToolLimitHook`
  (caps each tool's call count). Tune per workspace via
  ``default_limit`` / ``limits``.
* ``threshold_compact`` — :class:`looplet.budget.ThresholdCompactHook`
  with an inline ``budget:`` dict (parsed into
  :class:`looplet.budget.ContextBudget`).

Adding a new built-in: write a small builder ``def build(*,
resources, **kwargs) -> LoopHook`` and register its name in
:data:`AVAILABLE`.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["AVAILABLE", "build_builtin_hook"]


# Builders are lazy so ``looplet.workspace`` can import this module
# without dragging unrelated subsystems on every load.
def _build_skill_activation(*, resources: dict[str, Any], **kwargs: Any) -> Any:
    from looplet.skills import SkillActivationHook  # noqa: PLC0415

    manager = kwargs.pop("manager", None) or resources.get("skill_manager")
    if manager is None:
        raise ValueError(
            "builtin hook 'skill_activation' needs a SkillManager — "
            "either add resources/skill_manager.py (the convention) "
            "or pass `manager: ${ref:my_manager}` in config.yaml."
        )
    return SkillActivationHook(manager, **kwargs)


def _build_stagnation(*, resources: dict[str, Any], **kwargs: Any) -> Any:  # noqa: ARG001
    from looplet.stagnation import StagnationHook  # noqa: PLC0415

    # Workspace-author-friendly defaults. Any field is overridable.
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


AVAILABLE: dict[str, Callable[..., Any]] = {
    "skill_activation": _build_skill_activation,
    "stagnation": _build_stagnation,
    "per_tool_limit": _build_per_tool_limit,
    "threshold_compact": _build_threshold_compact,
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
