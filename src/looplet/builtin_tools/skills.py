"""``search_skills`` + ``activate_skill`` built-in tools.

Skills are agentskills.io ``SKILL.md`` bundles. A workspace makes them
loadable by:

1. Dropping ``SKILL.md`` files under ``skills/<name>/SKILL.md``.
2. Adding a ``resources/skill_manager.py`` that returns a
   :class:`looplet.skills.SkillManager` (a one-line builder ships in
   :func:`looplet.skills.build_skill_manager_for_workspace`).
3. Listing both built-ins in ``config.yaml``::

       builtin_tools:
         - search_skills
         - activate_skill

   And in ``hooks/skill_activation/`` add the standard hook so
   activated bodies actually land in the next prompt::

       # hooks/skill_activation/hook.py
       from looplet.skills import SkillActivationHook

       # hooks/skill_activation/config.yaml
       class_name: SkillActivationHook
       kwargs:
         manager: ${ref:skill_manager}

This eliminates the ``setup.py`` detour for the Pi-style skill flow.
"""

from __future__ import annotations

from typing import Any

from looplet.tools import ToolSpec
from looplet.types import ToolContext


def _search_execute(ctx: ToolContext, *, query: str, limit: int = 5) -> dict[str, Any]:
    manager = ctx.resources.get("skill_manager")
    if manager is None:
        return {
            "error": "no skill_manager resource — add resources/skill_manager.py",
            "remediation": (
                "Create resources/skill_manager.py exposing build() that "
                "returns a looplet.skills.SkillManager. See "
                "looplet.skills.build_skill_manager_for_workspace() for a one-liner."
            ),
        }
    cards = manager.search(query, limit=int(limit))
    return {"skills": [c.to_dict() for c in cards]}


def _activate_execute(ctx: ToolContext, *, name: str) -> dict[str, Any]:
    manager = ctx.resources.get("skill_manager")
    if manager is None:
        return {
            "error": "no skill_manager resource — add resources/skill_manager.py",
        }
    skill = manager.activate(name)
    return {
        "activated": skill.name,
        "description": skill.description,
        "active_skills": manager.active_names,
    }


SEARCH_SPEC = ToolSpec(
    name="search_skills",
    description=(
        "Search installed agentskills.io SKILL.md bundles by task description "
        "without loading them. Returns a list of skill cards: "
        "``[{name, description, path, tags}, ...]``. Pair with activate_skill "
        "to actually pull the skill body into your next prompt.\n\n"
        "Args:\n"
        "  query (str): free-text task description.\n"
        "  limit (int, optional): max results (default 5).\n"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Task description to match skill descriptions against.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    requires=["skill_manager"],
    execute=_search_execute,
)


ACTIVATE_SPEC = ToolSpec(
    name="activate_skill",
    description=(
        "Activate one installed skill by name. The skill's body (its "
        "SKILL.md instructions) is appended to subsequent prompts via "
        "SkillActivationHook. Returns ``{activated, description, "
        "active_skills}``.\n\n"
        "Args:\n"
        "  name (str): the ``name:`` field from the skill's frontmatter."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name (matches SKILL.md frontmatter ``name:`` field).",
            },
        },
        "required": ["name"],
    },
    requires=["skill_manager"],
    execute=_activate_execute,
)
