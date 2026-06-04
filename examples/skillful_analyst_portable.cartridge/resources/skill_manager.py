"""SkillManager built from this workspace's ``skills/`` directory.

Looked up by ``builtin_tools: [search_skills, activate_skill]`` and by
``hooks/skill_activation/config.yaml`` via ``${ref:skill_manager}``.
"""

from __future__ import annotations

from pathlib import Path

from looplet.skills import build_skill_manager_for_workspace


def build(runtime: dict | None = None):
    # Anchor to this workspace, not the host project root: we want the
    # SKILL.md files that live next to this resources/ folder.
    workspace_dir = Path(__file__).resolve().parent.parent
    return build_skill_manager_for_workspace(workspace_dir, runtime=runtime)
