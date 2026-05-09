"""Skills — composable bundles of tools + context + instructions.

A skill is the unit of agent capability. Instead of building one
monolithic agent with 40 tools and a massive prompt, build small
skills and compose them::

    python_skill = Skill(
        name="python",
        tools=[ToolSpec(name="bash", ...), ToolSpec(name="read", ...)],
        instructions="You are a Python developer. Run tests before done().",
        memory=StaticMemorySource("PEP 8 rules..."),
    )

    # Load into a loop
    python_skill.register(registry)
    config = LoopConfig(
        system_prompt=python_skill.instructions,
        memory_sources=[python_skill.memory],
    )

Skills are discoverable — the agent can list available skills and
load them on demand via a hook or tool.

This follows Anthropic's "build skills, not agents" pattern from
Code Summit 2025: composable skill modules, not monolithic loops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, cast

from looplet.tools import BaseToolRegistry, ToolSpec

__all__ = [
    "FileSkillStore",
    "Skill",
    "SkillActivationHook",
    "SkillCard",
    "SkillManager",
    "build_skill_manager_for_workspace",
    "install_skills",
    "make_skill_tools",
]


@dataclass
class Skill:
    """A composable bundle of tools + context + instructions.

    Args:
        name: Short identifier (e.g. "python", "security", "data").
        tools: List of ToolSpec instances this skill provides.
        instructions: System prompt fragment — appended to the
            agent's system prompt when this skill is active.
        memory: Optional PersistentMemorySource that carries
            domain knowledge surviving all compactions.
        description: One-line description for discovery/listing.

    Usage::

        skill = Skill(
            name="python",
            tools=[bash_spec, read_spec, write_spec],
            instructions="Write tests first. Use type hints.",
        )
        skill.register(my_registry)
    """

    name: str
    tools: list[ToolSpec] = field(default_factory=list)
    instructions: str = ""
    memory: Any = None  # PersistentMemorySource or None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_markdown(
        cls,
        text: str,
        *,
        source_path: str | Path | None = None,
        default_name: str | None = None,
    ) -> "Skill":
        """Build a skill from an Agent Skills ``SKILL.md`` document.

        The supported on-disk format mirrors Anthropic/Claude skills:
        a markdown file with YAML-style frontmatter containing at least
        ``name`` and ``description``.  looplet intentionally parses only a
        small dependency-free subset of YAML (``key: value`` and simple
        list values) so the core package keeps zero runtime dependencies.
        """
        metadata, body = _split_skill_markdown(text)
        name = str(metadata.get("name") or default_name or "").strip()
        if not name:
            raise ValueError("Skill markdown is missing required frontmatter field 'name'")
        description = str(metadata.get("description") or "").strip()
        if not description:
            raise ValueError(f"Skill {name!r} is missing required frontmatter field 'description'")
        return cls(
            name=name,
            instructions=body.strip(),
            description=description,
            tags=_coerce_tags(metadata.get("tags")),
            metadata=metadata,
            source_path=str(source_path) if source_path is not None else None,
        )

    def register(self, registry: BaseToolRegistry) -> int:
        """Register all tools from this skill into a registry.

        Returns the number of tools registered.
        """
        for spec in self.tools:
            registry.register(spec)
        return len(self.tools)

    def tool_names(self) -> list[str]:
        """Return the names of all tools in this skill."""
        return [t.name for t in self.tools]

    def as_catalog_entry(self) -> str:
        """One-line summary for skill discovery listings."""
        tools = ", ".join(self.tool_names()) or "(no tools)"
        desc = self.description or self.instructions[:80]
        return f"[{self.name}] {desc} — tools: {tools}"

    def card(self) -> "SkillCard":
        """Return a lightweight discovery card for this skill."""
        return SkillCard(
            name=self.name,
            description=self.description,
            path=self.source_path,
            tags=list(self.tags),
            metadata=dict(self.metadata),
        )


@dataclass
class SkillCard:
    """Lightweight skill discovery record.

    Cards are safe to show the agent before activation: they carry only
    metadata and a path, not the full instruction payload or scripts.
    """

    name: str
    description: str
    path: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for tool results and product UIs."""
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class FileSkillStore:
    """Filesystem-backed store for Claude/Agent Skills folders.

    A root can be either a single skill directory containing ``SKILL.md``
    or a directory tree containing many such directories.  Scripts and
    resources are discovered as files on disk but are deliberately inert:
    callers must wrap anything executable as a normal :class:`ToolSpec`.
    """

    def __init__(self, *roots: str | Path | Iterable[str | Path]) -> None:
        if len(roots) == 1 and not isinstance(roots[0], (str, Path)):
            root_iter = list(cast(Iterable[str | Path], roots[0]))
        else:
            root_iter = [cast(str | Path, root) for root in roots]
        if not root_iter:
            raise ValueError("FileSkillStore requires at least one root path")
        self.roots = [Path(root) for root in root_iter]
        self._skills: dict[str, Skill] = {}
        self.refresh()

    def refresh(self) -> None:
        """Re-scan roots and rebuild the in-memory skill index."""
        skills: dict[str, Skill] = {}
        for path in self._iter_skill_files():
            text = path.read_text(encoding="utf-8")
            skill = Skill.from_markdown(text, source_path=path, default_name=path.parent.name)
            if skill.name in skills:
                raise ValueError(
                    f"Duplicate skill name {skill.name!r}: {skills[skill.name].source_path} and {path}"
                )
            skills[skill.name] = skill
        self._skills = dict(sorted(skills.items()))

    def list(self) -> list[SkillCard]:
        """Return all known skills as lightweight cards."""
        return [skill.card() for skill in self._skills.values()]

    def load(self, name: str) -> Skill:
        """Load a full skill by name."""
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(self._skills) or "(none)"
            raise KeyError(f"Unknown skill {name!r}. Available skills: {available}") from exc

    def search(self, query: str, *, limit: int = 5) -> list[SkillCard]:
        """Return the best lexical matches for ``query``.

        This intentionally stays simple and dependency-free.  Product
        layers can replace it with embeddings or a vector index without
        touching the loop.
        """
        terms = _terms(query)
        if not terms:
            return self.list()[:limit]

        scored: list[tuple[int, str, Skill]] = []
        for skill in self._skills.values():
            score = _score_skill(skill, terms)
            if score > 0:
                scored.append((score, skill.name, skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [skill.card() for _, _, skill in scored[: max(0, limit)]]

    def _iter_skill_files(self) -> list[Path]:
        files: set[Path] = set()
        for root in self.roots:
            if root.is_file() and root.name == "SKILL.md":
                files.add(root)
                continue
            direct = root / "SKILL.md"
            if direct.is_file():
                files.add(direct)
            if root.is_dir():
                files.update(root.rglob("SKILL.md"))
        return sorted(files, key=lambda p: str(p))


class SkillManager:
    """Tracks lazy skill discovery and activation for one agent run."""

    def __init__(self, store: FileSkillStore) -> None:
        self.store = store
        self._active: dict[str, Skill] = {}

    @property
    def active_names(self) -> list[str]:
        """Active skills in activation order."""
        return list(self._active.keys())

    def search(self, query: str, *, limit: int = 5) -> list[SkillCard]:
        """Search available skills without activating them."""
        return self.store.search(query, limit=limit)

    def activate(self, name: str) -> Skill:
        """Activate a skill by name and return the full skill."""
        if name not in self._active:
            self._active[name] = self.store.load(name)
        return self._active[name]

    def active_skills(self) -> list[Skill]:
        """Return full active skill payloads."""
        return list(self._active.values())

    def render_active_instructions(self) -> str:
        """Render active skill instructions for prompt injection."""
        if not self._active:
            return ""
        sections = ["=== ACTIVE SKILLS ==="]
        for skill in self._active.values():
            sections.append(f"## {skill.name}\n\n{skill.instructions}".strip())
        return "\n\n".join(sections)


class SkillActivationHook:
    """Loop hook that injects only active skill instructions."""

    def __init__(self, manager: SkillManager) -> None:
        self.manager = manager

    def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> str | None:  # noqa: ARG002
        """Inject active skill instructions into the next prompt."""
        rendered = self.manager.render_active_instructions()
        return rendered or None


def make_skill_tools(
    manager: SkillManager,
    *,
    search_tool_name: str = "search_skills",
    activate_tool_name: str = "activate_skill",
) -> list[ToolSpec]:
    """Create optional tools for skill discovery and activation.

    These are convenience tools, not a new loop primitive.  Register
    them only for agents that should decide when to load skill context.
    """

    def search_skills(*, query: str, limit: int = 5) -> dict[str, Any]:
        cards = manager.search(query, limit=int(limit))
        return {"skills": [card.to_dict() for card in cards]}

    def activate_skill(*, name: str) -> dict[str, Any]:
        skill = manager.activate(name)
        return {
            "activated": skill.name,
            "description": skill.description,
            "active_skills": manager.active_names,
        }

    return [
        ToolSpec(
            name=search_tool_name,
            description="Search available skill bundles by task description without loading them.",
            parameters={
                "query": "task or capability to search for",
                "limit": "(optional) maximum number of skill cards to return",
            },
            execute=search_skills,
            concurrent_safe=True,
            free=True,
        ),
        ToolSpec(
            name=activate_tool_name,
            description="Activate one skill bundle so its instructions are injected into future prompts.",
            parameters={"name": "skill name returned by search_skills"},
            execute=activate_skill,
            concurrent_safe=False,
            free=True,
        ),
    ]


def install_skills(
    skills: list["Skill"],
    registry: BaseToolRegistry,
    *,
    base_system_prompt: str = "",
    base_memory_sources: list[Any] | None = None,
    separator: str = "\n\n",
) -> dict[str, Any]:
    """Load every skill into ``registry`` and return the config updates.

    Without this helper users have to (a) call ``skill.register(registry)``
    for each skill, (b) concatenate the skill instructions onto the
    system prompt themselves, and (c) extend the memory_sources list
    — forgetting any of the three silently drops part of the skill.

    Returns a dict with ``system_prompt`` and ``memory_sources`` keys,
    suitable for ``LoopConfig(**install_skills(...))`` or merging into
    an existing config via ``dataclasses.replace``.

    Example::

        cfg_updates = install_skills([python_skill, shell_skill], registry)
        config = LoopConfig(max_steps=10, **cfg_updates)

    Args:
        skills: List of :class:`Skill` instances to install.
        registry: Tool registry to register skill tools into.
        base_system_prompt: Existing system prompt to prepend.
        base_memory_sources: Existing memory sources to prepend.
        separator: String joining instruction fragments (default blank line).
    """
    prompt_parts: list[str] = []
    if base_system_prompt:
        prompt_parts.append(base_system_prompt)
    memory_sources: list[Any] = list(base_memory_sources or [])
    for skill in skills:
        skill.register(registry)
        if skill.instructions:
            prompt_parts.append(skill.instructions)
        if skill.memory is not None:
            memory_sources.append(skill.memory)
    return {
        "system_prompt": separator.join(prompt_parts),
        "memory_sources": memory_sources,
    }


def _split_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    end = normalized.find("\n---", 4)
    if end < 0:
        raise ValueError("Skill markdown frontmatter starts with '---' but has no closing '---'")
    raw_metadata = normalized[4:end]
    body_start = end + len("\n---")
    if normalized[body_start : body_start + 1] == "\n":
        body_start += 1
    return _parse_frontmatter(raw_metadata), normalized[body_start:]


def _parse_frontmatter(raw: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) and current_key:
            metadata[current_key] = f"{metadata[current_key]} {line.strip()}".strip()
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        current_key = key.strip()
        metadata[current_key] = _parse_scalar(value.strip())
    return metadata


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return value


def _coerce_tags(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())


def _score_skill(skill: Skill, terms: list[str]) -> int:
    name = skill.name.lower()
    description = skill.description.lower()
    tags = {tag.lower() for tag in skill.tags}
    body = skill.instructions.lower()
    score = 0
    for term in terms:
        if term == name:
            score += 12
        elif term in name:
            score += 6
        if term in tags:
            score += 5
        if term in description:
            score += 4
        if term in body:
            score += 1
    return score


def build_skill_manager_for_workspace(
    workspace_dir: "str | Path | None" = None,
    *,
    runtime: dict | None = None,
    skills_subdir: str = "skills",
) -> "SkillManager":
    """One-line builder for ``resources/skill_manager.py``.

    Resolves the skills directory (``<workspace>/<skills_subdir>``)
    from the workspace path encoded in ``runtime["workspace"]`` (the
    convention used by the workspace loader) — or from
    ``workspace_dir`` if you call this manually.

    Returns an empty manager (no SKILL.md files found) without raising,
    so a workspace can declare the resource even when its ``skills/``
    directory is empty.

    Use in ``resources/skill_manager.py``::

        from looplet.skills import build_skill_manager_for_workspace

        def build(runtime=None):
            return build_skill_manager_for_workspace(runtime=runtime)
    """
    base: Path | None = None
    if workspace_dir is not None:
        base = Path(workspace_dir)
    elif runtime is not None:
        ws = runtime.get("workspace") or runtime.get("workspace_root")
        if ws:
            base = Path(str(ws))
    if base is None:
        base = Path.cwd()
    skills_dir = base / skills_subdir
    store = FileSkillStore(skills_dir if skills_dir.is_dir() else base)
    return SkillManager(store)
