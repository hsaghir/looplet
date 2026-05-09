"""Verify looplet's Skill loader handles canonical agentskills.io SKILL.md.

The Agent Skills spec (https://agentskills.io) defines a markdown file
with YAML frontmatter carrying at minimum ``name`` and ``description``.
Pi, Claude Code, and looplet all consume this format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet import FileSkillStore, Skill

CANONICAL_SKILL_MD = """\
---
name: pdf-extractor
description: Extract structured data from PDF files using pdfplumber.
tags: [pdf, extraction, parsing]
---

# PDF Extractor

Use this skill when the user asks you to extract tables or text from a PDF.

## Steps

1. Use `pdfplumber.open(path)` to open the file.
2. Iterate `pdf.pages`; call `page.extract_tables()` for tables.
3. Return as JSON.
"""


def test_loads_canonical_skill_md(tmp_path: Path) -> None:
    skill = Skill.from_markdown(CANONICAL_SKILL_MD, source_path=str(tmp_path / "SKILL.md"))
    assert skill.name == "pdf-extractor"
    assert "Extract structured data" in skill.description
    assert "pdf" in skill.tags and "extraction" in skill.tags
    assert "pdfplumber.open" in skill.instructions


def test_missing_name_raises() -> None:
    bad = "---\ndescription: foo\n---\nbody"
    with pytest.raises(ValueError, match="name"):
        Skill.from_markdown(bad)


def test_missing_description_raises() -> None:
    bad = "---\nname: foo\n---\nbody"
    with pytest.raises(ValueError, match="description"):
        Skill.from_markdown(bad)


def test_file_skill_store_discovers_skill_md(tmp_path: Path) -> None:
    skill_dir = tmp_path / "pdf-extractor"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(CANONICAL_SKILL_MD)

    store = FileSkillStore(tmp_path)
    cards = store.list()
    names = [c.name for c in cards]
    assert "pdf-extractor" in names


def test_card_carries_path_for_safe_discovery(tmp_path: Path) -> None:
    skill_dir = tmp_path / "pdf-extractor"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(CANONICAL_SKILL_MD)

    store = FileSkillStore(tmp_path)
    cards = store.list()
    card = next(c for c in cards if c.name == "pdf-extractor")
    # Cards expose path + description but not full instruction body.
    assert card.path is not None
    assert "Extract structured data" in card.description
