"""Skills registry — discovers and serves prompt-based "tools" from disk.

Skills are SKILL.md files that contain instructions and context an LLM uses
to perform a task. They complement deterministic tools: tools call APIs and
return structured data; skills guide LLM reasoning for open-ended tasks
like literature review or hypothesis generation.

Directory layout:
    skills/
      lit-review/
        SKILL.md          # Skill definition (YAML front-matter + markdown body)
        references/       # Optional supporting files
      hypothesis-gen/
        SKILL.md
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    """A prompt-based skill loaded from a SKILL.md file."""

    name: str
    description: str
    content: str  # Full SKILL.md text (front-matter + body)
    skill_dir: Path

    @property
    def references_dir(self) -> Path | None:
        """Path to the references/ subdirectory, if it exists."""
        d = self.skill_dir / "references"
        return d if d.is_dir() else None

    def reference_files(self) -> list[Path]:
        """List all files in the references/ subdirectory."""
        d = self.references_dir
        if d is None:
            return []
        return sorted(d.iterdir())


class SkillsRegistry:
    """Discovers and serves prompt-based skills from a directory tree.

    Mirrors the pattern from agentic_demo.py but wrapped in a reusable class.
    """

    def __init__(self, skills_root: Path | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        if skills_root and skills_root.is_dir():
            self._load(skills_root)

    def _load(self, root: Path) -> None:
        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                text = skill_md.read_text()
                # Extract description from YAML front-matter
                match = re.search(
                    r"^---\s*\n.*?^description:\s*(.+?)(?:\n[a-z]|\n---)",
                    text,
                    re.MULTILINE | re.DOTALL,
                )
                desc = match.group(1).strip() if match else "(no description)"
                self._skills[child.name] = Skill(
                    name=child.name,
                    description=desc,
                    content=text,
                    skill_dir=child,
                )

    def get_skill(self, name: str) -> Skill:
        """Look up a skill by name. Raises KeyError if not found."""
        return self._skills[name]

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def describe_skills(self) -> str:
        """One-line-per-skill summary for injection into prompts."""
        return "\n".join(
            f"- {s.name}: {s.description[:150]}" for s in self._skills.values()
        )

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillsRegistry({len(self._skills)} skills)"
