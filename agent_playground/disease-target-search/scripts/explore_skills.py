"""Explore the skills registry — list available prompt-based skills."""

from targetsearch.core.skills import SkillsRegistry
from targetsearch.core.config import config

reg = SkillsRegistry(config.skills_dir)
print(f"{len(reg)} skills loaded:\n")
print(reg.describe_skills())

# Show the first 500 chars of each skill
for skill in reg.list_skills():
    print(f"\n{'='*60}")
    print(f"Skill: {skill.name}")
    print(f"Dir:   {skill.skill_dir}")
    print(f"{'='*60}")
    print(skill.content[:500])
    if len(skill.content) > 500:
        print(f"  ... ({len(skill.content) - 500} more chars)")
