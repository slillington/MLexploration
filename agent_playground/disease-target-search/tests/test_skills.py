"""Tests for the skills registry."""

from pathlib import Path

from targetsearch.core.skills import SkillsRegistry


class TestSkillsRegistry:
    def test_loads_existing_skills(self):
        skills_dir = Path(__file__).resolve().parents[1] / "skills"
        reg = SkillsRegistry(skills_dir)
        assert len(reg) > 0
        names = reg.list_names()
        assert "biopython" in names
        assert "paper-lookup" in names

    def test_skill_has_content(self):
        skills_dir = Path(__file__).resolve().parents[1] / "skills"
        reg = SkillsRegistry(skills_dir)
        skill = reg.get_skill("biopython")
        assert skill.name == "biopython"
        assert len(skill.content) > 100
        assert len(skill.description) > 10

    def test_describe_skills(self):
        skills_dir = Path(__file__).resolve().parents[1] / "skills"
        reg = SkillsRegistry(skills_dir)
        desc = reg.describe_skills()
        assert "biopython" in desc
        assert "esm" in desc

    def test_missing_dir_is_empty(self):
        reg = SkillsRegistry(Path("/nonexistent/path"))
        assert len(reg) == 0

    def test_none_dir_is_empty(self):
        reg = SkillsRegistry(None)
        assert len(reg) == 0
