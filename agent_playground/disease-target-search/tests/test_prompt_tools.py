"""Tests for prompt construction tools."""

from targetsearch.tools.prompt_tools import (
    create_expert_persona,
    create_extraction_guidelines,
    create_output_schema,
)


class TestCreateExpertPersona:
    def test_basic(self):
        result = create_expert_persona("ALS", ["genetics", "neuroscience"])
        assert "ALS" in result
        assert "genetics" in result
        assert "neuroscience" in result
        assert "critical evidence reviewer" in result.lower()

    def test_single_expertise(self):
        result = create_expert_persona("Crohn's disease", ["immunology"])
        assert "Crohn's disease" in result
        assert "immunology" in result

    def test_extraction_principles(self):
        result = create_expert_persona("NSCLC", ["oncology"])
        assert "observed data" in result.lower() or "interpretation" in result.lower()
        assert "HGNC" in result
        assert "preclinical" in result.lower() or "clinical" in result.lower()


class TestCreateOutputSchema:
    def test_paper_summary(self):
        result = create_output_schema("paper_summary")
        assert "paper_type" in result
        assert "key_findings" in result
        assert "genes_pathways_mentioned" in result
        # New fields from prompt upgrade
        assert "study_design" in result
        assert "evidence_strength" in result

    def test_review_mining(self):
        result = create_output_schema("review_mining")
        assert "cited_papers" in result
        assert "priority" in result

    def test_disease_profile(self):
        result = create_output_schema("disease_profile")
        assert "key_pathways" in result
        assert "somatic_genomics" in result
        assert "germline_genetics" in result
        assert "germline_note" in result
        assert "unmet_needs" in result

    def test_evidence_audit(self):
        result = create_output_schema("evidence_audit")
        assert "coverage_by_bucket" in result
        assert "contradictions" in result
        assert "unresolved_questions" in result

    def test_quality_critique(self):
        result = create_output_schema("quality_critique")
        assert "section_scores" in result
        assert "hard_failures" in result
        assert "weak_sections" in result
        assert "revision_instructions" in result

    def test_unknown_schema(self):
        result = create_output_schema("nonexistent")
        assert "ERROR" in result
        assert "paper_summary" in result  # lists available schemas


class TestCreateExtractionGuidelines:
    def test_full_text(self):
        result = create_extraction_guidelines("full_text")
        assert "full text" in result.lower()
        assert "effect sizes" in result.lower()
        # Domain-aware extraction
        assert "preclinical" in result.lower()
        assert "clinical" in result.lower()

    def test_abstract(self):
        result = create_extraction_guidelines("abstract")
        assert "abstract" in result.lower()
        assert "evidence_strength" in result

    def test_review(self):
        result = create_extraction_guidelines("review")
        assert "review" in result.lower()
        assert "prioritization" in result.lower()
        assert "causal chain" in result.lower()
        assert "genetic support" in result.lower()
        assert "druggability" in result.lower()
        assert "citation handling" in result.lower()

    def test_unknown_type(self):
        result = create_extraction_guidelines("nonexistent")
        assert "ERROR" in result


class TestToolRegistration:
    def test_tools_registered(self):
        from targetsearch.core.registry import registry

        names = registry.list_names(tags=["prompts"])
        assert "create_expert_persona" in names
        assert "create_output_schema" in names
        assert "create_extraction_guidelines" in names

    def test_schema_excludes_no_context_params(self):
        """Prompt tools are leaf tools — no ActionContext in schema."""
        from targetsearch.core.registry import registry

        for name in ["create_expert_persona", "create_output_schema", "create_extraction_guidelines"]:
            assert registry.tool_needs_context(name) is False
