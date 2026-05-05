"""Tests for agent_tools — registration, schema, and gap extraction."""

from targetsearch.core.registry import registry
from targetsearch.tools.agent_tools import _extract_gaps, _extract_overall_assessment


class TestToolRegistration:
    def test_orchestration_tools_registered(self):
        names = registry.list_names(tags=["orchestration"])
        assert "run_search_agent" in names
        assert "run_feedback_agent" in names

    def test_needs_context(self):
        assert registry.tool_needs_context("run_search_agent") is True
        assert registry.tool_needs_context("run_feedback_agent") is True

    def test_schema_excludes_context(self):
        for name in ["run_search_agent", "run_feedback_agent"]:
            schema = registry.get_tool(name).to_openai_schema()
            props = schema["function"]["parameters"]["properties"]
            assert "ctx" not in props

    def test_search_agent_schema(self):
        schema = registry.get_tool("run_search_agent").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "disease_area" in props
        assert "gaps_to_fill" in props

    def test_feedback_agent_schema_no_params(self):
        schema = registry.get_tool("run_feedback_agent").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        # No LLM-visible params (ctx is excluded)
        assert len(props) == 0


class TestExtractGaps:
    def test_standard_format(self):
        text = """GAPS:
1. No GWAS data for SOD1 → Search: "SOD1 GWAS ALS"
2. Missing druggability assessment for TDP-43 → Search: "TDP-43 druggability"

OVERALL: needs more evidence

REASONING: The profile lacks genetic validation."""
        gaps = _extract_gaps(text)
        assert len(gaps) == 2
        assert "No GWAS data for SOD1" in gaps[0]
        assert "Missing druggability assessment for TDP-43" in gaps[1]

    def test_arrow_variants(self):
        text = """GAPS:
1. Missing data -> Search: "query"
"""
        gaps = _extract_gaps(text)
        assert len(gaps) == 1
        assert "Missing data" in gaps[0]

    def test_no_gaps_section(self):
        text = "The profile looks adequate. No major gaps identified."
        gaps = _extract_gaps(text)
        assert gaps == []

    def test_empty_gaps(self):
        text = """GAPS:

OVERALL: adequate"""
        gaps = _extract_gaps(text)
        assert gaps == []


class TestExtractOverallAssessment:
    def test_adequate(self):
        text = "GAPS:\n\nOVERALL: adequate\n\nREASONING: Good coverage."
        assert _extract_overall_assessment(text) == "adequate"

    def test_needs_more_evidence(self):
        text = "GAPS:\n1. Missing data\n\nOVERALL: needs more evidence\n\nREASONING: Thin."
        assert _extract_overall_assessment(text) == "needs more evidence"

    def test_case_insensitive_header(self):
        text = "Overall: adequate"
        assert _extract_overall_assessment(text) == "adequate"

    def test_missing(self):
        text = "No overall section here."
        assert _extract_overall_assessment(text) == ""

    def test_whitespace(self):
        text = "OVERALL:   needs more evidence  "
        assert _extract_overall_assessment(text) == "needs more evidence"


class TestOrchestratorSetup:
    def test_orchestrator_tool_visibility(self):
        from targetsearch.agents.disease_intel import DiseaseIntelAgent

        agent = DiseaseIntelAgent()
        tool_names = {t["function"]["name"] for t in agent.tools}
        assert tool_names == {
            "run_search_agent",
            "run_feedback_agent",
            "synthesize_disease_profile",
        }

    def test_orchestrator_does_not_see_leaf_tools(self):
        from targetsearch.agents.disease_intel import DiseaseIntelAgent

        agent = DiseaseIntelAgent()
        tool_names = {t["function"]["name"] for t in agent.tools}
        # Should NOT see search tools, paper tools, or coordination tools
        assert "pubmed_search" not in tool_names
        assert "summarize_paper" not in tool_names
        assert "fetch_and_classify_papers" not in tool_names
        assert "batch_summarize_papers" not in tool_names
