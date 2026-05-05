"""Tests for paper_tools — registration, schema, and parsing logic.

LLM-dependent tests (summarize_paper, mine_review_references) are not
included here since they require a live model. These tests verify the
deterministic parts: registration, schema exclusion, and parsing helpers.
"""

from targetsearch.core.registry import registry
from targetsearch.tools.paper_tools import (
    _build_paper_header,
    _parse_paper_summary,
    _parse_review_mining,
)


class TestToolRegistration:
    def test_tools_registered(self):
        names = registry.list_names(tags=["paper"])
        assert "summarize_paper" in names
        assert "mine_review_references" in names

    def test_no_context_needed(self):
        """Paper tools are leaf tools — no ActionContext."""
        assert registry.tool_needs_context("summarize_paper") is False
        assert registry.tool_needs_context("mine_review_references") is False

    def test_schema_has_correct_params(self):
        schema = registry.get_tool("summarize_paper").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "paper_text" in props
        assert "disease_area" in props
        assert "metadata" in props


class TestBuildPaperHeader:
    def test_full_metadata(self):
        header = _build_paper_header({
            "title": "Test Paper",
            "authors": ["Smith J", "Doe A"],
            "journal": "Nature",
            "year": 2024,
            "pmid": "12345",
            "source_type": "full_text",
        })
        assert "Test Paper" in header
        assert "Smith J" in header
        assert "Nature" in header
        assert "2024" in header
        assert "12345" in header
        assert "full_text" in header

    def test_many_authors_truncated(self):
        authors = [f"Author{i}" for i in range(10)]
        header = _build_paper_header({"authors": authors})
        assert "et al." in header

    def test_empty_metadata(self):
        header = _build_paper_header({})
        assert "abstract" in header  # default source_type


class TestParsePaperSummary:
    def test_valid_json(self):
        raw = '{"paper_type": "primary research", "objective": "Test", "key_findings": []}'
        result = _parse_paper_summary(raw, {"pmid": "123", "source_type": "abstract"})
        assert result["pmid"] == "123"
        assert result["paper_type"] == "primary research"
        assert result["source_type"] == "abstract"

    def test_metadata_merge(self):
        raw = '{"paper_type": "clinical trial"}'
        metadata = {
            "pmid": "456",
            "title": "My Paper",
            "authors": ["A", "B"],
            "year": 2023,
            "journal": "Science",
            "source_type": "full_text",
        }
        result = _parse_paper_summary(raw, metadata)
        assert result["pmid"] == "456"
        assert result["title"] == "My Paper"
        assert result["year"] == 2023
        assert result["source_type"] == "full_text"

    def test_invalid_json_returns_metadata(self):
        raw = "not valid json at all"
        result = _parse_paper_summary(raw, {"pmid": "789"})
        assert result["pmid"] == "789"

    def test_new_optional_fields_preserved(self):
        """New prompt fields (study_design, evidence_strength) pass through parsing."""
        raw = '{"paper_type": "primary research", "study_design": "CRISPR screen in A549 cells", "evidence_strength": "moderate", "objective": "test", "key_findings": []}'
        result = _parse_paper_summary(raw, {"pmid": "999"})
        assert result["study_design"] == "CRISPR screen in A549 cells"
        assert result["evidence_strength"] == "moderate"
        assert result["pmid"] == "999"

    def test_missing_optional_fields_no_error(self):
        """Old-format responses without new fields still parse correctly."""
        raw = '{"paper_type": "clinical trial", "objective": "test", "key_findings": []}'
        result = _parse_paper_summary(raw, {"pmid": "888"})
        assert result["paper_type"] == "clinical trial"
        assert result["pmid"] == "888"
        # New fields default to empty string
        assert result["study_design"] == ""
        assert result["evidence_strength"] == ""


class TestParseReviewMining:
    def test_valid_result(self):
        raw = '{"review_title": "A Review", "cited_papers": [{"pmid": "111", "description": "test", "priority": "high"}]}'
        result = _parse_review_mining(raw)
        assert result["review_title"] == "A Review"
        assert len(result["cited_papers"]) == 1
        assert result["cited_papers"][0]["pmid"] == "111"
        assert result["cited_papers"][0]["priority"] == "high"

    def test_invalid_pmid_normalized(self):
        raw = '{"cited_papers": [{"pmid": "not-a-number", "description": "test"}]}'
        result = _parse_review_mining(raw)
        assert result["cited_papers"][0]["pmid"] is None

    def test_null_pmid_preserved(self):
        raw = '{"cited_papers": [{"pmid": null, "description": "test"}]}'
        result = _parse_review_mining(raw)
        assert result["cited_papers"][0]["pmid"] is None

    def test_invalid_json(self):
        result = _parse_review_mining("garbage")
        assert result["cited_papers"] == []
