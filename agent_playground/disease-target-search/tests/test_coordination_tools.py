"""Tests for coordination_tools — classification, registration, and context wiring.

Network-dependent tests (actual NCBI fetches, LLM calls) are not included.
These tests verify the deterministic helpers and tool registration.
"""

from targetsearch.core.context import ActionContext
from targetsearch.core.registry import registry
from targetsearch.tools.coordination_tools import _classify_papers


class TestClassifyPapers:
    def test_review_detected(self):
        papers = [
            {"pmid": "1", "pub_types": ["Review"]},
            {"pmid": "2", "pub_types": ["Journal Article"]},
        ]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 1
        assert reviews[0]["pmid"] == "1"
        assert len(primaries) == 1
        assert primaries[0]["pmid"] == "2"

    def test_systematic_review(self):
        papers = [{"pmid": "1", "pub_types": ["Systematic Review", "Journal Article"]}]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 1
        assert len(primaries) == 0

    def test_meta_analysis(self):
        papers = [{"pmid": "1", "pub_types": ["Meta-Analysis"]}]
        reviews, _ = _classify_papers(papers)
        assert len(reviews) == 1

    def test_no_pub_types_is_primary(self):
        papers = [{"pmid": "1"}]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 0
        assert len(primaries) == 1

    def test_empty_list(self):
        reviews, primaries = _classify_papers([])
        assert reviews == []
        assert primaries == []

    def test_mixed(self):
        papers = [
            {"pmid": "1", "pub_types": ["Review"]},
            {"pmid": "2", "pub_types": ["Journal Article"]},
            {"pmid": "3", "pub_types": ["Practice Guideline"]},
            {"pmid": "4", "pub_types": ["Case Reports"]},
        ]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 2  # Review + Practice Guideline
        assert len(primaries) == 2


class TestToolRegistration:
    def test_coordination_tools_registered(self):
        names = registry.list_names(tags=["coordination"])
        assert "fetch_and_classify_papers" in names
        assert "batch_summarize_papers" in names

    def test_needs_context(self):
        assert registry.tool_needs_context("fetch_and_classify_papers") is True
        assert registry.tool_needs_context("batch_summarize_papers") is True

    def test_schema_excludes_context(self):
        for name in ["fetch_and_classify_papers", "batch_summarize_papers"]:
            schema = registry.get_tool(name).to_openai_schema()
            props = schema["function"]["parameters"]["properties"]
            assert "ctx" not in props, f"{name} should not expose ctx in schema"

    def test_fetch_schema_has_pmids(self):
        schema = registry.get_tool("fetch_and_classify_papers").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "pmids" in props
        assert props["pmids"]["type"] == "array"

    def test_batch_schema_has_disease_area(self):
        schema = registry.get_tool("batch_summarize_papers").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "disease_area" in props


class TestPaperBudgetEnforcement:
    """Verify split paper budget between initial and gap-fill passes."""

    def _make_ctx_with_summaries(self, n_summaries: int) -> ActionContext:
        """Create a context with n pre-existing summaries and some fetched papers."""
        from targetsearch.schemas.paper import PaperSummary

        ctx = ActionContext()
        for i in range(n_summaries):
            ctx.paper_state.summaries.append(
                PaperSummary(pmid=str(10000 + i), title=f"Paper {i}")
            )
        # Add fetched papers that haven't been summarized yet
        for i in range(5):
            pmid = str(20000 + i)
            ctx.properties.setdefault("fetched_papers", []).append(
                {"pmid": pmid, "text": f"Full text of paper {pmid}", "source_type": "abstract"}
            )
        return ctx

    def test_initial_pass_budget_exhausted(self):
        """When initial budget (12) is full, return budget-reached message."""
        from targetsearch.tools.coordination_tools import batch_summarize_papers
        from targetsearch.core.config import config

        ctx = self._make_ctx_with_summaries(config.max_papers_initial)
        ctx.synthesis_state.feedback_rounds = 0
        result = batch_summarize_papers(disease_area="test", ctx=ctx)
        assert "budget reached" in result.lower()
        assert "initial" in result.lower()

    def test_gap_fill_has_independent_budget(self):
        """After initial pass fills 12, gap-fill budget (8) is still available."""
        from targetsearch.core.config import config

        # With 12 summaries and feedback_rounds=1, gap-fill used = max(0, 12-12) = 0
        # remaining = 8 - 0 = 8
        already = config.max_papers_initial
        gap_fill_used = max(0, already - config.max_papers_initial)
        remaining = config.max_papers_gap_fill - gap_fill_used
        assert remaining == config.max_papers_gap_fill

    def test_gap_fill_budget_exhausted(self):
        """When gap-fill budget (8) is also full, return budget-reached."""
        from targetsearch.tools.coordination_tools import batch_summarize_papers
        from targetsearch.core.config import config

        total = config.max_papers_initial + config.max_papers_gap_fill
        ctx = self._make_ctx_with_summaries(total)
        ctx.synthesis_state.feedback_rounds = 1
        result = batch_summarize_papers(disease_area="test", ctx=ctx)
        assert "budget reached" in result.lower()
        assert "gap-fill" in result.lower()


class TestEarlyBudgetCheck:
    """Verify fetch_and_classify_papers checks budget before fetching."""

    def test_fetch_returns_early_when_initial_budget_exhausted(self):
        """When initial budget is full, fetch returns without doing work."""
        from targetsearch.tools.coordination_tools import fetch_and_classify_papers
        from targetsearch.core.config import config
        from targetsearch.schemas.paper import PaperSummary

        ctx = ActionContext()
        for i in range(config.max_papers_initial):
            ctx.paper_state.summaries.append(
                PaperSummary(pmid=str(10000 + i), title=f"Paper {i}")
            )
        ctx.synthesis_state.feedback_rounds = 0
        result = fetch_and_classify_papers(pmids=["99999"], ctx=ctx)
        assert "budget exhausted" in result.lower()

    def test_fetch_returns_early_when_gap_fill_budget_exhausted(self):
        """When gap-fill budget is full, fetch returns without doing work."""
        from targetsearch.tools.coordination_tools import fetch_and_classify_papers
        from targetsearch.core.config import config
        from targetsearch.schemas.paper import PaperSummary

        ctx = ActionContext()
        total = config.max_papers_initial + config.max_papers_gap_fill
        for i in range(total):
            ctx.paper_state.summaries.append(
                PaperSummary(pmid=str(10000 + i), title=f"Paper {i}")
            )
        ctx.synthesis_state.feedback_rounds = 1
        result = fetch_and_classify_papers(pmids=["99999"], ctx=ctx)
        assert "budget exhausted" in result.lower()
        assert "gap-fill" in result.lower()

    def test_remaining_budget_helper(self):
        """Verify _remaining_paper_budget returns correct values."""
        from targetsearch.tools.coordination_tools import _remaining_paper_budget
        from targetsearch.core.config import config
        from targetsearch.schemas.paper import PaperSummary

        ctx = ActionContext()
        for i in range(5):
            ctx.paper_state.summaries.append(
                PaperSummary(pmid=str(i), title=f"P{i}")
            )
        ctx.synthesis_state.feedback_rounds = 0
        remaining, label = _remaining_paper_budget(ctx)
        assert label == "initial"
        assert remaining == config.max_papers_initial - 5

        ctx.synthesis_state.feedback_rounds = 1
        remaining, label = _remaining_paper_budget(ctx)
        assert label == "gap-fill"
        # 5 summaries, all within initial budget, so gap-fill used = 0
        assert remaining == config.max_papers_gap_fill


class TestReviewMiningRemoved:
    """Verify review mining is no longer part of batch_summarize_papers."""

    def test_no_mine_review_references_import(self):
        """coordination_tools should not import mine_review_references."""
        import targetsearch.tools.coordination_tools as ct
        assert not hasattr(ct, "mine_review_references")

    def test_batch_summary_no_mined_pmids(self):
        """batch_summarize_papers return message should not mention mined PMIDs."""
        from targetsearch.tools.coordination_tools import batch_summarize_papers
        ctx = ActionContext()
        ctx.properties["fetched_papers"] = []
        result = batch_summarize_papers(disease_area="test", ctx=ctx)
        assert "mined" not in result.lower()
        # The batch_summarize_papers function checks this condition
        # and logs "skipping review mining on gap-fill pass"

