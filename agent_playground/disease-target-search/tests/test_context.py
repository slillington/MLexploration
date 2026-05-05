"""Tests for ActionContext construction, mutation, and summarize()."""

from targetsearch.core.context import (
    ActionContext,
    DiseaseInfo,
    Metadata,
    PaperState,
    SearchState,
    SynthesisState,
    TargetState,
)
from targetsearch.schemas.disease import DiseaseProfile
from targetsearch.schemas.paper import PaperSummary


def test_default_construction():
    ctx = ActionContext()
    assert ctx.context_id  # UUID is auto-generated
    assert ctx.disease_info.name == ""
    assert ctx.search_state.queries_executed == []
    assert ctx.paper_state.summaries == []
    assert ctx.target_state.opentargets_results == []
    assert ctx.synthesis_state.has_been_run is False
    assert ctx.metadata.tool_call_count == 0
    assert ctx.properties == {}


def test_mutation():
    ctx = ActionContext()
    ctx.disease_info.name = "Crohn's disease"
    ctx.disease_info.synonyms = ["CD", "regional enteritis"]
    ctx.search_state.queries_executed.append("Crohn's disease drug targets")
    ctx.search_state.total_papers_found = 42
    ctx.search_state.pmids_collected.extend(["123", "456"])

    assert ctx.disease_info.name == "Crohn's disease"
    assert len(ctx.search_state.pmids_collected) == 2
    assert ctx.search_state.total_papers_found == 42


def test_paper_state_mutation():
    ctx = ActionContext()
    summary = PaperSummary(
        pmid="12345",
        title="Test Paper",
        paper_type="primary research",
        source_type="full_text",
    )
    ctx.paper_state.summaries.append(summary)
    ctx.paper_state.papers_fetched = 1
    ctx.paper_state.papers_with_full_text = 1

    assert len(ctx.paper_state.summaries) == 1
    assert ctx.paper_state.summaries[0].pmid == "12345"


def test_synthesis_state_mutation():
    ctx = ActionContext()
    profile = DiseaseProfile(disease_name="Test Disease")
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.profile = profile
    ctx.synthesis_state.gaps = ["Missing genetic evidence", "No clinical data"]

    assert ctx.synthesis_state.has_been_run is True
    assert ctx.synthesis_state.profile.disease_name == "Test Disease"
    assert len(ctx.synthesis_state.gaps) == 2


def test_synthesized_pmids():
    ctx = ActionContext()
    assert ctx.synthesis_state.synthesized_pmids == set()
    ctx.synthesis_state.synthesized_pmids = {"12345", "67890"}
    assert "12345" in ctx.synthesis_state.synthesized_pmids
    assert len(ctx.synthesis_state.synthesized_pmids) == 2


def test_properties_escape_hatch():
    ctx = ActionContext()
    ctx.properties["custom_score"] = 0.85
    ctx.properties["extra_pmids"] = ["789", "101"]

    assert ctx.properties["custom_score"] == 0.85
    assert len(ctx.properties["extra_pmids"]) == 2


def test_summarize_empty():
    ctx = ActionContext()
    summary = ctx.summarize()
    assert "not yet identified" in summary
    assert "0 queries" in summary
    assert "0 summarized" in summary
    assert "not yet run" in summary


def test_summarize_populated():
    ctx = ActionContext()
    ctx.disease_info.name = "ALS"
    ctx.disease_info.synonyms = ["amyotrophic lateral sclerosis", "Lou Gehrig's disease"]
    ctx.search_state.queries_executed = ["ALS drug targets", "ALS genetics"]
    ctx.search_state.total_papers_found = 50
    ctx.search_state.pmids_collected = [str(i) for i in range(20)]
    ctx.paper_state.papers_fetched = 15
    ctx.paper_state.papers_with_full_text = 10
    ctx.paper_state.papers_abstract_only = 5
    ctx.paper_state.summaries = [
        PaperSummary(pmid=str(i), title=f"Paper {i}") for i in range(15)
    ]
    ctx.target_state.opentargets_results = [{"id": "t1"}, {"id": "t2"}]
    ctx.target_state.known_drugs = [{"name": "riluzole"}]
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.gaps = ["Missing SOD1 data", "No biomarker evidence"]
    ctx.synthesis_state.feedback_rounds = 1
    ctx.metadata.tool_call_count = 25
    ctx.metadata.iteration_count = 3

    summary = ctx.summarize()
    assert "ALS" in summary
    assert "2 queries" in summary
    assert "50 papers found" in summary
    assert "15 summarized" in summary
    assert "2 targets loaded" in summary
    assert "completed" in summary
    assert "2" in summary  # gaps
    assert "25 tool calls" in summary


def test_context_id_unique():
    ctx1 = ActionContext()
    ctx2 = ActionContext()
    assert ctx1.context_id != ctx2.context_id


def test_metadata_created_at():
    ctx = ActionContext()
    assert ctx.metadata.created_at  # ISO timestamp is set
    assert "T" in ctx.metadata.created_at  # ISO format


def test_overall_assessment_default():
    ctx = ActionContext()
    assert ctx.synthesis_state.overall_assessment == ""


def test_overall_assessment_in_summary():
    ctx = ActionContext()
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.overall_assessment = "needs more evidence"
    summary = ctx.summarize()
    assert "Feedback assessment: needs more evidence" in summary


def test_overall_assessment_not_shown_when_empty():
    ctx = ActionContext()
    ctx.synthesis_state.has_been_run = True
    summary = ctx.summarize()
    assert "Feedback assessment" not in summary


def test_queries_in_summary():
    ctx = ActionContext()
    ctx.search_state.queries_executed = [
        "pubmed_search: NSCLC EGFR",
        "semantic_scholar_search: lung cancer targets",
    ]
    summary = ctx.summarize()
    assert "Queries executed:" in summary
    assert "pubmed_search: NSCLC EGFR" in summary
    assert "semantic_scholar_search: lung cancer targets" in summary


def test_queries_not_shown_when_empty():
    ctx = ActionContext()
    summary = ctx.summarize()
    assert "Queries executed:" not in summary


def test_quality_scores_in_summary():
    ctx = ActionContext()
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.quality_status = "fail"
    ctx.synthesis_state.quality_scores = {"pathways": 4.0, "unmet_needs": 8.0}
    summary = ctx.summarize()
    assert "Quality: fail" in summary
    assert "pathways: 4.0" in summary
    assert "unmet_needs: 8.0" in summary


def test_contradictions_in_summary():
    ctx = ActionContext()
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.contradiction_notes = ["EGFR conflict", "KRAS conflict"]
    summary = ctx.summarize()
    assert "Contradictions: 2" in summary


def test_unresolved_claims_in_summary():
    ctx = ActionContext()
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.unresolved_claims = ["Is X druggable?"]
    summary = ctx.summarize()
    assert "Unresolved claims: 1" in summary


def test_synthesis_stage_in_summary():
    ctx = ActionContext()
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.synthesis_stage = "critique"
    ctx.synthesis_state.synthesis_passes_run = 3
    summary = ctx.summarize()
    assert "Stage: critique" in summary
    assert "Internal passes: 3" in summary
