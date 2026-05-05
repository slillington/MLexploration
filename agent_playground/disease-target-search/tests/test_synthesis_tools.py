"""Tests for synthesis_tools — registration, schema, and context integration."""

from targetsearch.core.context import ActionContext
from targetsearch.core.registry import registry
from targetsearch.schemas.disease import (
    DiseaseProfile,
    ExistingTherapy,
    GeneticAssociation,
    GermlineAssociation,
    Pathway,
    SomaticAlteration,
)
from targetsearch.schemas.paper import KeyFinding, PaperSummary
from targetsearch.tools.synthesis_tools import (
    _assess_quality,
    _build_evidence_index,
    _classify_hard_failures,
    _format_compact_result,
    _merge_audits,
    _serialize_summaries,
)


class TestToolRegistration:
    def test_registered(self):
        names = registry.list_names(tags=["synthesis"])
        assert "synthesize_disease_profile" in names

    def test_needs_context(self):
        assert registry.tool_needs_context("synthesize_disease_profile") is True

    def test_schema_excludes_context(self):
        schema = registry.get_tool("synthesize_disease_profile").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "ctx" not in props


class TestSerializeSummaries:
    def test_empty(self):
        ctx = ActionContext()
        result = _serialize_summaries(ctx=ctx)
        assert result == "[]"

    def test_with_summaries(self):
        ctx = ActionContext()
        ctx.paper_state.summaries = [
            PaperSummary(
                pmid="123",
                title="Test Paper",
                year=2024,
                paper_type="primary research",
                objective="Test objective",
                key_findings=[
                    KeyFinding(
                        finding="Found something",
                        evidence_type="in vitro",
                        model_system="HeLa cells",
                        effect_size="p<0.01",
                        genes_proteins=["TP53"],
                    )
                ],
                methods_summary="Western blot",
                limitations="Small sample",
                target_relevance="TP53 is druggable",
                genes_pathways_mentioned=["TP53", "p53 pathway"],
                source_type="full_text",
            )
        ]
        result = _serialize_summaries(ctx=ctx)
        assert '"pmid": "123"' in result
        assert '"finding": "Found something"' in result
        assert '"TP53"' in result
        assert '"source_type": "full_text"' in result

    def test_drops_methods_summary(self):
        """methods_summary is dropped to reduce token cost."""
        ctx = ActionContext()
        ctx.paper_state.summaries = [
            PaperSummary(
                pmid="456",
                title="Full Fields",
                methods_summary="Complex methods",
                limitations="Many limitations",
                target_relevance="Highly relevant",
                genes_pathways_mentioned=["BRCA1", "DNA repair"],
            )
        ]
        result = _serialize_summaries(ctx=ctx)
        assert "Complex methods" not in result
        assert "methods_summary" not in result

    def test_preserves_key_fields(self):
        """limitations, target_relevance, genes_pathways are preserved."""
        ctx = ActionContext()
        ctx.paper_state.summaries = [
            PaperSummary(
                pmid="456",
                title="Full Fields",
                methods_summary="Complex methods",
                limitations="Many limitations",
                target_relevance="Highly relevant",
                genes_pathways_mentioned=["BRCA1", "DNA repair"],
            )
        ]
        result = _serialize_summaries(ctx=ctx)
        assert "Many limitations" in result
        assert "Highly relevant" in result
        assert "BRCA1" in result

    def test_accepts_explicit_list(self):
        """Can pass a list of summaries directly instead of ctx."""
        summaries = [
            PaperSummary(pmid="789", title="Direct List"),
        ]
        result = _serialize_summaries(summaries_list=summaries)
        assert '"pmid": "789"' in result


class TestFormatCompactResult:
    def _make_ctx(self, **kwargs):
        ctx = ActionContext()
        for k, v in kwargs.items():
            setattr(ctx.synthesis_state, k, v)
        return ctx

    def test_includes_counts(self):
        profile = DiseaseProfile(
            disease_name="Test Disease",
            key_pathways=[Pathway(name="PI3K/AKT"), Pathway(name="MAPK")],
            somatic_genomics=[
                SomaticAlteration(gene_symbol="EGFR", alteration_type="mutation"),
                SomaticAlteration(gene_symbol="KRAS", alteration_type="amplification"),
            ],
            germline_genetics=[
                GermlineAssociation(gene_symbol="MUC5B", association_type="GWAS"),
            ],
            existing_therapies=[ExistingTherapy(drug_name="Erlotinib")],
            unmet_needs=["Resistance to EGFR inhibitors"],
            literature_summary="EGFR and KRAS are key drivers.",
        )
        ctx = self._make_ctx(quality_status="pass", synthesis_passes_run=4)
        result = _format_compact_result(profile, ctx)
        assert "Synthesis complete for Test Disease" in result
        assert "Pathways (2)" in result
        assert "PI3K/AKT" in result
        assert "MAPK" in result
        assert "Somatic genomics (2)" in result
        assert "EGFR" in result
        assert "KRAS" in result
        assert "Germline genetics (1)" in result
        assert "MUC5B" in result
        assert "Existing therapies (1)" in result
        assert "Erlotinib" in result
        assert "Unmet needs (1)" in result
        assert "Resistance to EGFR inhibitors" in result
        assert "EGFR and KRAS are key drivers" in result
        assert "Quality: pass" in result
        assert "Internal passes: 4" in result

    def test_empty_profile(self):
        profile = DiseaseProfile(disease_name="Empty")
        ctx = self._make_ctx()
        result = _format_compact_result(profile, ctx)
        assert "Synthesis complete for Empty" in result
        assert "Pathways (0)" in result
        assert "none" in result

    def test_literature_summary_included(self):
        profile = DiseaseProfile(
            disease_name="X",
            literature_summary="A long narrative about the disease.",
        )
        ctx = self._make_ctx()
        result = _format_compact_result(profile, ctx)
        assert "A long narrative about the disease." in result

    def test_diagnostics_included(self):
        profile = DiseaseProfile(disease_name="Diag Test")
        ctx = self._make_ctx(
            quality_status="fail",
            quality_scores={"pathways": 4.0, "somatic_genomics": 7.0},
            contradiction_notes=["EGFR: conflicting expression data"],
            unresolved_claims=["Is TROP2 surface-expressed?"],
            synthesis_passes_run=6,
        )
        result = _format_compact_result(profile, ctx)
        assert "Quality: fail" in result
        assert "pathways: 4.0" in result
        assert "somatic_genomics: 7.0" in result
        assert "Contradictions (1)" in result
        assert "EGFR: conflicting expression data" in result
        assert "Unresolved claims (1)" in result
        assert "Is TROP2 surface-expressed?" in result
        assert "Internal passes: 6" in result


class TestMergeAudits:
    def test_merges_coverage_counts(self):
        audits = [
            {"coverage_by_bucket": {"disease_biology": 5, "clinical": 2}, "contradictions": [], "unresolved_questions": []},
            {"coverage_by_bucket": {"disease_biology": 3, "human_genetics": 4}, "contradictions": [], "unresolved_questions": []},
        ]
        merged = _merge_audits(audits)
        assert merged["coverage_by_bucket"]["disease_biology"] == 8
        assert merged["coverage_by_bucket"]["clinical"] == 2
        assert merged["coverage_by_bucket"]["human_genetics"] == 4

    def test_merges_contradictions(self):
        audits = [
            {"coverage_by_bucket": {}, "contradictions": [{"topic": "A"}], "unresolved_questions": []},
            {"coverage_by_bucket": {}, "contradictions": [{"topic": "B"}], "unresolved_questions": []},
        ]
        merged = _merge_audits(audits)
        assert len(merged["contradictions"]) == 2

    def test_deduplicates_questions(self):
        audits = [
            {"coverage_by_bucket": {}, "contradictions": [], "unresolved_questions": ["Is X druggable?", "Is Y expressed?"]},
            {"coverage_by_bucket": {}, "contradictions": [], "unresolved_questions": ["is x druggable?", "Is Z relevant?"]},
        ]
        merged = _merge_audits(audits)
        assert len(merged["unresolved_questions"]) == 3

    def test_empty_audits(self):
        merged = _merge_audits([])
        assert merged["coverage_by_bucket"] == {}
        assert merged["contradictions"] == []
        assert merged["unresolved_questions"] == []


class TestAssessQuality:
    def test_pass(self):
        critique = {"section_scores": {"a": 8.0, "b": 7.0}, "hard_failures": []}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "pass"
        assert needs is False

    def test_fail_low_scores(self):
        critique = {"section_scores": {"a": 3.0, "b": 4.0}, "hard_failures": []}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "fail"
        assert needs is True

    def test_fail_hard_failures(self):
        critique = {"section_scores": {"a": 9.0, "b": 8.0}, "hard_failures": ["Unsupported claim"]}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "fail"
        assert needs is True

    def test_degraded_no_scores(self):
        critique = {"section_scores": {}, "hard_failures": []}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "degraded"
        assert needs is False

    def test_threshold_boundary(self):
        critique = {"section_scores": {"a": 6.0, "b": 6.0}, "hard_failures": []}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "pass"
        assert needs is False

    def test_string_scores_coerced(self):
        """LLM may return scores as strings — they should be coerced to float."""
        critique = {"section_scores": {"a": "8", "b": "7.5"}, "hard_failures": []}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "pass"
        assert needs is False

    def test_mixed_string_and_float_scores(self):
        critique = {"section_scores": {"a": "3", "b": 4.0}, "hard_failures": []}
        status, needs = _assess_quality(critique, threshold=6.0)
        assert status == "fail"
        assert needs is True

    def test_persistent_failures_tolerated(self):
        """Failures that match previous round are persistent and tolerated."""
        critique = {
            "section_scores": {"a": 8.0, "b": 7.0},
            "hard_failures": ["EGFR is intracellular"],
        }
        previous = ["EGFR is intracellular"]
        status, needs = _assess_quality(
            critique, threshold=6.0,
            previous_failures=previous, max_new_hard_failures=0,
        )
        assert status == "pass"
        assert needs is False

    def test_new_failures_trigger_refinement(self):
        """New failures (not in previous round) trigger refinement."""
        critique = {
            "section_scores": {"a": 8.0, "b": 7.0},
            "hard_failures": ["MET is mislabeled as eQTL"],
        }
        previous = ["EGFR is intracellular"]
        status, needs = _assess_quality(
            critique, threshold=6.0,
            previous_failures=previous, max_new_hard_failures=0,
        )
        assert status == "fail"
        assert needs is True

    def test_new_failures_within_tolerance(self):
        """New failures within max_new_hard_failures are tolerated."""
        critique = {
            "section_scores": {"a": 8.0, "b": 7.0},
            "hard_failures": ["MET is mislabeled"],
        }
        previous = []
        status, needs = _assess_quality(
            critique, threshold=6.0,
            previous_failures=previous, max_new_hard_failures=2,
        )
        assert status == "pass"
        assert needs is False

    def test_mix_of_new_and_persistent(self):
        """Mix of new and persistent: only new count toward threshold."""
        critique = {
            "section_scores": {"a": 8.0, "b": 7.0},
            "hard_failures": [
                "EGFR is intracellular",  # persistent
                "Wrong PMID for MET claim",  # new
            ],
        }
        previous = ["EGFR is intracellular"]
        # 1 new failure, max_new=2 → pass
        status, needs = _assess_quality(
            critique, threshold=6.0,
            previous_failures=previous, max_new_hard_failures=2,
        )
        assert status == "pass"
        assert needs is False


class TestClassifyHardFailures:
    def test_all_new_when_no_previous(self):
        new, persistent = _classify_hard_failures(
            ["failure A", "failure B"], [],
        )
        assert new == ["failure A", "failure B"]
        assert persistent == []

    def test_exact_match_is_persistent(self):
        new, persistent = _classify_hard_failures(
            ["failure A", "failure B"],
            ["failure A"],
        )
        assert new == ["failure B"]
        assert persistent == ["failure A"]

    def test_substring_match_is_persistent(self):
        new, persistent = _classify_hard_failures(
            ["EGFR is intracellular, not a surface target"],
            ["EGFR is intracellular"],
        )
        assert new == []
        assert persistent == ["EGFR is intracellular, not a surface target"]

    def test_reverse_substring_match(self):
        new, persistent = _classify_hard_failures(
            ["EGFR is intracellular"],
            ["EGFR is intracellular, not a surface target"],
        )
        assert new == []
        assert persistent == ["EGFR is intracellular"]

    def test_empty_current(self):
        new, persistent = _classify_hard_failures([], ["old failure"])
        assert new == []
        assert persistent == []


class TestBuildEvidenceIndex:
    def _make_summary(self, pmid="12345678", title="Test Paper", n_findings=3,
                      genes=None, source_type="full_text", paper_type="primary research"):
        genes = genes or ["EGFR", "MET"]
        findings = [
            KeyFinding(finding=f"Finding {i}", genes_proteins=genes[:1])
            for i in range(n_findings)
        ]
        return PaperSummary(
            pmid=pmid, title=title, key_findings=findings,
            genes_pathways_mentioned=genes, source_type=source_type,
            paper_type=paper_type,
        )

    def test_single_paper(self):
        ps = self._make_summary()
        index = _build_evidence_index([ps])
        assert "PMID 12345678" in index
        assert "Test Paper" in index
        assert "3 findings" in index
        assert "EGFR" in index
        assert "MET" in index
        assert "full_text" in index

    def test_multiple_papers(self):
        papers = [
            self._make_summary(pmid="111", title="Paper A"),
            self._make_summary(pmid="222", title="Paper B", source_type="abstract"),
        ]
        index = _build_evidence_index(papers)
        lines = index.strip().split("\n")
        assert len(lines) == 2
        assert "PMID 111" in lines[0]
        assert "PMID 222" in lines[1]
        assert "abstract" in lines[1]

    def test_empty_list(self):
        index = _build_evidence_index([])
        assert index == ""

    def test_long_gene_list_truncated(self):
        genes = [f"GENE{i}" for i in range(12)]
        ps = self._make_summary(genes=genes)
        index = _build_evidence_index([ps])
        assert "+4 more" in index
        # First 8 genes should be present
        for g in genes[:8]:
            assert g in index

    def test_long_title_truncated(self):
        long_title = "A" * 120
        ps = self._make_summary(title=long_title)
        index = _build_evidence_index([ps])
        # Title should be truncated to 80 chars
        assert "A" * 80 in index
        assert "A" * 81 not in index

    def test_much_smaller_than_full_serialization(self):
        papers = [
            self._make_summary(pmid=str(i), title=f"Paper {i}", n_findings=5,
                               genes=["EGFR", "MET", "TROP2", "HER3"])
            for i in range(40)
        ]
        index = _build_evidence_index(papers)
        full = _serialize_summaries(summaries_list=papers)
        # Index should be at least 10x smaller than full serialization
        assert len(index) < len(full) / 10
