"""Tests for PaperSummary schema and DiseaseProfile with paper_summaries."""

from targetsearch.schemas.paper import KeyFinding, PaperSummary
from targetsearch.schemas.disease import DiseaseProfile


class TestPaperSummary:
    def test_minimal(self):
        ps = PaperSummary(pmid="12345")
        assert ps.pmid == "12345"
        assert ps.key_findings == []
        assert ps.source_type == "abstract"

    def test_full(self):
        ps = PaperSummary(
            pmid="12345678",
            doi="10.1234/test",
            title="TGF-beta in pulmonary fibrosis",
            authors=["Smith J", "Doe A"],
            year=2023,
            journal="Nature Medicine",
            paper_type="primary research",
            objective="Investigate TGF-beta signaling in IPF fibroblasts.",
            key_findings=[
                KeyFinding(
                    finding="TGF-beta1 activates SMAD3 in IPF fibroblasts",
                    evidence_type="in vitro",
                    model_system="primary human lung fibroblasts",
                    effect_size="3.2-fold increase (p<0.001)",
                    genes_proteins=["TGFB1", "SMAD3"],
                ),
                KeyFinding(
                    finding="SMAD3 inhibition reduced collagen deposition",
                    evidence_type="in vivo",
                    model_system="bleomycin mouse model",
                    effect_size="45% reduction vs control",
                    genes_proteins=["SMAD3", "COL1A1"],
                ),
            ],
            methods_summary="Used primary fibroblasts from IPF patients and bleomycin mouse model.",
            limitations="Mouse model may not fully recapitulate human disease.",
            target_relevance="SMAD3 inhibition could reduce fibrosis progression.",
            genes_pathways_mentioned=["TGFB1", "SMAD3", "COL1A1", "TGF-beta signaling"],
            source_type="full_text",
        )
        assert len(ps.key_findings) == 2
        assert ps.key_findings[0].genes_proteins == ["TGFB1", "SMAD3"]
        assert ps.source_type == "full_text"

    def test_roundtrip_json(self):
        ps = PaperSummary(
            pmid="99999",
            title="Test paper",
            key_findings=[
                KeyFinding(finding="Something was found", genes_proteins=["TP53"]),
            ],
        )
        json_str = ps.model_dump_json()
        ps2 = PaperSummary.model_validate_json(json_str)
        assert ps2.pmid == "99999"
        assert ps2.key_findings[0].genes_proteins == ["TP53"]

    def test_key_finding_defaults(self):
        kf = KeyFinding(finding="A finding")
        assert kf.evidence_type == ""
        assert kf.model_system == ""
        assert kf.effect_size == ""
        assert kf.genes_proteins == []

    def test_null_coercion_key_finding(self):
        """LLMs return null for missing fields — should coerce to empty string."""
        kf = KeyFinding.model_validate({
            "finding": "A finding",
            "effect_size": None,
            "evidence_type": None,
            "model_system": None,
        })
        assert kf.effect_size == ""
        assert kf.evidence_type == ""
        assert kf.model_system == ""

    def test_null_coercion_paper_summary(self):
        """LLMs return null for missing fields — should coerce to empty string."""
        ps = PaperSummary.model_validate({
            "pmid": "12345",
            "title": None,
            "paper_type": None,
            "objective": None,
            "methods_summary": None,
            "limitations": None,
            "target_relevance": None,
        })
        assert ps.title == ""
        assert ps.paper_type == ""
        assert ps.objective == ""
        assert ps.methods_summary == ""
        assert ps.limitations == ""
        assert ps.target_relevance == ""
        # source_type should keep its default when not provided
        assert ps.source_type == "abstract"

    def test_null_coercion_preserves_values(self):
        """Non-null values should pass through unchanged."""
        kf = KeyFinding.model_validate({
            "finding": "Real finding",
            "effect_size": "p<0.01",
            "evidence_type": "in vitro",
        })
        assert kf.finding == "Real finding"
        assert kf.effect_size == "p<0.01"
        assert kf.evidence_type == "in vitro"


class TestDiseaseProfileWithPapers:
    def test_profile_with_paper_summaries(self):
        ps = PaperSummary(pmid="11111", title="Paper 1")
        profile = DiseaseProfile(
            disease_name="test disease",
            paper_summaries=[ps],
        )
        assert len(profile.paper_summaries) == 1
        assert profile.paper_summaries[0].pmid == "11111"

    def test_profile_default_empty_summaries(self):
        profile = DiseaseProfile(disease_name="test")
        assert profile.paper_summaries == []

    def test_profile_roundtrip_with_papers(self):
        ps = PaperSummary(
            pmid="22222",
            title="Test",
            key_findings=[KeyFinding(finding="x")],
        )
        profile = DiseaseProfile(
            disease_name="test",
            paper_summaries=[ps],
        )
        json_str = profile.model_dump_json()
        profile2 = DiseaseProfile.model_validate_json(json_str)
        assert len(profile2.paper_summaries) == 1
        assert profile2.paper_summaries[0].key_findings[0].finding == "x"
