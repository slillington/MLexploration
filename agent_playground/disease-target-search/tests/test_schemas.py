"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from targetsearch.schemas.disease import (
    DiseaseProfile,
    ExistingTherapy,
    GeneticAssociation,
    GermlineAssociation,
    Pathway,
    SomaticAlteration,
)
from targetsearch.schemas.target import (
    Citation,
    Target,
    TargetHypothesis,
    TherapeuticStrategy,
)
from targetsearch.schemas.evaluation import EvaluationReport, FeasibilityScore


class TestDiseaseProfile:
    def test_minimal(self):
        p = DiseaseProfile(disease_name="test disease")
        assert p.disease_name == "test disease"
        assert p.key_pathways == []
        assert p.somatic_genomics == []
        assert p.germline_genetics == []
        assert p.genetic_associations == []

    def test_full(self):
        p = DiseaseProfile(
            disease_name="Idiopathic Pulmonary Fibrosis",
            synonyms=["IPF", "cryptogenic fibrosing alveolitis"],
            description="Progressive scarring of the lungs.",
            key_pathways=[
                Pathway(
                    name="TGF-beta signaling",
                    key_genes=["TGFB1", "SMAD3"],
                    evidence_summary="Central driver of fibroblast activation.",
                )
            ],
            somatic_genomics=[
                SomaticAlteration(
                    gene_symbol="TERT",
                    alteration_type="mutation",
                    evidence_summary="Telomerase mutations in familial IPF.",
                )
            ],
            germline_genetics=[
                GermlineAssociation(
                    gene_symbol="MUC5B",
                    association_type="GWAS",
                    evidence_summary="rs35705950 promoter variant.",
                    source="GWAS Catalog",
                )
            ],
            existing_therapies=[
                ExistingTherapy(
                    drug_name="Nintedanib",
                    target="FGFR/VEGFR/PDGFR",
                    mechanism="Tyrosine kinase inhibitor",
                    status="approved",
                    limitations="Slows but does not halt progression.",
                )
            ],
            unmet_needs=["No therapy reverses fibrosis"],
            literature_summary="IPF is a progressive disease...",
        )
        assert len(p.key_pathways) == 1
        assert p.key_pathways[0].key_genes == ["TGFB1", "SMAD3"]

    def test_roundtrip_json(self):
        p = DiseaseProfile(
            disease_name="test",
            synonyms=["t"],
            key_pathways=[Pathway(name="p1", key_genes=["G1"])],
        )
        json_str = p.model_dump_json()
        p2 = DiseaseProfile.model_validate_json(json_str)
        assert p2.disease_name == "test"
        assert p2.key_pathways[0].name == "p1"


class TestTargetHypothesis:
    def test_minimal(self):
        h = TargetHypothesis(
            target=Target(gene_symbol="EGFR"),
            strategy=TherapeuticStrategy(
                modality="small molecule",
                mechanism="inhibitor",
                rationale="EGFR is overexpressed.",
            ),
            mechanistic_rationale="EGFR drives proliferation.",
        )
        assert h.target.gene_symbol == "EGFR"
        assert h.confidence == "medium"  # default

    def test_with_citations(self):
        h = TargetHypothesis(
            target=Target(gene_symbol="BRAF", protein_name="B-Raf"),
            strategy=TherapeuticStrategy(
                modality="small molecule",
                mechanism="inhibitor",
                rationale="V600E mutation.",
            ),
            mechanistic_rationale="BRAF V600E constitutively activates MAPK.",
            supporting_evidence=[
                Citation(
                    pmid="12068308",
                    title="Mutations of the BRAF gene...",
                    year=2002,
                    key_finding="BRAF mutations in 66% of melanomas.",
                )
            ],
            confidence="high",
        )
        assert len(h.supporting_evidence) == 1
        assert h.supporting_evidence[0].pmid == "12068308"


class TestFeasibilityScore:
    def test_valid_scores(self):
        s = FeasibilityScore(
            genetic_evidence=0.8,
            druggability=0.7,
            competitive_landscape=0.3,
            safety=0.6,
            overall=0.6,
        )
        assert s.overall == 0.6

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            FeasibilityScore(genetic_evidence=1.5)

    def test_negative_score(self):
        with pytest.raises(ValidationError):
            FeasibilityScore(safety=-0.1)


class TestEvaluationReport:
    def test_construction(self):
        r = EvaluationReport(
            hypothesis=TargetHypothesis(
                target=Target(gene_symbol="TP53"),
                strategy=TherapeuticStrategy(
                    modality="gene therapy",
                    mechanism="restoration",
                    rationale="Restore tumor suppressor function.",
                ),
                mechanistic_rationale="TP53 loss removes cell cycle checkpoint.",
            ),
            scores=FeasibilityScore(
                genetic_evidence=0.9,
                druggability=0.2,
                safety=0.5,
                overall=0.4,
            ),
            key_risks=["Undruggable protein class", "Broad expression"],
            next_steps=["Screen for reactivating compounds"],
        )
        assert r.hypothesis.target.gene_symbol == "TP53"
        assert len(r.key_risks) == 2
