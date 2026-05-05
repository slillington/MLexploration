"""Data models for disease intelligence.

These schemas represent the output of the Disease Intel agent — a structured
profile of a disease including its biology, genetics, and treatment landscape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from targetsearch.schemas.paper import PaperSummary


class Pathway(BaseModel):
    """A biological pathway implicated in the disease."""

    name: str
    description: str = ""
    key_genes: list[str] = Field(default_factory=list)
    evidence_summary: str = ""


class GeneticAssociation(BaseModel):
    """A gene linked to the disease through genetic evidence.

    Kept for backward compatibility — new code should use
    SomaticAlteration and GermlineAssociation instead.
    """

    gene_symbol: str
    association_type: str = ""  # GWAS, Mendelian, somatic, eQTL, etc.
    evidence_summary: str = ""
    source: str = ""  # e.g. "GWAS Catalog", "ClinVar", "Open Targets"


class SomaticAlteration(BaseModel):
    """A somatic genomic alteration observed in the disease."""

    gene_symbol: str
    alteration_type: str = ""  # mutation, amplification, fusion, loss, overexpression
    frequency: str = ""  # e.g. "~30% of NSCLC", optional
    evidence_summary: str = ""
    source: str = ""


class GermlineAssociation(BaseModel):
    """A germline genetic association with the disease."""

    gene_symbol: str
    association_type: str = ""  # GWAS, Mendelian, eQTL
    evidence_summary: str = ""
    source: str = ""  # e.g. "GWAS Catalog", "ClinVar", "Open Targets"


class ExistingTherapy(BaseModel):
    """A current or past therapeutic approach for the disease."""

    drug_name: str
    target: str = ""  # Gene/protein target
    mechanism: str = ""  # e.g. "tyrosine kinase inhibitor"
    status: str = ""  # approved, phase III, discontinued, etc.
    limitations: str = ""


class DiseaseProfile(BaseModel):
    """Structured output of the Disease Intel agent.

    Captures everything downstream agents need to generate and evaluate
    target hypotheses: pathways, genetics, current treatments, and gaps.
    """

    disease_name: str
    synonyms: list[str] = Field(default_factory=list)
    description: str = ""

    # Biology
    key_pathways: list[Pathway] = Field(default_factory=list)
    somatic_genomics: list[SomaticAlteration] = Field(default_factory=list)
    germline_genetics: list[GermlineAssociation] = Field(default_factory=list)
    germline_note: str = ""  # e.g. "No germline evidence found for this disease"

    # Deprecated — kept for backward compatibility with older profiles
    genetic_associations: list[GeneticAssociation] = Field(default_factory=list)

    # Treatment landscape
    existing_therapies: list[ExistingTherapy] = Field(default_factory=list)
    unmet_needs: list[str] = Field(default_factory=list)

    # Structured evidence from the paper pipeline
    paper_summaries: list[PaperSummary] = Field(default_factory=list)

    # Synthesized narrative for downstream agents
    literature_summary: str = ""
