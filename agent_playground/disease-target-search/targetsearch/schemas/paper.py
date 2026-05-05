"""Data models for paper summaries.

PaperSummary is the single schema that all papers — primary research and
reviews alike — get summarized into by the PaperAgent. This uniformity
means the synthesis agent receives a consistent input regardless of
paper type.

Reviews are also processed as primary sources when they contain original
analysis (meta-analyses, systematic reviews with pooled statistics).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class KeyFinding(BaseModel):
    """A single finding extracted from a paper."""

    finding: str = ""  # What was observed/concluded
    evidence_type: str = ""  # in vivo, in vitro, clinical, computational, meta-analysis, etc.
    model_system: str = ""  # cell line, mouse, human cohort, etc.
    effect_size: str = ""  # e.g. "OR=2.3 (95% CI 1.8-2.9)", "p<0.001", "3-fold increase"
    genes_proteins: list[str] = Field(default_factory=list)  # Genes/proteins involved

    @model_validator(mode="before")
    @classmethod
    def coerce_none_strings(cls, values: dict) -> dict:
        """LLMs return null for missing string fields — coerce to empty string."""
        if isinstance(values, dict):
            for field_name in ("finding", "evidence_type", "model_system", "effect_size"):
                if values.get(field_name) is None:
                    values[field_name] = ""
        return values


class PaperSummary(BaseModel):
    """Structured summary of a single paper, produced by PaperAgent.

    Designed to capture what matters for drug target discovery:
    what was found, how strong the evidence is, and what genes/pathways
    are implicated.
    """

    pmid: str | None = None
    doi: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str = ""

    # What kind of paper is this
    paper_type: str = ""  # primary research, meta-analysis, case study, etc.

    # Core content
    objective: str = ""  # What question the paper set out to answer
    study_design: str = ""  # e.g. "randomized phase II", "CRISPR screen in cell lines"
    key_findings: list[KeyFinding] = Field(default_factory=list)
    methods_summary: str = ""  # Brief description of experimental approach
    limitations: str = ""  # Authors' stated limitations or obvious gaps

    # Evidence quality
    evidence_strength: str = ""  # strong | moderate | weak | insufficient

    # Drug-target-discovery-relevant interpretation
    target_relevance: str = ""  # How findings relate to potential drug targets
    genes_pathways_mentioned: list[str] = Field(default_factory=list)

    # Provenance
    source_type: str = "abstract"  # "full_text" or "abstract" — what the agent had access to

    @model_validator(mode="before")
    @classmethod
    def coerce_none_strings(cls, values: dict) -> dict:
        """LLMs return null for missing string fields — coerce to empty string."""
        if isinstance(values, dict):
            str_fields = (
                "title", "journal", "paper_type", "objective",
                "methods_summary", "limitations", "target_relevance", "source_type",
            )
            for field_name in str_fields:
                if field_name in values and values[field_name] is None:
                    values[field_name] = ""
        return values
