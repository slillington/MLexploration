"""Data models for target hypotheses.

These schemas represent the output of the Hypothesis agent — proposed
drug targets paired with therapeutic strategies, backed by evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Target(BaseModel):
    """A protein or gene proposed as a drug target."""

    gene_symbol: str
    uniprot_id: str | None = None
    protein_name: str = ""
    protein_class: str | None = None  # kinase, GPCR, ion channel, etc.


class TherapeuticStrategy(BaseModel):
    """How to drug the target."""

    modality: str  # small molecule, antibody, degrader, ASO, gene therapy, etc.
    mechanism: str  # inhibitor, activator, degrader, agonist, etc.
    rationale: str  # Why this modality for this target


class Citation(BaseModel):
    """A reference to a primary source."""

    pmid: str | None = None
    doi: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    key_finding: str = ""  # One-sentence summary of relevance


class TargetHypothesis(BaseModel):
    """A proposed target–strategy pair with supporting evidence.

    This is the core output unit of the Hypothesis agent.
    """

    target: Target
    strategy: TherapeuticStrategy
    mechanistic_rationale: str  # First-principles reasoning chain
    supporting_evidence: list[Citation] = Field(default_factory=list)
    novelty_assessment: str = ""  # Known target? Novel angle?
    confidence: str = "medium"  # high / medium / low
