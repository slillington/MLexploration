"""Data models for target evaluation and scoring.

These schemas represent the output of the Evaluation agent — feasibility
scores and risk assessments for each target hypothesis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from targetsearch.schemas.target import TargetHypothesis


class FeasibilityScore(BaseModel):
    """Multi-axis feasibility assessment for a target hypothesis.

    Each axis is scored 0.0–1.0 where higher is better.
    """

    genetic_evidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Strength of human genetic evidence linking target to disease",
    )
    druggability: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Likelihood the target can be modulated by the proposed modality",
    )
    competitive_landscape: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Higher = less competition = more opportunity",
    )
    safety: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Higher = fewer anticipated safety liabilities",
    )
    overall: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Weighted composite score",
    )


class EvaluationReport(BaseModel):
    """Full evaluation of a single target hypothesis."""

    hypothesis: TargetHypothesis
    scores: FeasibilityScore
    detailed_assessment: str = ""
    key_risks: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)  # Suggested validation experiments
