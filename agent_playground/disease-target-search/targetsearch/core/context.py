"""ActionContext — structured working memory for the agent pipeline.

ActionContext flows through the orchestrator's tool-calling loop. Coordination
tools read and mutate it in place; the orchestrator sees a concise summary
on each turn via summarize().

Typed sections give IDE autocomplete and type checking. The `properties`
dict is an escape hatch for ad-hoc data that doesn't warrant a new field.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from targetsearch.core.config import config
from typing import Any

from targetsearch.schemas.disease import DiseaseProfile
from targetsearch.schemas.paper import PaperSummary


# ── Typed section dataclasses ──────────────────────────────────────────


@dataclass
class DiseaseInfo:
    """Basic disease identity resolved during early search."""

    name: str = ""
    synonyms: list[str] = field(default_factory=list)
    ontology_ids: dict[str, str] = field(default_factory=dict)  # source → id


@dataclass
class SearchState:
    """Tracks what searches have been executed."""

    queries_executed: list[str] = field(default_factory=list)
    total_papers_found: int = 0
    pmids_collected: list[str] = field(default_factory=list)


@dataclass
class PaperState:
    """Papers fetched and summarized."""

    papers_fetched: int = 0
    papers_with_full_text: int = 0
    papers_abstract_only: int = 0
    summaries: list[PaperSummary] = field(default_factory=list)
    review_pmids_discovered: list[str] = field(default_factory=list)


@dataclass
class TargetState:
    """Open Targets and drug data."""

    opentargets_results: list[dict[str, Any]] = field(default_factory=list)
    known_drugs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SynthesisState:
    """Synthesis output and feedback."""

    has_been_run: bool = False
    profile: DiseaseProfile | None = None
    gaps: list[str] = field(default_factory=list)
    feedback_rounds: int = 0
    overall_assessment: str = ""  # "adequate" or "needs more evidence"

    # Multi-pass synthesis diagnostics
    synthesis_stage: str = ""  # current stage: audit, draft, critique, refine, done
    synthesis_passes_run: int = 0  # total internal passes (audit+draft+critique+refine)
    coverage_by_bucket: dict[str, int] = field(default_factory=dict)
    contradiction_notes: list[str] = field(default_factory=list)
    unresolved_claims: list[str] = field(default_factory=list)
    quality_scores: dict[str, float] = field(default_factory=dict)  # section → 0-10
    quality_status: str = ""  # "pass", "fail", or "degraded"
    synthesized_pmids: set[str] = field(default_factory=set)  # PMIDs included in last synthesis


@dataclass
class Metadata:
    """Timing and counters."""

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tool_call_count: int = 0
    iteration_count: int = 0


# ── ActionContext ──────────────────────────────────────────────────────


@dataclass
class ActionContext:
    """Structured working memory shared across the orchestrator and sub-agents.

    Coordination tools accept this as a parameter (auto-injected by the
    agent loop) and mutate it in place. Leaf tools never see it.
    """

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Typed sections
    disease_info: DiseaseInfo = field(default_factory=DiseaseInfo)
    search_state: SearchState = field(default_factory=SearchState)
    paper_state: PaperState = field(default_factory=PaperState)
    target_state: TargetState = field(default_factory=TargetState)
    synthesis_state: SynthesisState = field(default_factory=SynthesisState)
    metadata: Metadata = field(default_factory=Metadata)

    # Escape hatch for ad-hoc data
    properties: dict[str, Any] = field(default_factory=dict)

    def summarize(self) -> str:
        """Concise status report for orchestrator prompt injection.

        ~200-400 tokens. Tells the LLM what has been done and what's
        missing so it can decide the next action. Does NOT replace the
        full data — tools read typed sections directly.
        """
        lines = [f"=== Context {self.context_id[:8]} ==="]

        # Disease
        if self.disease_info.name:
            syn = ", ".join(self.disease_info.synonyms[:5]) or "none"
            lines.append(
                f"Disease: {self.disease_info.name} (synonyms: {syn})"
            )
        else:
            lines.append("Disease: not yet identified")

        # Search
        ss = self.search_state
        lines.append(
            f"Search: {len(ss.queries_executed)} queries, "
            f"{ss.total_papers_found} papers found, "
            f"{len(ss.pmids_collected)} PMIDs collected"
        )
        if ss.queries_executed:
            lines.append("  Queries executed:")
            for q in ss.queries_executed:
                lines.append(f"    - {q}")

        # Papers (with budget info)
        ps = self.paper_state
        already = len(ps.summaries)
        sy = self.synthesis_state
        if sy.feedback_rounds == 0:
            phase_budget = config.max_papers_initial
            phase_used = already
            phase_label = "initial"
        else:
            phase_budget = config.max_papers_gap_fill
            phase_used = max(0, already - config.max_papers_initial)
            phase_label = "gap-fill"
        budget_remaining = max(0, phase_budget - phase_used)
        lines.append(
            f"Papers: {ps.papers_fetched} fetched "
            f"({ps.papers_with_full_text} full-text, "
            f"{ps.papers_abstract_only} abstract-only), "
            f"{already} summarized, "
            f"{phase_used}/{phase_budget} {phase_label} budget "
            f"({budget_remaining} remaining)"
        )

        # Targets
        ts = self.target_state
        if ts.opentargets_results:
            lines.append(
                f"Open Targets: {len(ts.opentargets_results)} targets loaded"
            )
        if ts.known_drugs:
            lines.append(f"Known drugs: {len(ts.known_drugs)}")

        # Synthesis
        sy = self.synthesis_state
        if sy.has_been_run:
            lines.append("Synthesis: completed")
            if sy.synthesis_stage:
                lines.append(f"  Stage: {sy.synthesis_stage}")
            if sy.synthesis_passes_run:
                lines.append(f"  Internal passes: {sy.synthesis_passes_run}")
            if sy.quality_status:
                lines.append(f"  Quality: {sy.quality_status}")
            if sy.quality_scores:
                scores = ", ".join(
                    f"{k}: {v:.1f}" for k, v in sy.quality_scores.items()
                )
                lines.append(f"  Section scores: {scores}")
            if sy.contradiction_notes:
                lines.append(f"  Contradictions: {len(sy.contradiction_notes)}")
            if sy.unresolved_claims:
                lines.append(f"  Unresolved claims: {len(sy.unresolved_claims)}")
            if sy.overall_assessment:
                lines.append(f"  Feedback assessment: {sy.overall_assessment}")
            if sy.gaps:
                lines.append(f"  Gaps identified: {len(sy.gaps)}")
                for g in sy.gaps[:5]:
                    lines.append(f"    - {g}")
            lines.append(f"  Feedback rounds: {sy.feedback_rounds}")
        else:
            lines.append("Synthesis: not yet run")

        # Metadata
        lines.append(
            f"Stats: {self.metadata.tool_call_count} tool calls, "
            f"{self.metadata.iteration_count} iterations"
        )

        return "\n".join(lines)
