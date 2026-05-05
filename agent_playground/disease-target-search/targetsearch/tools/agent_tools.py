"""Agent tools — expose sub-agents as registered tools for the orchestrator.

These are coordination tools (accept ActionContext). The orchestrator calls
them as single tool invocations; internally they instantiate a sub-agent
with its own tool-calling loop and shared ActionContext.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from targetsearch.core.context import ActionContext
from targetsearch.core.registry import registry

log = logging.getLogger(__name__)


@registry.tool(
    description=(
        "Run the search sub-agent to find papers and target data for a disease. "
        "The agent formulates search queries, fetches papers, classifies them, "
        "and produces summaries. If gaps_to_fill is provided, the agent focuses "
        "on finding evidence to address those specific gaps."
    ),
    tags=["orchestration"],
    params={
        "disease_area": "Disease to search for (e.g. 'idiopathic pulmonary fibrosis')",
        "gaps_to_fill": "Specific evidence gaps to address (optional)",
    },
    returns="Summary of what the search agent found",
)
def run_search_agent(
    disease_area: str,
    ctx: ActionContext,
    gaps_to_fill: list[str] | None = None,
) -> str:
    """Instantiate and run a SearcherAgent with the shared context."""
    # Import here to avoid circular imports at module level
    from targetsearch.agents.searcher import SearcherAgent

    agent = SearcherAgent()

    # Build the user message
    if gaps_to_fill:
        gaps_str = "\n".join(f"- {g}" for g in gaps_to_fill)
        user_message = (
            f"Search for drug target evidence for: {disease_area}\n\n"
            f"Focus on filling these specific gaps:\n{gaps_str}"
        )
    else:
        user_message = (
            f"Find papers and target data for: {disease_area}\n\n"
            f"Cast a wide net — search for reviews, genetic studies, "
            f"mechanism papers, and treatment landscape."
        )

    # Inject prior queries so the searcher avoids repeating them
    if ctx.search_state.queries_executed:
        prior = "\n".join(
            f"  {i + 1}. {q}"
            for i, q in enumerate(ctx.search_state.queries_executed)
        )
        user_message += (
            f"\n\n## Previously executed queries (do not repeat these)\n{prior}"
        )

    # Update disease info in context
    if not ctx.disease_info.name:
        ctx.disease_info.name = disease_area

    log.info("run_search_agent: starting for %s", disease_area)
    result = agent.run(user_message, context=ctx)
    log.info("run_search_agent: completed")

    return result


@registry.tool(
    description=(
        "Run the feedback sub-agent to critique the current disease profile "
        "synthesis from multiple expert perspectives. Identifies gaps in "
        "evidence and suggests targeted searches to fill them."
    ),
    tags=["orchestration"],
    params={},
    returns="Structured feedback with identified gaps and search suggestions",
)
def run_feedback_agent(ctx: ActionContext) -> str:
    """Instantiate and run a FeedbackAgent with the shared context."""
    from targetsearch.agents.feedback import FeedbackAgent
    from targetsearch.core.config import config

    # Enforce feedback round limit
    if ctx.synthesis_state.feedback_rounds >= config.max_feedback_rounds:
        return (
            f"Feedback round limit reached ({config.max_feedback_rounds}). "
            "No further feedback cycles will be run. Produce your final answer."
        )

    agent = FeedbackAgent()

    # Build user message from current context state
    profile = ctx.synthesis_state.profile
    if profile is None:
        return "ERROR: No synthesis has been run yet. Call synthesize_disease_profile first."

    # Provide the profile and paper summaries for critique
    profile_json = json.dumps(
        profile.model_dump(exclude={"paper_summaries"}),
        indent=2,
        default=str,
    )

    n_papers = len(ctx.paper_state.summaries)
    sy = ctx.synthesis_state

    user_message = (
        f"Critique this disease profile for {ctx.disease_info.name}.\n\n"
        f"## Synthesized Profile\n\n{profile_json}\n\n"
        f"## Evidence Base\n\n"
        f"- {n_papers} papers summarized\n"
        f"- {len(ctx.search_state.queries_executed)} search queries executed\n"
        f"- {len(ctx.target_state.opentargets_results)} Open Targets results\n"
    )

    # Include synthesis diagnostics if available
    diagnostics_parts = []
    if sy.quality_status:
        diagnostics_parts.append(f"Quality status: {sy.quality_status}")
    if sy.quality_scores:
        scores_str = ", ".join(f"{k}: {v:.1f}" for k, v in sy.quality_scores.items())
        diagnostics_parts.append(f"Section scores: {scores_str}")
    if sy.coverage_by_bucket:
        cov_str = ", ".join(f"{k}: {v}" for k, v in sy.coverage_by_bucket.items())
        diagnostics_parts.append(f"Evidence coverage: {cov_str}")
    if sy.contradiction_notes:
        diagnostics_parts.append(
            "Contradictions:\n" + "\n".join(f"  - {c}" for c in sy.contradiction_notes)
        )
    if sy.unresolved_claims:
        diagnostics_parts.append(
            "Unresolved claims:\n" + "\n".join(f"  - {u}" for u in sy.unresolved_claims)
        )
    if diagnostics_parts:
        user_message += (
            "\n## Synthesis Diagnostics\n\n" + "\n".join(diagnostics_parts) + "\n"
        )

    log.info("run_feedback_agent: starting critique")
    result = agent.run(user_message, context=ctx)

    # Parse gaps and overall assessment from the feedback
    gaps = _extract_gaps(result)
    if gaps:
        ctx.synthesis_state.gaps = gaps

    overall = _extract_overall_assessment(result)
    if overall:
        ctx.synthesis_state.overall_assessment = overall

    ctx.synthesis_state.feedback_rounds += 1

    log.info(
        "run_feedback_agent: completed, %d gaps identified, assessment=%s",
        len(gaps),
        overall or "not found",
    )
    return result


def _extract_overall_assessment(feedback_text: str) -> str:
    """Extract the OVERALL assessment from feedback agent output.

    Looks for a line like ``OVERALL: adequate`` or
    ``OVERALL: needs more evidence``.  Returns the value after the colon,
    or an empty string if not found.
    """
    for line in feedback_text.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("OVERALL:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _extract_gaps(feedback_text: str) -> list[str]:
    """Extract gap descriptions from feedback agent output.

    Looks for lines matching the pattern:
      1. [gap description] → Search: "..."
    or numbered lines under a GAPS: header.
    """
    gaps = []
    in_gaps_section = False

    for line in feedback_text.split("\n"):
        line = line.strip()

        if line.upper().startswith("GAPS:"):
            in_gaps_section = True
            continue
        if line.upper().startswith("OVERALL:") or line.upper().startswith("REASONING:"):
            in_gaps_section = False
            continue

        if in_gaps_section and line:
            # Strip leading number and punctuation
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned:
                # Extract just the gap description (before → Search:)
                if "→" in cleaned:
                    cleaned = cleaned.split("→")[0].strip()
                elif "->" in cleaned:
                    cleaned = cleaned.split("->")[0].strip()
                gaps.append(cleaned)

    return gaps
