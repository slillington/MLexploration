"""Disease Intelligence agent — orchestrator architecture.

The DiseaseIntelAgent is a true orchestrator that plans, executes tools,
evaluates gaps, and loops. It sees only high-level tools:
  - run_search_agent: dispatch the search sub-agent
  - run_feedback_agent: dispatch the feedback sub-agent
  - synthesize_disease_profile: synthesize accumulated evidence

The orchestrator's loop: search → synthesize → feedback → loop or stop.

Usage:
    from targetsearch.agents.disease_intel import DiseaseIntelAgent

    agent = DiseaseIntelAgent()
    profile = agent.run("idiopathic pulmonary fibrosis")
"""

from __future__ import annotations

import logging

from targetsearch.agents.base import Agent
from targetsearch.core.context import ActionContext
from targetsearch.schemas.disease import DiseaseProfile

# Import tool modules to register them with the global registry
import targetsearch.tools.agent_tools  # noqa: F401
import targetsearch.tools.synthesis_tools  # noqa: F401

log = logging.getLogger(__name__)

_ORCHESTRATOR_PROMPT = """\
You are a drug target discovery orchestrator. Your job is to build a \
comprehensive disease profile by coordinating search and analysis sub-agents.

## Available tools

- run_search_agent(disease_area, gaps_to_fill): Dispatches a search \
sub-agent that finds papers, fetches full text, AND summarizes them \
into structured paper summaries. Call this first to gather evidence. \
If gaps are identified later, call it again with specific gaps_to_fill.

- synthesize_disease_profile(): Synthesizes all accumulated paper \
summaries and Open Targets data into a DiseaseProfile. This tool \
REQUIRES paper summaries to exist — it will fail if run_search_agent \
has not completed successfully first.

- run_feedback_agent(): Dispatches a feedback sub-agent that critiques \
the synthesis from multiple expert perspectives (geneticist, medicinal \
chemist, clinician, bioinformatician). Returns identified gaps and \
search suggestions.

## Workflow — follow this order exactly

1. Call run_search_agent with the disease name. Wait for it to complete. \
It will search, fetch, and summarize papers automatically.
2. Call synthesize_disease_profile ONLY after run_search_agent has \
finished. If synthesis returns an error about missing paper summaries, \
call run_search_agent again before retrying.
3. Call run_feedback_agent to critique the profile and identify gaps.
4. If significant gaps are identified, call run_search_agent again with \
the gaps_to_fill parameter, then call synthesize_disease_profile again.
5. Stop when the feedback agent reports the profile is adequate, or \
after 2 feedback rounds (to avoid infinite loops).

## Critical ordering rules

- NEVER call synthesize_disease_profile before run_search_agent.
- NEVER call run_feedback_agent before synthesize_disease_profile.
- Each tool depends on the output of the previous one.
- When re-searching to fill gaps, pass the specific gaps as gaps_to_fill \
so the search agent can focus.
- After the final synthesis, produce a brief summary of the disease \
profile as your final text response.
"""


class DiseaseIntelAgent(Agent):
    """Orchestrates the full disease intelligence pipeline.

    Subclass of Agent — uses the standard tool-calling loop with
    orchestration-level tools only.
    """

    name = "disease_intel"
    tool_tags = ["orchestration", "synthesis"]

    def __init__(self) -> None:
        super().__init__(system_prompt=_ORCHESTRATOR_PROMPT)

    def run(self, disease_name: str) -> DiseaseProfile:  # type: ignore[override]
        """Build a disease profile through the orchestrator loop.

        Args:
            disease_name: Disease to research.

        Returns:
            A DiseaseProfile with paper_summaries populated.
        """
        ctx = ActionContext()
        ctx.disease_info.name = disease_name

        log.info("[%s] Starting orchestration for: %s", self.name, disease_name)

        # Run the orchestrator's tool-calling loop
        user_message = (
            f"Build a comprehensive disease profile for drug target discovery: "
            f"{disease_name}"
        )
        super().run(user_message, context=ctx)

        # Extract the profile from context
        profile = ctx.synthesis_state.profile
        if profile is None:
            log.warning("[%s] No profile in context after orchestration", self.name)
            profile = DiseaseProfile(
                disease_name=disease_name,
                description="Orchestration completed but no profile was synthesized.",
            )

        return profile
