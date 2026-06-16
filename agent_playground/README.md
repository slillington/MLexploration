# Agent Playground
A collection of AI agent projects demonstrating agentic design patterns. Includes a Coursera project implementing the GAME loop (Goals, Actions, Memory, Environment) for autonomous agents, a disease-target intelligence search system, and simple agent orchestration demo.

## Disease-target-search
An LLM-driven multi-agent system for **drug target discovery**. Given a disease name, it searches biomedical literature, extracts structured findings, synthesizes a disease profile, and iterates via expert feedback to fill knowledge gaps.

## Architecture

```
User Query (disease name)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              DiseaseIntelAgent (Orchestrator)            │
│                                                         │
│  Plans workflow, dispatches sub-agents, evaluates gaps   │
│  Tools: run_search_agent, run_feedback_agent,           │
│         synthesize_disease_profile                       │
└────────┬─────────────────────────────────┬──────────────┘
         │                                 │
  ┌──────▼──────────┐           ┌──────────▼──────────┐
  │  SearcherAgent  │           │   FeedbackAgent     │
  │                 │           │                     │
  │ Literature search│          │ Expert critiques    │
  │ Paper fetching  │           │ (geneticist,        │
  │ Summarization   │           │  chemist, clinician,│
  │ Target lookup   │           │  bioinformatician)  │
  └─────────────────┘           │ Gap identification  │
                                └─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   DiseaseProfile                         │
│  Pathways · Genetic associations · Existing therapies   │
│  Unmet needs · Paper summaries with key findings        │
└─────────────────────────────────────────────────────────┘
```

The orchestrator enforces strict ordering (search → synthesize → feedback → loop) and loops until the profile is sufficiently complete or the paper budget is exhausted.

# Other projects
These other projects, simple-agent-test and coursera_project, are small artifacts from the 2022 Coursera Vanderbilt course on building AI agents. The course teaches building agentic systems from scratch, including ReAct loops, tool registration and calling, and context management.