# disease-target-search

An LLM-driven multi-agent system for **drug target discovery**. Given a disease name, it searches biomedical literature, extracts structured findings, synthesizes a disease profile, and iterates via expert feedback to fill knowledge gaps.

## Quick Start

```bash
cd disease-target-search
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run with default disease (idiopathic pulmonary fibrosis)
python run_disease_intel.py

# Run with a specific disease
python run_disease_intel.py "rheumatoid arthritis"
```

Requires a `GITHUB_TOKEN` (model: `github_copilot/gpt-5.4`) and optionally an NCBI API key in `.env`.

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

## Package Structure

```
disease-target-search/
├── run_disease_intel.py          # CLI entry point
├── pyproject.toml                # Package metadata & dependencies
├── targetsearch/                 # Main package
│   ├── agents/                   # Agent definitions
│   │   ├── base.py              # Base Agent (function-calling loop)
│   │   ├── disease_intel.py     # Orchestrator agent
│   │   ├── searcher.py          # Literature search sub-agent
│   │   └── feedback.py          # Expert feedback sub-agent
│   ├── core/                    # Framework primitives
│   │   ├── config.py            # Centralized config (models, keys, budgets)
│   │   ├── context.py           # ActionContext — typed working memory
│   │   ├── llm.py              # LiteLLM wrapper
│   │   ├── registry.py         # @tool decorator & ToolRegistry
│   │   └── skills.py           # SKILL.md file discovery & loading
│   ├── schemas/                 # Pydantic data models
│   │   ├── disease.py          # DiseaseProfile, Pathway, GeneticAssociation
│   │   ├── paper.py            # PaperSummary, KeyFinding
│   │   ├── target.py           # TargetHypothesis, TherapeuticStrategy
│   │   └── evaluation.py       # FeasibilityScore, EvaluationReport
│   └── tools/                   # Registered tool functions
│       ├── literature.py        # PubMed & Semantic Scholar search
│       ├── targets.py           # Open Targets disease/drug queries
│       ├── ontology.py          # Disease & Gene Ontology lookups
│       ├── fulltext.py          # PMC full-text & PubMed abstract fetch
│       ├── paper_tools.py       # LLM-based paper summarization
│       ├── coordination_tools.py # Compose fetch + classify + summarize
│       ├── synthesis_tools.py   # Aggregate findings into disease profile
│       ├── agent_tools.py       # Sub-agent dispatch tools
│       ├── prompt_tools.py      # Expert persona & schema templates
│       └── triage_tools.py      # Paper classification & filtering
├── docs/                        # Design documents & specs
├── scripts/                     # Standalone exploration utilities
├── tests/                       # Deterministic test suite
└── testrun-evaluation/          # Output quality & performance analysis
```

## Core Concepts

### Tool Registry

All LLM-callable capabilities are decorated with `@registry.tool(tags=[...])`. This enables:
- Automatic OpenAI-compatible function schema generation from type hints
- Tag-based filtering — each agent sees only its relevant tools
- A clean boundary between "things the LLM can call" and internal helpers

### ActionContext (Working Memory)

Typed, structured state that flows through the pipeline:
- **disease_info** — name, synonyms, ontology IDs
- **search_state** — queries executed, PMIDs collected
- **paper_state** — summaries (list of `PaperSummary`)
- **target_state** — Open Targets results, known drugs
- **synthesis_state** — disease profile, gaps, feedback rounds

Tools read/write the context directly; the LLM sees a compressed summary (~200 tokens) injected before each turn.

### Agent Base Class

Implements the OpenAI function-calling loop:
1. Send messages → LLM responds with `tool_calls`
2. Execute matching tool function (auto-injecting `ActionContext`)
3. Append result to messages → repeat until the agent signals completion

Subclasses define: `system_prompt`, `tool_tags`, and `parse_output()`.

## Tools by Agent

| Agent | Tags | Key Tools |
|-------|------|-----------|
| **Orchestrator** | `orchestration`, `synthesis` | `run_search_agent`, `run_feedback_agent`, `synthesize_disease_profile` |
| **Searcher** | `literature`, `targets`, `ontology`, `coordination` | `pubmed_search`, `semantic_scholar_search`, `opentargets_*`, `fetch_and_classify_papers`, `batch_summarize_papers` |
| **Feedback** | `feedback`, `prompts` | `create_expert_persona`, `create_output_schema` |

## Scripts

Standalone utilities for exploring individual APIs and tools:

| Script | Purpose |
|--------|---------|
| `search_pubmed.py` | PubMed API exploration |
| `search_semantic_scholar.py` | Semantic Scholar queries |
| `explore_ontology.py` | Disease/Gene Ontology testing |
| `explore_targets.py` | Open Targets query testing |
| `explore_registry.py` | Tool registry introspection |
| `summarize_paper.py` | Single-paper summarization |
| `get_full_text.py` | Full-text fetch testing |
| `mine_review.py` | Review reference mining |

## Tests

```bash
pytest tests/
```

All tests are deterministic (LLM and API calls are mocked). Coverage spans the agent loop, tool registry, paper classification, context management, schema validation, and coordination logic.

## Dependencies

- [LiteLLM](https://github.com/BerriAI/litellm) — provider-agnostic LLM calls
- [Pydantic](https://docs.pydantic.dev/) ≥2.0 — structured data validation
- [httpx](https://www.python-httpx.org/) / requests — HTTP clients
- python-dotenv — environment configuration
- Python ≥3.12
