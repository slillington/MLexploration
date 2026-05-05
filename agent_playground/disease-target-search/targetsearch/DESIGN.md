# targetsearch — Architecture & Information Flow

## Overview

`targetsearch` is a tool-centric agent system for drug target discovery. Given a disease name, it searches biomedical literature and databases, summarizes papers, synthesizes a disease profile, and iterates based on expert feedback — all driven by LLM tool-calling loops.

The architecture follows three principles:

1. **Tools as the LLM-facing interface.** Every capability the LLM can invoke is a registered tool. Internal helpers are plain Python functions.
2. **ActionContext as structured working memory.** A typed state object flows through the pipeline. Tools read and write it; the LLM sees a concise summary.
3. **Orchestrator-driven workflow.** The top-level agent plans, dispatches sub-agents, evaluates gaps, and loops — rather than following a hardcoded sequence.

## Package Structure

```
targetsearch/
├── core/
│   ├── config.py          # Model, API keys, limits
│   ├── context.py         # ActionContext and typed sections
│   ├── llm.py             # litellm wrapper (github_copilot/gpt-5-mini)
│   ├── registry.py        # @tool decorator, ToolRegistry, auto-injection
│   └── skills.py          # SKILL.md loader
├── tools/
│   ├── literature.py      # pubmed_search, semantic_scholar_search
│   ├── targets.py         # opentargets_disease_targets, opentargets_target_drugs
│   ├── ontology.py        # disease_ontology_search, gene_ontology_lookup
│   ├── fulltext.py        # pmc_fulltext_fetch, pubmed_fetch_by_pmids, helpers
│   ├── prompt_tools.py    # create_expert_persona, create_output_schema, ...
│   ├── paper_tools.py     # summarize_paper, mine_review_references
│   ├── synthesis_tools.py # synthesize_disease_profile
│   ├── coordination_tools.py  # fetch_and_classify_papers, batch_summarize_papers
│   └── agent_tools.py     # run_search_agent, run_feedback_agent
├── agents/
│   ├── base.py            # Agent base class with tool-calling loop
│   ├── disease_intel.py   # DiseaseIntelAgent (orchestrator)
│   ├── searcher.py        # SearcherAgent (sub-agent)
│   └── feedback.py        # FeedbackAgent (sub-agent)
└── schemas/
    ├── disease.py         # DiseaseProfile, Pathway, GeneticAssociation, ...
    ├── paper.py           # PaperSummary, KeyFinding
    ├── target.py          # TargetHypothesis, Target, TherapeuticStrategy, ...
    └── evaluation.py      # FeasibilityScore, EvaluationReport
```

## Tool Registry & Auto-Injection

Tools are plain Python functions decorated with `@registry.tool(...)`. The decorator records metadata (description, tags, caching) without changing the function's behavior. At runtime:

- **Tag-based filtering** controls which tools each agent sees. The orchestrator sees `["orchestration", "synthesis"]`; the searcher sees `["literature", "targets", "disease", "ontology", "coordination"]`.
- **OpenAI schema generation** inspects type hints to produce function-calling schemas automatically.
- **ActionContext auto-injection**: If a tool's signature includes a parameter typed as `ActionContext`, the agent loop injects it when dispatching the call. The LLM never sees this parameter in the schema.

### Tool Categories

| Category | ActionContext? | Examples | Called by |
|----------|---------------|----------|-----------|
| **Leaf tools** | No | `pubmed_search`, `summarize_paper`, `create_expert_persona` | Sub-agents, coordination tools |
| **Coordination tools** | Yes | `fetch_and_classify_papers`, `batch_summarize_papers` | SearcherAgent |
| **Synthesis tools** | Yes | `synthesize_disease_profile` | Orchestrator |
| **Agent tools** | Yes | `run_search_agent`, `run_feedback_agent` | Orchestrator |

Leaf tools are stateless, independently testable, and safe for parallel execution. Coordination tools manage state and compose leaf tools internally.

## ActionContext

`ActionContext` is a dataclass with typed sections that serves as shared working memory:

```python
@dataclass
class ActionContext:
    context_id: str          # UUID for tracing
    disease_info: DiseaseInfo       # name, synonyms, ontology IDs
    search_state: SearchState       # queries executed, PMIDs collected
    paper_state: PaperState         # papers fetched, summaries (list[PaperSummary])
    target_state: TargetState       # Open Targets results, known drugs
    synthesis_state: SynthesisState # profile, gaps, feedback rounds
    metadata: Metadata              # timestamps, tool call count, iteration count
    properties: dict[str, Any]      # escape hatch for ad-hoc data
```

The `summarize()` method produces a ~200-token status report injected into the LLM's messages before each turn. This tells the LLM what has been done and what's missing, without re-reading all prior tool results. Tools that need actual data (e.g., `synthesize_disease_profile`) read the full, uncompressed objects from the typed sections directly.

## Agent Hierarchy

```
DiseaseIntelAgent (orchestrator)
│  tool_tags: ["orchestration", "synthesis"]
│  sees: run_search_agent, run_feedback_agent, synthesize_disease_profile
│
├── SearcherAgent (sub-agent, via run_search_agent tool)
│     tool_tags: ["literature", "targets", "disease", "ontology", "coordination"]
│     sees: pubmed_search, semantic_scholar_search, opentargets_*,
│           disease_ontology_search, fetch_and_classify_papers,
│           batch_summarize_papers
│
└── FeedbackAgent (sub-agent, via run_feedback_agent tool)
      tool_tags: ["prompts"]
      sees: create_expert_persona, create_output_schema,
            create_extraction_guidelines
```

Each agent is an `Agent` subclass with its own system prompt, tool subset, and tool-calling loop. Sub-agents share the orchestrator's `ActionContext` and modify it in place.

## Information Flow — Worked Example

Here's what happens when you run:

```python
agent = DiseaseIntelAgent()
profile = agent.run("idiopathic pulmonary fibrosis")
```

### Phase 1: Search

The orchestrator calls `run_search_agent(disease_area="idiopathic pulmonary fibrosis")`.

This instantiates a `SearcherAgent` and runs its tool-calling loop. The SearcherAgent makes several tool calls:

```
SearcherAgent loop:
  1. disease_ontology_search("idiopathic pulmonary fibrosis")
     → Gets canonical name "IPF", synonyms, MONDO/EFO IDs
     → Writes to ctx.disease_info

  2. opentargets_disease_targets("idiopathic pulmonary fibrosis")
     → Gets genetically associated targets (MUC5B, TERT, SFTPC, ...)
     → Writes to ctx.target_state

  3. pubmed_search("idiopathic pulmonary fibrosis drug targets review")
     → Returns 10 papers with PMIDs
  4. pubmed_search("IPF GWAS genetic association")
     → Returns 10 more papers
  5. pubmed_search("IPF TGF-beta fibrosis mechanism")
     → Returns 10 more papers

  6. fetch_and_classify_papers(pmids=[...30 PMIDs...])
     → Fetches metadata from PubMed (pub_types for classification)
     → Classifies: 5 reviews, 25 primary articles
     → Resolves PMC IDs, fetches full text where available
     → Writes to ctx.paper_state and ctx.search_state

  7. batch_summarize_papers(disease_area="idiopathic pulmonary fibrosis")
     → For each review with full text: calls mine_review_references
       to extract cited PMIDs (discovers 15 new papers)
     → Fetches text for newly discovered papers
     → Fans out summarize_paper in parallel (4 workers)
     → Each summarize_paper call: builds prompt with create_expert_persona
       + create_output_schema + create_extraction_guidelines, sends paper
       text to LLM, parses response into PaperSummary
     → Writes list[PaperSummary] to ctx.paper_state.summaries
```

After the SearcherAgent finishes, the context contains ~35 PaperSummary objects, Open Targets data, and search metadata.

### Phase 2: Synthesis

The orchestrator calls `synthesize_disease_profile()`.

This tool reads the full, uncompressed `list[PaperSummary]` from `ctx.paper_state.summaries` — every KeyFinding, methods_summary, limitations, and genes_pathways_mentioned field is preserved. It also reads `ctx.target_state` for Open Targets data. It builds a large user message with all this evidence, sends it to the LLM with the disease profile output schema, and parses the response into a `DiseaseProfile`. The profile is written to `ctx.synthesis_state.profile`.

### Phase 3: Feedback

The orchestrator calls `run_feedback_agent()`.

This instantiates a `FeedbackAgent` that receives the synthesized profile and evidence base statistics. The FeedbackAgent uses prompt tools to adopt multiple expert perspectives:

```
FeedbackAgent loop:
  1. create_expert_persona("IPF", ["genetics"])
     → Builds geneticist persona, evaluates genetic evidence
  2. create_expert_persona("IPF", ["medicinal chemistry"])
     → Builds chemist persona, evaluates druggability
  3. (continues for clinician, bioinformatician perspectives)
  4. Produces structured output:
     GAPS:
     1. No GWAS data for LOXL2 → Search: "LOXL2 GWAS pulmonary fibrosis"
     2. Missing clinical trial data for pirfenidone resistance → Search: ...
     OVERALL: needs more evidence
```

Identified gaps are written to `ctx.synthesis_state.gaps`.

### Phase 4: Iteration (if needed)

The orchestrator sees the gaps and calls `run_search_agent` again with `gaps_to_fill=["No GWAS data for LOXL2", ...]`. The SearcherAgent runs targeted queries, fetches and summarizes the new papers, and the orchestrator re-synthesizes and re-evaluates. This loop continues until the FeedbackAgent reports the profile is adequate or the iteration limit (2 feedback rounds) is reached.

### Result

The orchestrator extracts `ctx.synthesis_state.profile` — a `DiseaseProfile` containing:
- Disease name, synonyms, description
- Key pathways with cited evidence
- Genetic associations (GWAS, Mendelian, somatic)
- Existing therapies and their limitations
- Unmet medical needs
- Literature summary highlighting convergent evidence, contradictions, and gaps
- All PaperSummary objects attached for downstream use

## Data Flow Diagram

```
User: "idiopathic pulmonary fibrosis"
  │
  ▼
DiseaseIntelAgent.run()
  │ creates ActionContext
  │ enters tool-calling loop
  │
  ├─ run_search_agent ──────────────────────────────────────┐
  │    SearcherAgent tool-calling loop:                     │
  │    ├─ disease_ontology_search ──► ctx.disease_info      │
  │    ├─ opentargets_disease_targets ──► ctx.target_state  │
  │    ├─ pubmed_search (×3-4) ──► PMIDs                   │
  │    ├─ fetch_and_classify_papers ──► ctx.paper_state     │
  │    │    ├─ pubmed_fetch_by_pmids (metadata)             │
  │    │    ├─ pmids_to_pmcids (PMC resolution)             │
  │    │    └─ fetch_paper_text (full text / abstract)      │
  │    └─ batch_summarize_papers ──► ctx.paper_state        │
  │         ├─ mine_review_references (for reviews)         │
  │         └─ summarize_paper (×N, parallel)               │
  │              └─ llm_text() → PaperSummary               │
  │                                                         │
  ├─ synthesize_disease_profile ────────────────────────────┤
  │    reads ctx.paper_state.summaries (full, uncompressed) │
  │    reads ctx.target_state                               │
  │    └─ llm_text() → DiseaseProfile ──► ctx.synthesis     │
  │                                                         │
  ├─ run_feedback_agent ────────────────────────────────────┤
  │    FeedbackAgent tool-calling loop:                     │
  │    ├─ create_expert_persona (×4 perspectives)           │
  │    └─ produces gaps ──► ctx.synthesis_state.gaps         │
  │                                                         │
  ├─ (if gaps) run_search_agent(gaps_to_fill=[...]) ───────┤
  ├─ (if gaps) synthesize_disease_profile ─────────────────┤
  │                                                         │
  ▼                                                         │
  return ctx.synthesis_state.profile (DiseaseProfile)  ◄────┘
```

## LLM Configuration

All LLM calls go through `core/llm.py` which wraps `litellm`. The model is `github_copilot/gpt-5-mini` (configured in `core/config.py`). Two entry points:

- `llm_call(messages, tools=...)` — returns the full response object (used by the Agent base class for tool-calling)
- `llm_text(messages)` — returns just the text content (used by leaf tools like `summarize_paper`)

## Testing

Tests are organized by component. All tests for deterministic logic (classification, parsing, schema generation, context mutation) run without LLM or network calls. The test suite verifies:

- ActionContext construction, mutation, and `summarize()` output
- Auto-injection of ActionContext and exclusion from OpenAI schemas
- Tool registration, tag filtering, and schema generation
- Paper classification (review vs primary)
- JSON parsing with metadata merging
- Gap extraction from feedback text
- Agent tool visibility (orchestrator sees only its 3 tools, not leaf tools)
