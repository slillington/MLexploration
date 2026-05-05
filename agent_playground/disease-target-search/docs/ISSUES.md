# Known Issues

## `gene_ontology_lookup` fails with gene symbols

**File:** `targetsearch/tools/ontology.py`

QuickGO's API requires UniProt accessions (e.g. `Q9HC29`), not gene symbols (e.g. `NOD2`). The current implementation passes gene symbols directly, which returns a 400 error.

**Fix:** Add a gene symbol → UniProt accession resolution step using the UniProt REST API before calling QuickGO.

---

## ActionContext is mutable shared state — unsafe for future parallelism

**Files:** `targetsearch/core/context.py`, `targetsearch/tools/coordination_tools.py`

ActionContext is mutated in place by coordination tools. This is safe today because all parallelism happens at the leaf tool level (leaf tools don't touch the context), and coordination tools run sequentially within a single agent's tool-calling loop. However, three planned extensions would break this:

### 1. Parallel sub-agents writing to the same context

Running multiple SearcherAgents in parallel (e.g., one for genetic evidence, another for clinical trial data) would cause concurrent writes to `ctx.search_state.pmids_collected`, `ctx.paper_state.summaries`, and `ctx.paper_state.papers_fetched`. Python lists are not thread-safe for concurrent mutation. Both agents could pass the PMID dedup check before either writes, causing duplicate processing and corrupted counters.

### 2. Streaming pipeline (feedback-while-searching)

Starting synthesis and feedback on early results while the searcher is still finding papers would cause the FeedbackAgent to read `ctx.paper_state.summaries` while `batch_summarize_papers` is still appending to it. Feedback would be based on incomplete data, and gap assessments would not match the profile they're critiquing.

### 3. Multi-disease comparative analysis

Running two full pipelines (e.g., IPF and COPD) sharing a context to find target overlaps would cause both to overwrite `ctx.disease_info`, mix papers in `ctx.paper_state.summaries` with no provenance tracking, and race on `ctx.synthesis_state.profile` (last write wins).

### Possible solutions

- **Scoped contexts**: Each sub-agent gets a fork of the parent context, merged back under a lock.
- **Immutable snapshots**: Tools read frozen snapshots and return deltas that a single coordinator applies.
- **Section-level locking**: `threading.Lock` per context section, acquired by coordination tools before mutation.

The current leaf/coordination tool boundary is the right tradeoff for the single-disease sequential pipeline. These solutions should be implemented when the extensions above are needed.

---

## Design doc drift: ontology and Open Targets results are not persisted into `ActionContext`

**Files:** `targetsearch/tools/ontology.py`, `targetsearch/tools/targets.py`, `targetsearch/tools/synthesis_tools.py`, `targetsearch/core/context.py`

The architecture in `targetsearch/DESIGN.md` says disease ontology resolution should populate `ctx.disease_info` and Open Targets queries should populate `ctx.target_state`. The current implementations of `disease_ontology_search` and `opentargets_disease_targets` are stateless leaf tools and do not write their results into the shared context.

That leaves several context fields effectively unused in the main disease-intel path:

- `ctx.disease_info.synonyms`
- `ctx.disease_info.ontology_ids`
- `ctx.target_state.opentargets_results`
- `ctx.target_state.known_drugs`

This matters because `synthesize_disease_profile()` reads Open Targets results and disease synonyms from the context, so the synthesis step currently gets less structured evidence than the design intends unless another tool manually stores it first.

**Fix:** Either convert these tools into context-aware coordination tools that update `ActionContext`, or add small wrapper tools that call the existing leaf tools and persist the relevant fields into `ctx.disease_info` and `ctx.target_state`.

---

## Search bookkeeping fields exist in `ActionContext` but are never updated

**Files:** `targetsearch/core/context.py`, `targetsearch/tools/literature.py`, `targetsearch/tools/agent_tools.py`

`SearchState` tracks `queries_executed` and `total_papers_found`, and both the context summary and feedback prompt report those values. However, the current search flow never records search queries or result counts when calling `pubmed_search` or `semantic_scholar_search`.

As a result:

- `ctx.search_state.queries_executed` stays empty
- `ctx.search_state.total_papers_found` stays `0`
- agent summaries under-report what the searcher has actually done
- the feedback agent critiques an incomplete evidence-base summary

This is a design/implementation mismatch rather than a cosmetic issue, because the orchestrator relies on context summaries to decide what has already happened.

**Fix:** Introduce context-aware wrappers for literature search calls, or have the SearcherAgent use coordination tools that both invoke the leaf search tools and update `ctx.search_state` with the query text and number of hits returned.

---

## Feedback round limit is described in prompts but not enforced in code

**Files:** `targetsearch/agents/disease_intel.py`, `targetsearch/tools/agent_tools.py`, `targetsearch/core/context.py`

The orchestrator prompt says the pipeline should stop after 2 feedback rounds to avoid infinite loops. The code increments `ctx.synthesis_state.feedback_rounds`, but nothing checks that counter and no hard stop is implemented.

Today the system relies on the LLM following prompt instructions rather than on deterministic control flow. If the model keeps finding new gaps, the loop can continue until some other limit is hit.

**Fix:** Enforce the round cap in code. The simplest option is to check `ctx.synthesis_state.feedback_rounds` before dispatching another feedback or gap-filling search cycle and terminate the orchestration loop once the configured maximum is reached.

---

## FeedbackAgent cannot verify citations against paper summaries

**Files:** `targetsearch/agents/feedback.py`, `targetsearch/tools/agent_tools.py`

The FeedbackAgent critiques the synthesized profile but has no way to check whether a claim in the profile is actually supported by the cited paper. It receives the profile JSON and aggregate evidence stats, but not the underlying paper summaries. It can identify coverage gaps (missing evidence buckets) but cannot fact-check citations — e.g., whether PMID 12345678 actually reports the EGFR finding attributed to it.

This means unsupported or misattributed claims in the profile pass through feedback unchallenged, and the critique is limited to structural/coverage issues.

**Fix:** Add a read-only `get_paper_summary(pmid)` tool that returns the `PaperSummary` for a given PMID from `ctx.paper_state.summaries`. Give the FeedbackAgent access to this tool via a dedicated tag (e.g., `"evidence_read"`). The agent could then spot-check high-impact claims by retrieving the cited paper's summary and comparing the finding text against the profile's assertion.
