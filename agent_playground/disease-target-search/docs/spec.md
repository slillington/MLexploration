# Spec: Early Budget Gating and Review Summarization

## Problem Statement

The paper budget (`max_papers`) is checked too late in the pipeline. The current flow is:

```
search → triage → fetch → mine reviews → fetch mined papers → [budget check] → summarize
```

Everything before the budget check runs unconditionally. In the NSCLC test run, the gap-fill pass searched, triaged, fetched, and then hit "budget reached" — all those API calls and LLM calls produced nothing.

Additionally, review mining inflates the paper count. In the test run, 11 review mining calls consumed 139K prompt tokens (16% of total) and 667 seconds (38% of wall-clock time) to discover 19 PMIDs. Those mined papers competed with search results for the same budget slots. A review article is more valuable as a high-density summary than as a source of PMIDs to chase.

## Changes

### Change 1: Replace review mining with review summarization

**Current behavior:** `batch_summarize_papers` calls `mine_review_references` on each review with full text. This extracts cited PMIDs, fetches those papers, and adds them to the summarization queue — where they compete with search results for budget slots.

**New behavior:** Reviews are summarized using the same `summarize_paper` path as primary papers, but with review-specific extraction guidelines that maximize the information extracted from the review itself. No PMIDs are mined or chased.

**Prompt change:** Replace the current review extraction guidelines in `_EXTRACTION_GUIDELINES["review"]` with guidelines designed for the `paper_summary` schema (not the `review_mining` schema). The current guidelines talk about extracting cited papers with priorities — that's the mining schema. The new guidelines should instruct the model to extract the review's synthesized claims as key findings, with inline PMID citations where the review provides them.

#### New review extraction guidelines (full text)

```
## Extraction guidelines (review article)

A review article synthesizes evidence from many primary sources. Your
job is to extract the review's key conclusions as structured findings,
not to catalog every paper it cites.

### What to extract

- Extract the review's major conclusions as key_findings. Each finding
  should be a synthesized claim that the review supports with evidence
  from multiple primary sources.
  Example: "EGFR exon 20 insertion mutations occur in 2-3% of NSCLC
  and are resistant to first- and second-generation EGFR TKIs but
  respond to amivantamab and mobocertinib (PMID 34911336, 35534623)."

- For each finding, include the PMIDs of the primary sources the review
  cites as evidence, in the genes_proteins field list the relevant
  genes, and in evidence_type use "review synthesis".

- Include effect sizes when the review reports pooled or comparative
  statistics (e.g., "pooled ORR 40% across 3 trials").

- For model_system, describe the scope of evidence the review covers
  (e.g., "meta-analysis of 12 phase II-III trials", "narrative review
  of preclinical and clinical data").

### What NOT to extract

- Do not extract background statements that provide no actionable
  insight (e.g., "Lung cancer is the leading cause of cancer death").
- Do not extract every individual study the review mentions. Extract
  the review's synthesized conclusions, not a list of its references.
- Do not set evidence_type to "in vivo" or "in vitro" — the review
  itself is not performing experiments. Use "review synthesis" or
  "meta-analysis" as appropriate.

### Prioritization

Extract findings that address these dimensions of target assessment:

1. **Target biology and causal chain.** Pathway mechanisms linking
   target modulation to disease outcome. Feedback loops, compensatory
   pathways, or redundancies that could limit efficacy. On-target
   safety liabilities based on the target's normal physiological role
   (e.g., knockout phenotypes, tissue expression, known toxicities).

2. **Genetic support.** Human genetic evidence linking the target to
   disease: GWAS associations, rare variant studies, Mendelian
   randomization, eQTL data. Whether the genetic evidence supports
   inhibition vs. activation. Genetic signals defining patient
   subpopulations likely to respond.

3. **Druggability and modality.** Whether the target can be drugged
   and with what modality (antibody, ADC, bispecific, small molecule,
   PROTAC, cell therapy). Modality-specific challenges: PK/PD,
   tissue penetration, antigen density, internalization, half-life.
   Comparative efficacy across modalities when reviewed.

4. **Target expression and accessibility.** Cell-surface expression
   levels and tumor selectivity vs. normal tissue expression (safety
   window). For biologics: internalization rate (relevant for ADCs),
   shedding, antigen density. Expression heterogeneity across tumor
   subtypes or disease stages.

5. **Clinical relevance and competitive landscape.** Programs in the
   clinic targeting this pathway. Biomarker-defined patient subgroups
   and response predictors. Resistance mechanisms to existing
   therapies. Regulatory precedents and endpoints. Unmet therapeutic
   needs that a new agent could address.

6. **Differentiation from existing agents.** What distinguishes
   molecules targeting the same pathway: binding epitope, valency,
   Fc engineering, payload chemistry (ADCs), conditional activity,
   bispecific geometry. Head-to-head or cross-trial comparisons when
   available.

7. **Translational evidence.** Whether preclinical efficacy
   translated to clinical benefit — and when it did not, why (e.g.,
   insufficient tumor penetration, compensatory pathway activation,
   inadequate patient selection). Predictive biomarkers that emerged
   from translational studies.

8. **Combination rationale.** Mechanistic basis for combination
   therapies (e.g., anti-PD-1 + anti-TIGIT, EGFR + MET bispecific).
   Synergy data from preclinical or clinical studies. Whether
   combinations overcome resistance to monotherapy. Sequencing
   considerations.

9. **Patient stratification and biomarkers.** Companion diagnostic
   feasibility and availability. Prevalence of the biomarker-positive
   population. Whether the biomarker is prognostic vs. predictive.
   Co-occurrence patterns that define responder subgroups (e.g., EGFR
   mutation + TP53 co-mutation).

Deprioritize: historical context, epidemiology, staging/diagnosis,
and general disease biology that does not inform target selection,
druggability, or clinical strategy.

### Citation handling

- You will receive a REFERENCE LIST of valid PMIDs from this article's
  bibliography. ONLY cite PMIDs that appear in that list.
- If the review attributes a finding to a study whose PMID is in the
  reference list, include it as "(PMID XXXXXXXX)" in the finding text.
- If the review attributes a finding to a study whose PMID is NOT in
  the reference list, describe the finding without a citation. Do not
  guess or fabricate PMIDs.
- If no reference list is provided (abstract-only reviews), do not
  include any inline PMID citations.
- Format: "(PMID XXXXXXXX)" or "(PMID XXXXXXXX, YYYYYYYY)" for
  multiple sources.

### Fields

- paper_type: "narrative review", "systematic review", or "meta-analysis"
- study_design: describe the review's scope and methodology (e.g.,
  "systematic review of biologic therapies in NSCLC, 2018-2024,
  covering 45 clinical trials")
- evidence_strength: assess based on the review's rigor:
  - strong: systematic review or meta-analysis with clear methodology
  - moderate: narrative review by domain experts with comprehensive
    coverage
  - weak: brief or selective review, opinion-heavy
  - insufficient: abstract-only or commentary without systematic
    evidence assessment
- target_relevance: summarize the review's overall implications for
  drug target selection — which targets does it highlight as most
  promising and why?
- Aim for 8-15 key_findings. Fewer is acceptable if the review is
  narrow; more than 15 suggests you are extracting individual studies
  rather than synthesized conclusions.
```

**Files:** `targetsearch/tools/prompt_tools.py`

#### Reference list injection in `summarize_paper`

When summarizing a review, append the structured reference list (already available from PMC fetch) to the user message as a PMID whitelist:

```python
# In summarize_paper, after building user_message:
if "review" in source_type.lower() and metadata.get("references"):
    ref_block = (
        "\n\nREFERENCE LIST (valid PMIDs from this article's bibliography):\n"
        + ", ".join(metadata["references"])
        + "\n\nOnly cite PMIDs from this list. If a finding's source is "
        "not in this list, describe the finding without a PMID citation."
    )
    user_message += ref_block
```

For abstract-only reviews (no structured references), the reference list is absent and the guidelines instruct the model to omit inline citations entirely.

**Files:** `targetsearch/tools/paper_tools.py`

### Change 2: Remove review mining from `batch_summarize_papers`

Remove the entire review mining block from `batch_summarize_papers`. This includes:
- The `mine_review_references` call loop
- The mined PMID deduplication
- The `_fetch_papers_parallel` call for mined papers
- The `ctx.paper_state.review_pmids_discovered` updates
- The `feedback_rounds == 0` guard (no longer needed since mining is gone)

Keep `mine_review_references` as a registered tool in `paper_tools.py` — don't delete it. It may be useful for manual exploration or future use cases.

Remove `review_pmids` tracking from `fetch_and_classify_papers` — the classification of papers as reviews vs primaries can stay (it's useful metadata), but the `ctx.properties["review_pmids"]` list is no longer consumed by anything.

**Files:** `targetsearch/tools/coordination_tools.py`

### Change 3: Early budget check in `fetch_and_classify_papers`

Add a budget check at the top of `fetch_and_classify_papers`, before any fetching:

```python
already_count = len(ctx.paper_state.summaries)
if ctx.synthesis_state.feedback_rounds == 0:
    remaining = config.max_papers_initial - already_count
else:
    remaining = config.max_papers_gap_fill - max(0, already_count - config.max_papers_initial)

if remaining <= 0:
    return (f"Paper budget exhausted ({already_count} summarized). "
            "Skipping fetch — proceed to synthesis.")

if len(new_pmids) > remaining:
    log.info("fetch_and_classify_papers: trimming %d PMIDs to %d (budget)",
             len(new_pmids), remaining)
    new_pmids = new_pmids[:remaining]
```

**Files:** `targetsearch/tools/coordination_tools.py`

### Change 4: Expose remaining budget in context summary

Add budget information to the context summary so the searcher can make informed decisions:

```
Papers: 10/12 initial budget (2 remaining)
```

or on gap-fill:

```
Papers: 3/8 gap-fill budget (5 remaining)
```

**Files:** `targetsearch/core/context.py`

### Change 5: Budget-aware triage

Add an optional `max_recommended` parameter to `triage_search_results`. When set, triage returns at most that many recommended PMIDs, selecting the highest-relevance papers. This prevents triaging 15 papers when only 3 budget slots remain.

The searcher can pass `max_recommended` based on the budget info in the context summary, or `batch_summarize_papers` can call triage internally with the remaining budget.

**Files:** `targetsearch/tools/triage_tools.py`

## Acceptance Criteria

1. **Review mining removed:** `batch_summarize_papers` does not call `mine_review_references`. No mined PMIDs are fetched or added to the summarization queue.
2. **Reviews are summarized as papers:** Reviews go through `summarize_paper` with the new review guidelines. The output is a `PaperSummary` with `paper_type` set to "narrative review", "systematic review", or "meta-analysis".
3. **Review summaries contain inline citations:** Key findings include "(PMID XXXXXXXX)" where the review provides citations.
4. **Early budget check:** `fetch_and_classify_papers` returns immediately when budget is exhausted. PMIDs are trimmed to remaining budget before fetching.
5. **Budget in context summary:** The searcher sees remaining budget in the context summary.
6. **Budget-aware triage:** `triage_search_results` accepts `max_recommended` and respects it.
7. **All existing tests pass** (291+), with new tests for budget gating and review summarization.

## Implementation Order

1. Change 3 — early budget check (immediate waste reduction, no behavioral change)
2. Change 4 — budget in context summary (enables informed searcher decisions)
3. Change 1 — new review extraction guidelines (prompt change only)
4. Change 2 — remove review mining block (depends on Change 1 being in place)
5. Change 5 — budget-aware triage (refinement, depends on Change 3)

## Files Touched

| File | Changes |
|---|---|
| `targetsearch/tools/coordination_tools.py` | Early budget check in `fetch_and_classify_papers`, remove review mining block from `batch_summarize_papers`, remove `review_pmids` tracking |
| `targetsearch/tools/prompt_tools.py` | Replace `_EXTRACTION_GUIDELINES["review"]` with new review summarization guidelines |
| `targetsearch/tools/triage_tools.py` | Add `max_recommended` parameter |
| `targetsearch/core/context.py` | Add budget info to context summary |
| `tests/test_coordination_tools.py` | Tests for early budget check, review mining removal |
| `tests/test_prompt_tools.py` | Update review guidelines test |

## Estimated Token Savings

| Source | Before | After | Savings |
|---|---|---|---|
| `mine_review_references` (11 calls) | 187K | 0 | -187K |
| Mined paper fetch + summarize (~19 papers) | ~90K | 0 | -90K |
| Wasted fetch on budget-exhausted gap-fill | ~30K | 0 | -30K |
| **Total** | | | **~307K/run** |

Wall-clock savings: ~12-15 minutes per run (review mining was 667s, mined paper processing ~200s).