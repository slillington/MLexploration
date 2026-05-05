# Implementation Plan: Synthesis Pipeline Improvements

Based on the evaluation of run `20260418_161851_d26d3d`.

## Issue 1: Genetics section is persistently weak

**Problem:** `genetic_associations` starts at 3.4-4.1 every critique and never fully recovers. The model fills the section with expression data mislabeled as "eQTL" or with somatic alterations that define driver-*positive* disease. The critique catches this, the refine partially fixes it, and the cycle repeats.

**Root cause:** The `DiseaseProfile.genetic_associations` schema has a single `association_type` field that conflates germline population genetics, somatic tumor genomics, and tumor expression biology. The model has no structured way to say "no germline evidence found" — it fills the section with whatever it has.

**Plan:**

1. **Split the schema.** In `targetsearch/schemas/disease.py`, replace the single `genetic_associations` list with two fields:
   - `somatic_genomics: list[SomaticAlteration]` — somatic mutations, amplifications, fusions, co-occurring alterations. Fields: `gene_symbol`, `alteration_type` (mutation/amplification/fusion/loss/overexpression), `frequency` (optional), `evidence_summary`, `source`.
   - `germline_genetics: list[GermlineAssociation]` — GWAS hits, Mendelian variants, eQTL. Fields: `gene_symbol`, `association_type` (GWAS/Mendelian/eQTL), `evidence_summary`, `source`.

2. **Update the synthesis prompt** (`_build_synthesis_prompt`) to instruct the model to populate each list only with matching evidence, and to leave `germline_genetics` empty with a note if no germline evidence exists.

3. **Update the critique prompt** (`_build_critique_prompt`) to score `somatic_genomics` and `germline_genetics` separately. A missing `germline_genetics` section is acceptable if the profile explicitly states no germline evidence was found — this should not be penalized.

4. **Update the output schema** in `prompt_tools.py` for `disease_profile` and `quality_critique`.

5. **Update tests** for the new schema fields.

**Files:** `schemas/disease.py`, `tools/synthesis_tools.py`, `tools/prompt_tools.py`, `tests/test_synthesis_tools.py`

**Risk:** Schema change is breaking — `_format_compact_result`, `_merge_profiles`, and the output printer in `run_disease_intel.py` all reference `genetic_associations`. Migration is straightforward but touches many files.

---

## Issue 2: Hard failures never reach zero

**Problem:** Even the final critique (avg 8.1) has 4 hard failures. The refine pass fixes some but introduces new ones. `_assess_quality` treats any hard failure as `needs_refinement=True`, so refinement always exhausts its budget.

**Root cause:** Two problems:
1. The refine prompt says "fix hard failures" but doesn't enumerate *which specific failures* to fix, so the model may fix some and introduce others.
2. `_assess_quality` doesn't distinguish between *new* hard failures (introduced by refinement) and *persistent* ones (unfixable given the evidence).

**Plan:**

1. **Pass hard failure IDs to the refine prompt.** Number each hard failure in the critique output. In `_run_refinement`, include the numbered list and instruct: "Fix failures #1, #2, #3. Do not introduce new errors in sections you are not revising."

2. **Track hard failure persistence.** In the refine loop, compare the current critique's hard failures against the previous critique's. If the same failures persist after a refine pass (fuzzy match on topic/content), classify them as "unfixable given evidence" and stop counting them toward `needs_refinement`.

3. **Add a `max_hard_failures_for_pass` config.** If the number of *new* hard failures is ≤ this threshold (e.g., 2), treat the profile as passing even if some persistent failures remain. Default: 2.

4. **Log hard failure diff.** After each refine pass, log which failures were fixed, which persisted, and which are new.

**Files:** `tools/synthesis_tools.py`, `core/config.py`

**Complexity:** Medium. The fuzzy matching of hard failures across critique rounds is the trickiest part — start with exact string match, fall back to substring containment.

---

## Issue 3: Critique is token-inefficient (162:1 P:C ratio)

**Problem:** Each critique call sends ~98K prompt tokens (full profile + all paper summaries + audit) to produce ~600 tokens of output. The paper summaries are the bulk of the prompt but the critique barely references them — it's scoring the *profile*, not re-reading the papers.

**Plan:**

1. **Build a compact evidence index for critique.** Replace the raw `summaries_json` in the critique prompt with a condensed index:
   ```
   PMID 12345678: 5 findings, genes=[TROP2, EGFR], bucket=preclinical_therapeutic
   PMID 23456789: 3 findings, genes=[MET, HGF], bucket=clinical
   ...
   ```
   This is ~100-200 chars per paper vs ~2000-5000 chars for the full summary. For 40 papers: ~6K chars vs ~120K chars.

2. **Add `_build_evidence_index(summaries_list)` helper** that extracts PMID, finding count, gene list, and evidence bucket from each `PaperSummary`.

3. **Keep full summaries for draft and refine** — those passes need the detail. Only critique and audit use the compact index.

4. **Estimated savings:** ~90K tokens per critique call × 6 critiques = ~540K tokens saved (20% of total run cost).

**Files:** `tools/synthesis_tools.py`

**Risk:** Low. The critique's job is to score the profile, not to re-read papers. If the critique needs to verify a specific claim, the profile already contains PMID citations it can reference.

---

## Issue 4: Refine passes always max out

**Problem:** Both synthesis calls ran 2/2 refine passes. The quality threshold (6.0) is met after the first refine, but hard failures keep triggering additional passes (see Issue 2).

**Plan:**

This is largely addressed by Issue 2's fix (stop counting persistent hard failures). Additional changes:

1. **Separate score-based and failure-based stopping.** Change `_assess_quality` to return a richer result:
   ```python
   @dataclass
   class QualityAssessment:
       status: str          # "pass", "fail", "degraded"
       avg_score: float
       scores: dict[str, float]
       new_hard_failures: int
       persistent_hard_failures: int
       needs_refinement: bool
   ```
   `needs_refinement` is `True` only if `avg_score < threshold` OR `new_hard_failures > max_hard_failures_for_pass`.

2. **Log the decision.** When refinement stops, log whether it was score-based ("avg 8.1 ≥ 6.0") or failure-based ("2 persistent failures, 0 new").

**Files:** `tools/synthesis_tools.py`

**Dependency:** Builds on Issue 2.

---

## Issue 5: Searcher context bloat (108K tokens by final turn)

**Problem:** The searcher accumulates all tool results in `self._messages`. By turn 12, it carries 32 messages and 108K prompt tokens. Most of this is stale search results from early turns that the model no longer needs.

**Plan:**

1. **Add history compaction to `Agent.run()`.** After each tool-result round, if `self._messages` exceeds a token budget (configurable, default 60K chars ≈ ~15K tokens), compact older tool results:
   - Keep the system prompt and context summary intact.
   - Keep the last N assistant+tool exchanges intact (N=2 or 3).
   - Replace older tool result messages with a one-line summary: `"[tool result compacted] pubmed_search returned 10 papers (PMIDs: 12345, 23456, ...)"`.

2. **Add `_compact_tool_result(content: str, tool_name: str) -> str` helper** that extracts a summary from a tool result. For search tools, extract PMID list. For summarize tools, extract PMID + gene count. For other tools, truncate to first 200 chars.

3. **Add `history_compaction_threshold` to Config.** Default: 80000 (chars). Set to 0 to disable.

4. **Preserve the context summary** — it already contains the current pipeline state, so compacted history doesn't lose critical state.

**Files:** `agents/base.py`, `core/config.py`

**Risk:** Medium. The model may lose context about earlier search decisions. Mitigated by keeping the context summary (which includes queries executed) and the last 2-3 exchanges.

---

## Issue 6: Paper summarization is the wall-clock bottleneck

**Problem:** 40 sequential LLM calls taking 1017s (44% of total LLM time). These are completely independent.

**Plan:**

1. **Parallelize `_summarize_papers_batch` in `coordination_tools.py`.** The existing `_fetch_papers_parallel` already uses `ThreadPoolExecutor`. Apply the same pattern to paper summarization.

2. **Use `config.parallel_workers` (currently 4)** as the concurrency limit. Each worker calls `summarize_paper()` independently.

3. **Add error isolation.** If one summarization fails, log the error and continue with the rest. Don't let one paper crash the batch.

4. **Estimated savings:** With 4 workers and 40 papers: ~10 batches of 4 × ~25s each = ~250s vs 1017s sequential. Saves ~770s (12.8 min).

**Files:** `tools/coordination_tools.py`

**Risk:** Low. `summarize_paper` is stateless — it takes a paper dict and returns a `PaperSummary`. The only shared state is the LLM client, which is thread-safe (litellm uses httpx under the hood). Rate limiting is handled by the LLM provider, not by us.

**Note:** Check whether `config.parallel_workers` is already used for paper fetching in `_fetch_papers_parallel`. If so, we may want separate limits for fetch vs summarize, or a shared semaphore.

---

## Issue 8: Second synthesis re-processes everything from scratch

**Problem:** After feedback-driven re-search, `synthesize_disease_profile` re-audits and re-drafts all 40 papers, even though only ~10-15 are new. The second synthesis costs 424s and ~800K tokens.

**Plan:**

1. **Track which papers have been synthesized.** Add `synthesized_pmids: set[str]` to `SynthesisState` in `context.py`. After each synthesis, record the PMIDs that were included.

2. **Incremental synthesis path.** In `synthesize_disease_profile`, if `ctx.synthesis_state.profile` already exists and `synthesized_pmids` is non-empty:
   - Identify new papers: `new_summaries = [s for s in all_summaries if s.pmid not in synthesized_pmids]`
   - If no new papers, skip synthesis entirely and return the existing profile.
   - If new papers exist, run the multi-pass pipeline on *only the new papers* to produce a `delta_profile`.
   - Merge `delta_profile` into the existing profile using `_merge_profiles`.
   - Run critique + refine on the merged result (this still needs all papers for validation, but uses the compact evidence index from Issue 3).

3. **Estimated savings:** Second synthesis drops from ~800K tokens to ~300K (audit+draft only new papers, merge, critique+refine with compact index).

**Files:** `tools/synthesis_tools.py`, `core/context.py`

**Risk:** Medium. The merge step may produce a lower-quality profile than full re-synthesis because the delta profile lacks context from the original papers. Mitigated by the critique+refine pass on the merged result. Should be validated by comparing output quality between full and incremental synthesis on the same corpus.

---

## Implementation Order

Issues are ordered by dependency and impact:

| Phase | Issues | Rationale |
|---|---|---|
| A | 3 (compact critique index) | No dependencies, immediate token savings, easy to validate |
| B | 2 + 4 (hard failure tracking, refine stopping) | Coupled changes, fix the refine loop before adding more complexity |
| C | 1 (genetics schema split) | Schema change, do it before incremental synthesis adds more schema dependencies |
| D | 5 (searcher history compaction) | Independent, medium risk, test in isolation |
| E | 6 (parallel summarization) | Independent, low risk, big wall-clock savings |
| F | 8 (incremental synthesis) | Depends on A (compact index) and C (stable schema), highest complexity |

Phases A-C address profile quality. Phases D-F address performance. Each phase should be a separate commit with its own test run to validate.

---

## Config Changes for Smaller-Scale Iteration

To iterate faster on the core workflow without burning 28 minutes and 2.6M tokens per run:

```python
# In config.py — "fast iteration" defaults
max_papers: int = 15                    # was 40 — enough to trigger map-reduce (>batch_size) or stay single-batch
synthesis_batch_size: int = 10          # was 30 — smaller batches, faster audit/draft
max_feedback_rounds: int = 1            # was 2 — one feedback cycle is enough to test the loop
synthesis_max_internal_passes: int = 1  # was 2 — one refine pass to test the mechanism
max_tool_calls_per_turn: int = 20      # was 30 — tighter safety limit
```

Estimated run time with these settings: ~8-10 minutes, ~600K-800K tokens.

To test map-reduce specifically, set `max_papers=15, synthesis_batch_size=10` (triggers 2 batches).
To test single-batch, set `max_papers=10, synthesis_batch_size=15`.
