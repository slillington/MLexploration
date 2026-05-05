# Implementation Learnings

Collected from iterative development and test runs on `feat/multi-pass-synthesis`.
Last updated: 2026-04-19, run `4d1623` (NSCLC biologics query).

## Performance & Token Efficiency

### 1. gpt-5-mini rates all `evidence_strength` as "insufficient"

Every paper summarized by gpt-5-mini gets `evidence_strength: "insufficient"` — including full-text clinical trial reports and reviews where the study design is clearly described. The model follows the abstract-only heuristic ("evidence_strength should be insufficient if no methods detail") even when methods detail is present. This degrades the compact evidence index and critique quality.

Options: add few-shot examples to the prompt for gpt-5-mini, or keep `evidence_strength` assessment on gpt-5.4 while using gpt-5-mini for the rest of the summary.

### 2. Review mining is the second-largest token consumer

11 calls, 139K prompt tokens, 48K completion tokens. Each call sends the full review text (~12.7K tokens). Many mined PMIDs are never fetched because the paper budget is exhausted by the time mining completes.

Options: (a) mine only the top-N most relevant reviews by triage score, (b) truncate review text to the references/discussion section, (c) skip mining entirely when paper budget is nearly full.

### 3. Searcher prompt efficiency is 7.2%

188K prompt tokens produce only 13.6K completion tokens across 12 turns. The searcher replays its full message history on every turn. Compaction helps (reduced from 708K in the previous run), but the searcher still accumulates ~19K tokens by its final turn.

The gap-fill searcher starts fresh at 5.6K tokens — this is the right pattern. Consider whether the initial searcher could also benefit from a fresh context after triage (discard raw search results, keep only triage output and fetched PMIDs).

### 4. Feedback agent tool call waste (fixed)

`tool_tags = []` was falsy, so the feedback agent received all 19 tools. It made 4 redundant tool calls at 11.5K tokens each (46K total), producing 14 completion tokens per call. Fixed by changing the falsy check in `base.py`.

## Architecture & Data Flow

### 5. PMID citation loss during map-reduce merge

15 papers are summarized but only 9 PMIDs appear in the final profile. Evidence from 6 papers is lost during the merge step. The merge prompt does not explicitly instruct the LLM to preserve all PMID citations from both input profiles. When consolidating two batch profiles, the model drops citations it considers redundant.

Fix: add explicit instruction to the merge prompt: "preserve all PMID citations from both input profiles."

### 6. `paper_summaries` field in DiseaseProfile is always empty

`DiseaseProfile.paper_summaries: list[PaperSummary]` exists in the schema but synthesis never populates it. Summaries stay in `ctx.paper_state.summaries`. The field is dead weight in the profile JSON. Either populate it for downstream consumers or remove it.

### 7. Ontology and Open Targets results don't flow into ActionContext

`disease_ontology_search` and `opentargets_disease_targets` are stateless leaf tools. Their results are returned to the searcher's message history but never written to `ctx.disease_info` or `ctx.target_state`. As a result, `synthesize_disease_profile` reads empty Open Targets data.

Impact: the synthesis prompt says "Open Targets data" but receives nothing. The LLM synthesizes purely from paper summaries, missing structured target-disease association scores, genetic evidence scores, and known drug data that Open Targets provides.

### 8. Search bookkeeping is partially broken

`queries_executed` is tracked via the base agent's tool dispatch hook. `total_papers_found` stays 0 throughout the run. The context summary reports "0 papers found" even after 15 papers are summarized. This misleads the orchestrator when deciding whether more searching is needed.

## Quality & Correctness

### 9. Gap-fill paper budget exhaustion (fixed)

`max_papers=15` was consumed by the initial pass. Gap-fill search fetched papers that were never summarized. Fixed by splitting the budget: `max_papers_initial=12`, `max_papers_gap_fill=8`.

### 10. Gap-fill searcher mode selection works

The mode selection prompt ("If gaps_to_fill present → Gap-Filling Mode") works correctly. Seq 52 shows 8 direct PubMed queries, no ontology/OT calls, no broad search plan. The searcher correctly skips redundant steps.

### 11. Unmet needs quality is strong

All 10 unmet needs reference specific molecular targets, mechanisms, or biomarker gaps. The biologic-focused query correctly steered the profile toward antibodies, ADCs, and bispecifics. Examples: "Biomarkers that predict benefit from biologics beyond PD-L1 alone", "Effective biologic strategies for immune-cold subsets such as STK11- and/or KEAP1-altered tumors".

### 12. Germline genetics section handles somatic diseases correctly

For NSCLC, `germline_genetics: []` with `germline_note: "No germline associations were reported..."` is correct. The schema split (somatic_genomics vs germline_genetics) prevents the model from mislabeling expression data as eQTL, which was a persistent quality issue before the split.

### 13. Feedback agent redundant calls (fixed)

The feedback agent ran 5 LLM calls instead of 1. Seq 42-45 were identical (11,558 prompt, 14 completion, 1 tool call). Fixed by clearing `tool_tags` and fixing the falsy check.

## Prompt Engineering Patterns

### What worked

- **Explicit mode selection at the top of the prompt.** "If X → follow Mode A. Otherwise → follow Mode B." The searcher correctly follows gap-fill mode when gaps are present.
- **Inline persona descriptions outperform tool-generated personas.** The four feedback perspectives defined inline in the system prompt are more specific than what `create_expert_persona` returns. Removing the tool calls saved tokens and improved output.
- **Numbered extraction rules.** The 7 extraction rules in `summarize_paper` provide clear, auditable constraints. The model follows them more reliably than prose instructions.
- **`[Perspective]` tags in output format.** Adding `[Clinical]`, `[Genetics]`, etc. to the gap output format gives attribution without requiring separate LLM calls per perspective.
- **Compact evidence index for critique.** One line per paper (PMID, title, gene list, finding count) is sufficient for the critique pass. Full paper summaries are unnecessary — the critique scores the profile, not the papers.

### What didn't work

- **"Leave fields empty when data is missing" with gpt-5-mini.** The model defaults `evidence_strength` to "insufficient" rather than leaving it empty. It follows the letter of the abstract-only heuristic but not the spirit.
- **Generic persona instructions.** "You are an expert scientist" produces generic output. The rewrite to "critical evidence reviewer" with explicit principles (separate observations from interpretation, flag cross-disease evidence) was measurably better.
- **Single paragraph for gap-filling mode.** A short "if gaps are provided, focus on those" paragraph buried after a 9-step workflow was ignored. The model followed the dominant numbered workflow. Restructuring into explicit modes with equal visual weight fixed this.

## Prioritized Recommendations

| Priority | Issue | Token Savings | Effort |
|---|---|---|---|
| 1 | Fix `total_papers_found` bookkeeping | 0 (correctness) | Small |
| 2 | Persist OT results into ActionContext | 0 (quality) | Small |
| 3 | Fix PMID citation loss in merge prompt | 0 (quality) | Small |
| 4 | Cap review mining to top-3 reviews | ~80K/run | Small |
| 5 | Evaluate `evidence_strength` on gpt-5.4 vs gpt-5-mini | 0 (quality) | Test run |
| 6 | Remove or populate `paper_summaries` in DiseaseProfile | 0 (cleanup) | Small |
| 7 | Enforce feedback round limit in code | 0 (safety) | Small |
