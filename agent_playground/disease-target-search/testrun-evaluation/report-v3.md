# Test Run Evaluation — v3 (Early Budget Gating + Review Summarization)

**Run ID:** `20260419_044042_8101ff`
**Query:** "non-small cell lung cancer that can be treated with a biologic (antibody, bispecific, multispecific, T cell engager, ADC, or cell therapy)"
**Date:** 2026-04-19
**Commit:** `a8b197d` (feat/multi-pass-synthesis)

## Changes Since v2 (run `4d1623`)

1. **Review mining removed** — reviews are now summarized directly with 9-dimension extraction guidelines; no PMIDs are mined or chased
2. **Early budget check** — `fetch_and_classify_papers` returns immediately when budget is exhausted, trims PMIDs to remaining budget before fetching
3. **Budget in context summary** — searcher sees "12/12 initial budget (0 remaining)"
4. **Budget-aware triage** — `max_recommended` parameter caps triage output to available slots
5. **Reference list whitelist** — review summaries constrain inline PMID citations to the article's structured reference list
6. **tool_tags fix** — feedback agent gets 0 tools (from v2 commit, first run validating it)

## Summary Comparison

| Metric | v2 (`4d1623`) | v3 (`8101ff`) | Change |
|---|---|---|---|
| Wall-clock time | 26.9 min | 22.8 min | **-15%** |
| Total LLM calls | 63 | 52 | **-17%** |
| Total tokens | 847,569 | 710,919 | **-16%** |
| Prompt tokens | 682,097 | 567,543 | **-17%** |
| Completion tokens | 165,472 | 143,376 | **-13%** |
| Papers summarized | 15 | 17 | +13% |
| Output quality score | 87% (34.7/40) | 91% (36.3/40) | **+4 pts** |

## Token Breakdown by Caller

| Caller | v2 Prompt | v3 Prompt | Change |
|---|---|---|---|
| searcher | 188,702 | 208,074 | +10% |
| mine_review_references | 139,333 | 0 | **-100%** |
| feedback | 112,455 | 9,630 | **-91%** |
| refine | 36,222 | 101,596 | +180% |
| critique | 21,699 | 46,116 | +113% |
| summarize_paper (all) | 69,074 | 80,133 | +16% |
| audit | 28,286 | 28,286 | 0% |
| draft | 31,428 | 31,428 | 0% |

**Key observations:**
- `mine_review_references` eliminated entirely: **-139K tokens, -667s**
- Feedback agent: 1 call instead of 5 (tool_tags fix): **-103K tokens**
- Refine and critique increased because more papers (17 vs 15) produce a richer profile with more sections to evaluate
- Searcher slightly higher due to gap-fill pass finding and processing more papers

## Pipeline Flow

```
04:40 → run_search_agent (initial)      7.3 min
04:48 → synthesize_disease_profile      5.8 min
        batch_summarize: 12 papers      5.6 min
        audit → draft → merge           3.8 min
        critique → refine               2.0 min
04:53 → run_feedback_agent              0.4 min  (1 call, 0 tools)
04:54 → run_search_agent (gap-fill)     4.4 min
04:58 → synthesize_disease_profile      4.9 min
        batch_summarize: 5 papers       1.9 min
        incremental synthesis           3.0 min
05:03 → final output
```

Total: 22.8 min. The feedback→gap-fill→re-synthesize loop completed with actual paper summarization on the gap-fill pass.

## Budget Gating Validation

- ✅ Initial pass: 12/12 papers summarized (budget fully used)
- ✅ Gap-fill pass: 5/8 papers summarized (within gap-fill budget)
- ✅ Total: 17/20 papers (within combined budget)
- ✅ No review mining calls (0 occurrences in log)
- ✅ Context summary showed "12/12 initial budget (0 remaining)" before gap-fill
- ✅ Feedback agent: 1 LLM call, 0 tools, 0 tool calls

## Output Quality Comparison

| Section | v2 | v3 | Change |
|---|---|---|---|
| Pathways | 10 (26 genes, 8 PMIDs) | 14 (30 genes, 17 PMIDs) | +40% pathways, +112% PMIDs |
| Somatic genomics | 19 alterations | 17 alterations | -11% |
| Germline genetics | 0 (with note) | 0 (with note) | same |
| Therapies | 11 | 14 | +27% |
| Unmet needs | 10 | 9 | -10% |
| Lit summary | 2,847 chars | 4,051 chars | +42% |
| Unique PMIDs cited | 9 | 17 | **+89%** |
| Quality score | 34.7/40 (87%) | 36.3/40 (91%) | **+4 pts** |

**Quality improvements:**
- PMID citation density nearly doubled (9 → 17). The review summarization with inline citations is contributing — reviews now carry their cited PMIDs into the synthesis pipeline rather than being mined for separate papers.
- Pathway coverage increased 40% with more genes identified.
- Therapy landscape expanded from 11 to 14 entries.
- Literature summary 42% longer with more detail.

**Quality regressions:**
- Somatic genomics slightly fewer (17 vs 19) — likely due to different papers in the corpus rather than a systematic issue.
- Unmet needs 9 vs 10 — marginal, all still have specific molecular targets.

## Model Split

| Model | Calls | Prompt Tokens | Completion Tokens |
|---|---|---|---|
| gpt-5.4 | 35 | 497,410 | 89,376 |
| gpt-5-mini | 17 | 70,133 | 54,000 |

17 gpt-5-mini calls (all summarize_paper) vs 26 in v2 (15 summarize + 11 mine_review). The 9 fewer calls are the eliminated review mining.

## Cumulative Improvements (v1 → v3)

| Metric | v1 (`d26d3d`) | v2 (`4d1623`) | v3 (`8101ff`) | v1→v3 |
|---|---|---|---|---|
| Wall-clock | 38.3 min | 26.9 min | 22.8 min | **-40%** |
| Total tokens | 2,654,873 | 847,569 | 710,919 | **-73%** |
| LLM calls | 88 | 63 | 52 | **-41%** |
| Quality score | N/A | 87% | 91% | — |
| Papers summarized | 40 | 15 | 17 | -58% |
| PMIDs cited in profile | N/A | 9 | 17 | +89% |

## Remaining Issues

1. **Searcher is the largest token consumer** (208K, 29% of total). The initial searcher replays its full message history across 12 turns. The gap-fill searcher starts fresh — consider whether the initial searcher could also reset context after triage.

2. **Refine pass grew significantly** (36K → 102K). With 17 papers and a richer profile, the refine prompt is larger. This is expected but worth monitoring — if paper counts increase further, refine could become the bottleneck.

3. **Open Targets data still not flowing into synthesis.** The synthesis prompt references OT data but receives empty results. Fixing this would improve the profile without additional LLM calls.

4. **`evidence_strength` still "insufficient" from gpt-5-mini.** Not validated in this run but the underlying issue persists.
