"""Coordination tools — manage state and orchestrate leaf tools.

These tools accept ActionContext (auto-injected) and encapsulate the
deterministic parallel logic that was in paper_pipeline.py. The
SearcherAgent calls them as single tool invocations.

Internal helpers (classification, rate-limited fetch) are plain Python
functions, not registered tools.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from targetsearch.core.config import config
from targetsearch.core.context import ActionContext
from targetsearch.core.registry import registry
from targetsearch.schemas.paper import PaperSummary
from targetsearch.tools.fulltext import (
    fetch_paper_text,
    pmids_to_pmcids,
    pubmed_fetch_by_pmids,
)
from targetsearch.tools.paper_tools import summarize_paper

log = logging.getLogger(__name__)

# Review-indicating PubMed publication types
_REVIEW_PUB_TYPES = {
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Practice Guideline",
    "Guideline",
}


# ── Internal helpers (not registered tools) ────────────────────────────


def _classify_papers(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split papers into reviews and primary articles based on pub_types."""
    reviews = []
    primaries = []
    for paper in papers:
        pub_types = set(paper.get("pub_types", []))
        if pub_types & _REVIEW_PUB_TYPES:
            reviews.append(paper)
        else:
            primaries.append(paper)
    return reviews, primaries


def _fetch_papers_parallel(
    pmids: list[str],
    pmcid_map: dict[str, str | None],
    workers: int,
    meta_by_pmid: dict[str, dict] | None = None,
) -> list[dict]:
    """Fetch paper text for a list of PMIDs with rate limiting.

    NCBI allows 3 req/s without an API key, 10 req/s with one.
    Uses a semaphore + minimum delay to stay under the limit.

    If *meta_by_pmid* is provided, pre-fetched metadata is forwarded to
    ``fetch_paper_text`` so it can skip the per-paper PubMed efetch call.
    """
    results: list[dict] = []
    semaphore = threading.Semaphore(workers)
    _last_request_lock = threading.Lock()
    _last_request_time = [0.0]
    min_delay = 0.35  # ~3 req/s without API key

    def _fetch(pmid: str) -> dict:
        with semaphore:
            with _last_request_lock:
                now = time.time()
                wait = min_delay - (now - _last_request_time[0])
                if wait > 0:
                    time.sleep(wait)
                _last_request_time[0] = time.time()
            pre = meta_by_pmid.get(pmid) if meta_by_pmid else None
            return fetch_paper_text(pmid, pmcid_map.get(pmid), metadata=pre)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, pmid): pmid for pmid in pmids}
        for future in as_completed(futures):
            pmid = futures[future]
            try:
                result = future.result()
                if result.get("text"):
                    results.append(result)
            except Exception as e:
                log.warning("Failed to fetch PMID %s: %s", pmid, e)

    return results


def _run_summarize_parallel(
    papers: list[dict],
    disease_area: str,
    workers: int,
) -> list[PaperSummary]:
    """Run summarize_paper on each paper in parallel."""
    summaries: list[PaperSummary] = []
    semaphore = threading.Semaphore(workers)

    def _summarize(paper_data: dict) -> dict[str, Any]:
        with semaphore:
            return summarize_paper(
                paper_text=paper_data["text"],
                disease_area=disease_area,
                metadata={
                    "pmid": paper_data.get("pmid"),
                    "doi": paper_data.get("doi"),
                    "title": paper_data.get("title", ""),
                    "authors": paper_data.get("authors", []),
                    "journal": paper_data.get("journal", ""),
                    "year": paper_data.get("year"),
                    "source_type": paper_data.get("source_type", "abstract"),
                },
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_summarize, paper): paper.get("pmid", "?")
            for paper in papers
        }
        for future in as_completed(futures):
            pmid = futures[future]
            try:
                result = future.result()
                summaries.append(PaperSummary.model_validate(result))
            except Exception as e:
                log.error("summarize_paper failed for PMID %s: %s", pmid, e)

    return summaries


# ── Registered coordination tools ──────────────────────────────────────


def _remaining_paper_budget(ctx: ActionContext) -> tuple[int, str]:
    """Return (remaining_slots, phase_label) for the current pass."""
    already_count = len(ctx.paper_state.summaries)
    if ctx.synthesis_state.feedback_rounds == 0:
        remaining = config.max_papers_initial - already_count
        label = "initial"
    else:
        gap_fill_used = max(0, already_count - config.max_papers_initial)
        remaining = config.max_papers_gap_fill - gap_fill_used
        label = "gap-fill"
    return max(0, remaining), label


@registry.tool(
    description=(
        "Fetch metadata and full text for a list of PMIDs. Classifies papers "
        "as review vs primary, resolves PMC IDs, fetches full text where "
        "available. Updates ActionContext with fetched paper data."
    ),
    tags=["coordination"],
    params={
        "pmids": "List of PubMed IDs to fetch",
    },
    returns="Summary of what was fetched (counts by type)",
)
def fetch_and_classify_papers(
    pmids: list[str],
    ctx: ActionContext,
) -> str:
    """Fetch metadata, classify, and retrieve full text for papers.

    Updates ctx.paper_state and ctx.search_state.
    """
    if not pmids:
        return "No PMIDs provided."

    workers = config.parallel_workers

    # Early budget check — don't fetch papers we can't summarize
    remaining, phase_label = _remaining_paper_budget(ctx)
    if remaining <= 0:
        already = len(ctx.paper_state.summaries)
        return (
            f"Paper budget exhausted for {phase_label} pass "
            f"({already} summarized). Skipping fetch — proceed to synthesis."
        )

    # Deduplicate against already-collected PMIDs
    existing = set(ctx.search_state.pmids_collected)
    new_pmids = [p for p in pmids if p not in existing]
    if not new_pmids:
        return f"All {len(pmids)} PMIDs already collected."

    # Trim to remaining budget
    if len(new_pmids) > remaining:
        log.info(
            "fetch_and_classify_papers: trimming %d PMIDs to %d (%s budget)",
            len(new_pmids), remaining, phase_label,
        )
        new_pmids = new_pmids[:remaining]

    # Fetch metadata to get pub_types for classification
    log.info("fetch_and_classify_papers: fetching metadata for %d PMIDs", len(new_pmids))
    metadata_list = pubmed_fetch_by_pmids(new_pmids)
    meta_by_pmid = {m["pmid"]: m for m in metadata_list}

    # Classify
    reviews, primaries = _classify_papers(metadata_list)

    # Resolve PMC IDs
    pmcid_map = pmids_to_pmcids(new_pmids)
    n_pmc = sum(1 for v in pmcid_map.values() if v)

    # Fetch full text in parallel, reusing already-fetched metadata
    fetched_papers = _fetch_papers_parallel(
        new_pmids, pmcid_map, workers, meta_by_pmid=meta_by_pmid,
    )

    # Update context
    ctx.search_state.pmids_collected.extend(new_pmids)
    n_full = sum(1 for p in fetched_papers if p.get("source_type") == "full_text")
    n_abstract = sum(1 for p in fetched_papers if p.get("source_type") == "abstract")
    ctx.paper_state.papers_fetched += len(fetched_papers)
    ctx.paper_state.papers_with_full_text += n_full
    ctx.paper_state.papers_abstract_only += n_abstract

    # Store fetched paper data in properties for batch_summarize_papers
    existing_fetched = ctx.properties.get("fetched_papers", [])
    existing_fetched.extend(fetched_papers)
    ctx.properties["fetched_papers"] = existing_fetched

    summary = (
        f"Fetched {len(fetched_papers)} papers: "
        f"{len(reviews)} reviews, {len(primaries)} primary. "
        f"{n_full} full-text, {n_abstract} abstract-only. "
        f"{n_pmc} had PMC access."
    )
    log.info("fetch_and_classify_papers: %s", summary)
    return summary


@registry.tool(
    description=(
        "Summarize all fetched papers in parallel. Reviews are summarized "
        "with review-specific extraction guidelines that preserve inline "
        "PMID citations. Updates ActionContext with PaperSummary objects."
    ),
    tags=["coordination"],
    params={
        "disease_area": "Disease being investigated",
    },
    returns="Summary of papers summarized",
)
def batch_summarize_papers(
    disease_area: str,
    ctx: ActionContext,
) -> str:
    """Summarize all fetched papers in parallel.

    Reviews are summarized using review-specific extraction guidelines
    rather than mined for cited PMIDs. This produces high-density
    evidence summaries without inflating the paper count.

    Reads fetched paper data from ctx.properties['fetched_papers'].
    Writes PaperSummary objects to ctx.paper_state.summaries.
    """
    workers = config.parallel_workers

    fetched_papers = ctx.properties.get("fetched_papers", [])
    if not fetched_papers:
        return "No fetched papers to summarize."

    # Identify which papers haven't been summarized yet
    already_summarized = {ps.pmid for ps in ctx.paper_state.summaries}
    papers_to_process = [
        p for p in fetched_papers
        if p.get("pmid") not in already_summarized and p.get("text")
    ]

    if not papers_to_process:
        return "All fetched papers already summarized."

    # Enforce paper budget — split between initial and gap-fill passes.
    already_count = len(ctx.paper_state.summaries)
    if ctx.synthesis_state.feedback_rounds == 0:
        phase_budget = config.max_papers_initial
        phase_used = already_count
        phase_label = "initial"
    else:
        phase_budget = config.max_papers_gap_fill
        phase_used = max(0, already_count - config.max_papers_initial)
        phase_label = "gap-fill"

    remaining_budget = max(0, phase_budget - phase_used)

    if len(papers_to_process) > remaining_budget:
        log.info(
            "batch_summarize_papers: trimming from %d to %d "
            "(%s budget %d, used %d)",
            len(papers_to_process),
            remaining_budget,
            phase_label,
            phase_budget,
            phase_used,
        )
        papers_to_process = papers_to_process[:remaining_budget]

    if not papers_to_process:
        return (
            f"Paper budget reached for {phase_label} pass "
            f"({phase_used}/{phase_budget} summarized). "
            "No additional papers will be processed."
        )

    # --- Summarize in parallel ---
    log.info(
        "batch_summarize_papers: summarizing %d papers with %d workers",
        len(papers_to_process),
        workers,
    )
    new_summaries = _run_summarize_parallel(papers_to_process, disease_area, workers)

    # Write to context
    ctx.paper_state.summaries.extend(new_summaries)

    # Clear processed papers from the staging area
    processed_pmids = {p.get("pmid") for p in papers_to_process}
    ctx.properties["fetched_papers"] = [
        p for p in fetched_papers if p.get("pmid") not in processed_pmids
    ]

    summary = (
        f"Summarized {len(new_summaries)} papers. "
        f"Total summaries: {len(ctx.paper_state.summaries)}."
    )
    log.info("batch_summarize_papers: %s", summary)
    return summary
