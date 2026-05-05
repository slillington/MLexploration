"""Search result triage — LLM-based relevance ranking before full-text fetch.

Sits between pubmed_search/semantic_scholar_search and
fetch_and_classify_papers. Takes accumulated search results (title,
abstract, metadata) and the disease context, returns a ranked and
filtered list of PMIDs worth fetching.

This is a leaf tool (no ActionContext). The SearcherAgent calls it
after accumulating search results and before calling
fetch_and_classify_papers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from targetsearch.core.llm import llm_text, parse_json_response
from targetsearch.core.registry import registry

log = logging.getLogger(__name__)

# Evidence bucket labels used in the triage output
EVIDENCE_BUCKETS = [
    "disease_biology",
    "human_genetics",
    "preclinical_therapeutic",
    "clinical",
]


def _build_triage_prompt(disease_area: str, search_context: str) -> str:
    """Build the system prompt for triage."""
    return f"""\
You are a biomedical literature triage specialist. You are evaluating \
search results for drug target discovery in {disease_area}.

## Your task

You will receive a list of papers (title, abstract, year, journal, PMID). \
For each paper, decide:

1. **Relevance** (0-10): How relevant is this paper to identifying or \
validating drug targets for {disease_area}? Score 0 for completely \
irrelevant, 10 for directly actionable target evidence.

2. **Evidence bucket**: Which category does this paper primarily fill?
   - disease_biology: mechanisms, pathways, cell states, biomarkers
   - human_genetics: GWAS, rare variants, Mendelian, eQTL, somatic
   - preclinical_therapeutic: target perturbation, drug studies, animal models
   - clinical: trials, approved therapies, clinical outcomes, resistance

3. **Redundancy**: Flag if this paper appears to cover the same ground \
as another paper in the list (same study, same finding, overlapping scope).

{search_context}

## Output format

Return a JSON object:
{{
  "triaged_papers": [
    {{
      "pmid": "12345678",
      "relevance_score": 8,
      "evidence_bucket": "human_genetics",
      "rationale": "GWAS meta-analysis identifying 3 novel loci for [disease]",
      "redundant_with": null
    }}
  ],
  "bucket_coverage": {{
    "disease_biology": 3,
    "human_genetics": 2,
    "preclinical_therapeutic": 1,
    "clinical": 0
  }},
  "recommendation": "Brief note on which buckets need more search coverage"
}}

Sort triaged_papers by relevance_score descending. Include ALL papers \
from the input — do not silently drop any. Output ONLY the JSON object."""


def _format_papers_for_triage(papers: list[dict[str, Any]]) -> str:
    """Format paper metadata compactly for the triage LLM call."""
    lines = []
    for i, p in enumerate(papers, 1):
        pmid = p.get("pmid", "?")
        title = p.get("title", "(no title)")
        year = p.get("year", "?")
        journal = p.get("journal", "")
        abstract = p.get("abstract", "")

        # Truncate long abstracts to save tokens
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."

        entry = f"[{i}] PMID: {pmid} | Year: {year} | Journal: {journal}\n"
        entry += f"    Title: {title}\n"
        if abstract:
            entry += f"    Abstract: {abstract}\n"
        else:
            entry += "    Abstract: (not available)\n"
        lines.append(entry)

    return "\n".join(lines)


def _parse_triage_result(
    raw: str, input_pmids: set[str]
) -> dict[str, Any]:
    """Parse triage LLM output, ensuring all input PMIDs are accounted for.

    If the LLM drops papers, they're added back with a default low score
    so nothing is silently lost.
    """
    try:
        data = parse_json_response(raw)
    except Exception as e:
        log.error("Failed to parse triage result: %s", e)
        # Return all papers with default scores on parse failure
        return {
            "triaged_papers": [
                {
                    "pmid": pmid,
                    "relevance_score": 5,
                    "evidence_bucket": "disease_biology",
                    "rationale": "Triage parse failed — included by default",
                    "redundant_with": None,
                }
                for pmid in input_pmids
            ],
            "bucket_coverage": {b: 0 for b in EVIDENCE_BUCKETS},
            "recommendation": "Triage failed — all papers included by default",
        }

    triaged = data.get("triaged_papers", [])

    # Normalize PMIDs to strings
    for paper in triaged:
        paper["pmid"] = str(paper.get("pmid", ""))
        paper["relevance_score"] = _clamp_score(paper.get("relevance_score", 5))
        paper["evidence_bucket"] = _normalize_bucket(
            paper.get("evidence_bucket", "disease_biology")
        )
        paper.setdefault("rationale", "")
        paper.setdefault("redundant_with", None)

    # Check for dropped papers and add them back
    triaged_pmids = {p["pmid"] for p in triaged}
    missing = input_pmids - triaged_pmids
    if missing:
        log.warning(
            "Triage LLM dropped %d papers — adding back with default score",
            len(missing),
        )
        for pmid in missing:
            triaged.append({
                "pmid": pmid,
                "relevance_score": 5,
                "evidence_bucket": "disease_biology",
                "rationale": "Not evaluated by triage — included by default",
                "redundant_with": None,
            })

    # Re-sort by relevance score descending
    triaged.sort(key=lambda p: p["relevance_score"], reverse=True)

    # Normalize bucket coverage
    bucket_coverage = data.get("bucket_coverage", {})
    for bucket in EVIDENCE_BUCKETS:
        bucket_coverage.setdefault(bucket, 0)

    return {
        "triaged_papers": triaged,
        "bucket_coverage": bucket_coverage,
        "recommendation": data.get("recommendation", ""),
    }


def _clamp_score(score: Any) -> int:
    """Clamp a relevance score to 0-10."""
    try:
        return max(0, min(10, int(score)))
    except (TypeError, ValueError):
        return 5


def _normalize_bucket(bucket: str) -> str:
    """Normalize an evidence bucket name to one of the valid values."""
    bucket = bucket.strip().lower().replace(" ", "_").replace("-", "_")
    if bucket in EVIDENCE_BUCKETS:
        return bucket
    # Fuzzy match
    for valid in EVIDENCE_BUCKETS:
        if bucket in valid or valid in bucket:
            return valid
    return "disease_biology"  # default


def _normalize_papers(papers: list[Any]) -> list[dict[str, Any]]:
    """Normalize paper inputs to dicts.

    The LLM may pass papers as dicts, JSON strings, or bare PMID strings.
    JSON strings are parsed into dicts; bare PMIDs become minimal dicts.
    """
    normalized: list[dict[str, Any]] = []
    for p in papers:
        if isinstance(p, dict):
            normalized.append(p)
        elif isinstance(p, str):
            stripped = p.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        normalized.append(parsed)
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            normalized.append({"pmid": p, "title": "", "abstract": "", "year": None, "journal": ""})
        else:
            log.warning("triage_search_results: skipping non-dict/str item: %s", type(p))
    return normalized


@registry.tool(
    description=(
        "Triage accumulated search results by relevance to drug target "
        "discovery. Takes a list of papers (with title, abstract, metadata) "
        "and returns them ranked by relevance score (0-10), categorized by "
        "evidence bucket, and flagged for redundancy. Use this after "
        "collecting search results and before calling fetch_and_classify_papers "
        "to focus the pipeline on the most relevant papers."
    ),
    tags=["literature"],
    params={
        "papers": (
            "List of paper dicts from pubmed_search or semantic_scholar_search. "
            "Each dict should have: pmid, title, abstract, year, journal."
        ),
        "disease_area": "Disease being investigated",
        "search_context": (
            "Brief description of what evidence you're looking for "
            "(e.g. 'initial broad search' or 'filling gaps in genetic evidence')"
        ),
        "min_relevance": (
            "Minimum relevance score (0-10) to include in the recommended "
            "PMID list. Default 4."
        ),
        "max_recommended": (
            "Maximum number of papers to recommend, regardless of how many "
            "pass the relevance threshold. Use to respect paper budget. "
            "Default 0 (no limit)."
        ),
    },
    returns=(
        "Dict with: recommended_pmids (filtered list), triaged_papers "
        "(full ranked list with scores), bucket_coverage (counts per bucket), "
        "recommendation (coverage assessment)"
    ),
)
def triage_search_results(
    papers: list[dict[str, Any]],
    disease_area: str,
    search_context: str = "initial broad search",
    min_relevance: int = 4,
    max_recommended: int = 0,
) -> dict[str, Any]:
    """Rank and filter search results by relevance before full-text fetch.

    Leaf tool — no ActionContext. Makes a single LLM call over titles
    and abstracts (~100-200 tokens per paper).
    """
    if not papers:
        return {
            "recommended_pmids": [],
            "triaged_papers": [],
            "bucket_coverage": {b: 0 for b in EVIDENCE_BUCKETS},
            "recommendation": "No papers to triage.",
            "total_input": 0,
            "total_recommended": 0,
            "filtered_out": 0,
        }

    papers = _normalize_papers(papers)

    # If all inputs were bare PMIDs (no abstracts to triage on), skip the
    # LLM call and return them all as recommended
    if all(not p.get("abstract") and not p.get("title") for p in papers):
        pmids = list(dict.fromkeys(str(p.get("pmid", "")) for p in papers if p.get("pmid")))
        if max_recommended > 0 and len(pmids) > max_recommended:
            pmids = pmids[:max_recommended]
        log.info("triage_search_results: all inputs are bare PMIDs, skipping LLM triage")
        return {
            "recommended_pmids": pmids,
            "triaged_papers": [
                {
                    "pmid": pmid,
                    "relevance_score": 5,
                    "evidence_bucket": "disease_biology",
                    "rationale": "No metadata available for triage — included by default",
                    "redundant_with": None,
                }
                for pmid in pmids
            ],
            "bucket_coverage": {b: 0 for b in EVIDENCE_BUCKETS},
            "recommendation": "No abstracts available — all papers included without ranking.",
            "total_input": len(pmids),
            "total_recommended": len(pmids),
            "filtered_out": 0,
        }

    # Deduplicate by PMID
    seen: set[str] = set()
    unique_papers: list[dict[str, Any]] = []
    for p in papers:
        pmid = str(p.get("pmid", ""))
        if pmid and pmid not in seen:
            seen.add(pmid)
            unique_papers.append(p)
        elif not pmid:
            unique_papers.append(p)

    input_pmids = {str(p.get("pmid", "")) for p in unique_papers if p.get("pmid")}

    # Build prompt and format papers
    context_note = f"## Search context\n\n{search_context}" if search_context else ""
    system_prompt = _build_triage_prompt(disease_area, context_note)
    papers_text = _format_papers_for_triage(unique_papers)

    user_message = (
        f"Triage these {len(unique_papers)} papers for drug target "
        f"discovery relevance in {disease_area}:\n\n{papers_text}"
    )

    log.info(
        "triage_search_results: %d papers for %s (%s)",
        len(unique_papers),
        disease_area,
        search_context,
    )

    raw = llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        caller="triage_search_results",
    )

    result = _parse_triage_result(raw, input_pmids)

    # Filter by minimum relevance score
    min_relevance = _clamp_score(min_relevance)
    recommended = [
        p for p in result["triaged_papers"]
        if p["relevance_score"] >= min_relevance
    ]
    filtered_out = len(result["triaged_papers"]) - len(recommended)

    recommended_pmids = [p["pmid"] for p in recommended if p.get("pmid")]

    # Apply max_recommended cap (highest-relevance papers kept)
    if max_recommended > 0 and len(recommended_pmids) > max_recommended:
        log.info(
            "triage_search_results: capping %d recommended to %d (budget)",
            len(recommended_pmids), max_recommended,
        )
        recommended_pmids = recommended_pmids[:max_recommended]

    log.info(
        "triage_search_results: %d/%d papers above threshold (min=%d), "
        "%d filtered out",
        len(recommended_pmids),
        len(unique_papers),
        min_relevance,
        filtered_out,
    )

    return {
        "recommended_pmids": recommended_pmids,
        "triaged_papers": result["triaged_papers"],
        "bucket_coverage": result["bucket_coverage"],
        "recommendation": result["recommendation"],
        "total_input": len(unique_papers),
        "total_recommended": len(recommended_pmids),
        "filtered_out": filtered_out,
    }
