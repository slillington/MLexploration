"""Extract primary source PMIDs from a review article.

Usage:
    .venv/bin/python scripts/mine_review.py 33359599 "idiopathic pulmonary fibrosis"
    .venv/bin/python scripts/mine_review.py 39551787 "hematological cancers"
"""

import sys

from targetsearch.tools.fulltext import fetch_paper_text, pmids_to_pmcids
from targetsearch.tools.paper_tools import mine_review_references

if len(sys.argv) < 2:
    print("Usage: .venv/bin/python scripts/mine_review.py <PMID> [disease_area]")
    sys.exit(1)

pmid = sys.argv[1]
disease_area = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "drug target discovery"

try:
    # Fetch the paper
    print(f"Fetching PMID {pmid}...")
    pmcids = pmids_to_pmcids([pmid])
    paper = fetch_paper_text(pmid, pmcids.get(pmid))

    print(f"Source: {paper['source_type']}")
    print(f"Title: {paper['title']}")
    print(f"Text length: {len(paper['text'])} chars")
    print(f"Structured refs from XML: {len(paper.get('references', []))} PMIDs\n")

    if not paper["text"]:
        print("No text available for this paper.")
        sys.exit(1)

    if paper["source_type"] != "full_text":
        print("Warning: No full text available — review mining works best with full text.")
        print("         It can still run on the abstract but will find fewer references.\n")

    # Mine the review
    print(f"Mining with disease context: {disease_area}...\n")
    result = mine_review_references(
        review_text=paper["text"],
        disease_area=disease_area,
        structured_refs=paper.get("references", []),
    )

    cited_papers = result.get("cited_papers", [])
    high_priority = [c for c in cited_papers if c.get("pmid") and c.get("priority") == "high"]
    all_pmids = [c["pmid"] for c in cited_papers if c.get("pmid")]

    print(f"{'='*60}")
    print(f"Review: {result.get('review_title', '(unknown)')}")
    print(f"Cited papers found: {len(cited_papers)}")
    print(f"High priority: {len(high_priority)}")
    print(f"All PMIDs: {len(all_pmids)}\n")

    for cp in cited_papers:
        pmid_str = str(cp.get("pmid", "None"))
        print(f"  [{cp.get('priority', '?'):6s}] PMID {pmid_str:>10s}: {cp.get('description', '')[:70]}")
        if cp.get("relevance"):
            print(f"           Relevance: {cp['relevance'][:70]}")
except Exception as e:
    print(f"Failed: {e}")
