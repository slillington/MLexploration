"""Fetch full text for a paper from PMC Open Access.

Usage:
    .venv/bin/python scripts/get_full_text.py [PMID]

Default PMID: 39551787
"""

import sys

from targetsearch.tools.fulltext import fetch_paper_text, pmids_to_pmcids

pmid = sys.argv[1] if len(sys.argv) > 1 else "39551787"

try:
    # Check which papers have full text
    pmcids = pmids_to_pmcids([pmid])
    for p, pmcid in pmcids.items():
        status = f"PMC: {pmcid}" if pmcid else "abstract only"
        print(f"PMID {p}: {status}")

    # Fetch the best available text
    paper = fetch_paper_text(pmid, pmcids.get(pmid))
    print(f"\nSource: {paper['source_type']}")
    print(f"Title: {paper['title']}")
    print(f"Text length: {len(paper['text'])} chars")
    print(f"References found: {len(paper.get('references', []))} PMIDs")
    print(f"\nFirst 500 chars:\n{paper['text'][:500]}")
except Exception as e:
    print(f"Failed to fetch PMID {pmid}: {e}")