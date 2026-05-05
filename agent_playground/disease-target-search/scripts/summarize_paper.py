"""Summarize a single paper with the summarize_paper tool.

Usage:
    .venv/bin/python scripts/summarize_paper.py 33359599 "idiopathic pulmonary fibrosis"
    .venv/bin/python scripts/summarize_paper.py 39551787 "hematological cancers"
"""

import sys

from targetsearch.schemas.paper import PaperSummary
from targetsearch.tools.fulltext import fetch_paper_text, pmids_to_pmcids
from targetsearch.tools.paper_tools import summarize_paper

if len(sys.argv) < 2:
    print("Usage: .venv/bin/python scripts/summarize_paper.py <PMID> [disease_area]")
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
    print(f"Text length: {len(paper['text'])} chars\n")

    if not paper["text"]:
        print("No text available for this paper.")
        sys.exit(1)

    # Summarize
    print(f"Summarizing with disease context: {disease_area}...\n")
    result = summarize_paper(
        paper_text=paper["text"],
        disease_area=disease_area,
        metadata=paper,
    )
    summary = PaperSummary.model_validate(result)

    print(f"{'='*60}")
    print(f"Title: {summary.title}")
    print(f"Type: {summary.paper_type}")
    print(f"Objective: {summary.objective}")
    print(f"\nKey findings:")
    for kf in summary.key_findings:
        print(f"  - {kf.finding}")
        print(f"    Evidence: {kf.evidence_type}, Model: {kf.model_system}")
        print(f"    Effect: {kf.effect_size}")
        print(f"    Genes: {kf.genes_proteins}")
    print(f"\nMethods: {summary.methods_summary}")
    print(f"Limitations: {summary.limitations}")
    print(f"Target relevance: {summary.target_relevance}")
    print(f"All genes/pathways: {summary.genes_pathways_mentioned}")
except Exception as e:
    print(f"Failed: {e}")
