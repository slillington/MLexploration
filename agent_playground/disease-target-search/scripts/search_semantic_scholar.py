"""Search Semantic Scholar for highly-cited papers.

Usage:
    .venv/bin/python scripts/search_semantic_scholar.py "CRISPR gene therapy sickle cell"
    .venv/bin/python scripts/search_semantic_scholar.py "PD-1 checkpoint inhibitor melanoma"
"""

import sys

from targetsearch.tools.literature import semantic_scholar_search

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "CRISPR gene therapy sickle cell"

print(f"Query: {query}\n")

try:
    results = semantic_scholar_search(query, max_results=5)
    for r in results:
        print(f"{r['title'][:70]}")
        print(f"  Year: {r['year']}  Citations: {r['citation_count']}  PMID: {r['pmid']}")
        abstract = r["abstract"][:120] + "..." if r["abstract"] else "(no abstract)"
        print(f"  {abstract}\n")

    print(f"{len(results)} results returned")
except Exception as e:
    print(f"Search failed: {e}")
