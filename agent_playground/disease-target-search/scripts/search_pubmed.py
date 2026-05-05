"""Search PubMed with a custom query.

Usage:
    .venv/bin/python scripts/search_pubmed.py "KRAS lung cancer drug resistance"
    .venv/bin/python scripts/search_pubmed.py "TGF-beta fibrosis review"
"""

import sys

from targetsearch.tools.literature import pubmed_search

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "KRAS[MeSH] AND lung neoplasms[MeSH] AND drug resistance"

print(f"Query: {query}\n")

try:
    results = pubmed_search(query, max_results=5)
    for r in results:
        print(f"PMID {r['pmid']} ({r['year']}): {r['title'][:80]}")
        print(f"  Authors: {', '.join(r['authors'][:3])}")
        print(f"  Abstract: {r['abstract'][:150]}...\n")

    print(f"{len(results)} results returned")
except Exception as e:
    print(f"Search failed: {e}")
