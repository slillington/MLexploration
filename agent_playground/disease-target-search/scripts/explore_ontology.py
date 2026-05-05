"""Explore disease and gene ontologies.

Usage:
    .venv/bin/python scripts/explore_ontology.py "Crohn disease"
    .venv/bin/python scripts/explore_ontology.py "glioblastoma"
"""

import sys

from targetsearch.tools.ontology import disease_ontology_search, gene_ontology_lookup

disease = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Crohn disease"

print(f"=== Disease ontology: {disease} ===\n")
results = disease_ontology_search(disease, max_results=3)
for r in results:
    print(f"  {r['id']}: {r['label']}")
    print(f"    {r['description'][:120]}")
    if r["synonyms"]:
        print(f"    Synonyms: {r['synonyms'][:3]}")
    if r["xrefs"]:
        print(f"    Cross-refs: {r['xrefs'][:3]}")
    print()

# Look up GO terms for a well-known target
# Pick a gene relevant to the disease for demonstration
gene = "NOD2" if "crohn" in disease.lower() else "EGFR"
print(f"=== GO terms for {gene} ===\n")
try:
    go = gene_ontology_lookup(gene)
    if go["go_terms"]:
        for term in go["go_terms"][:10]:
            print(f"  {term['id']:15s} [{term['aspect']:25s}] {term['name']}")
    else:
        print(f"  No GO terms found for {gene}")
except Exception as e:
    print(f"  gene_ontology_lookup failed: {e}")
    print(f"  (Known issue: QuickGO requires UniProt accessions, not gene symbols)")
