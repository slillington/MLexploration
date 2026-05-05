"""Explore Open Targets — disease targets and known drugs.

Usage:
    .venv/bin/python scripts/explore_targets.py "amyotrophic lateral sclerosis"
    .venv/bin/python scripts/explore_targets.py "Crohn disease"
"""

import sys

from targetsearch.tools.targets import opentargets_disease_targets, opentargets_target_drugs

disease = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "amyotrophic lateral sclerosis"

result = opentargets_disease_targets(disease, max_results=10)
print(f"Disease: {result['disease_name']} ({result['disease_id']})")
print(f"Description: {result['description'][:200]}\n")

print("=== Top associated targets ===\n")
for t in result["targets"]:
    scores = t["evidence_scores"]
    print(f"  {t['gene_symbol']:10s} score={t['overall_score']:.3f}  {t['gene_name'][:40]}")
    print(f"             evidence: {scores}")

# Pick the top target and look up its drugs
if result["targets"]:
    top = result["targets"][0]
    print(f"\n=== Drugs targeting {top['gene_symbol']} ===\n")
    try:
        drugs = opentargets_target_drugs(top["ensembl_id"])
        if drugs:
            for d in drugs[:5]:
                print(f"  {d['drug_name']:20s} phase={d['max_phase']}  {d['mechanism'][:50]}")
        else:
            print("  (no drugs found)")
    except Exception as e:
        print(f"  Drug lookup failed: {e}")
