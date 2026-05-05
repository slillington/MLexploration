"""Analyze the quality of the final DiseaseProfile output.

Examines:
- Profile completeness (all sections populated)
- Evidence citation density (PMIDs per section)
- Gene coverage breadth
- Therapy landscape coverage
- Unmet needs specificity

Usage:
    uv run python testrun-evaluation/analyze_output_quality.py NSCLC_output.txt
"""

import json
import re
import sys
from pathlib import Path


def extract_profile_json(text: str) -> dict | None:
    """Extract the JSON DiseaseProfile from the output text."""
    # Find the JSON block between RESULT header and PAPERS header
    m = re.search(r"RESULT: DiseaseProfile\n=+\n\n(\{.*?\})\n\n\n=+\nPAPERS",
                  text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: find largest JSON object
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = text[start:i + 1]
                if len(candidate) > 1000:
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                start = None
    return None


def extract_pmids(text: str) -> list[str]:
    """Extract all PMID references from a string."""
    return re.findall(r"PMID[:\s]*(\d{7,8})", text)


def analyze_profile(profile: dict) -> dict:
    """Analyze profile quality metrics."""
    results: dict = {}

    # Basic completeness
    results["disease_name"] = profile.get("disease_name", "")
    results["n_synonyms"] = len(profile.get("synonyms", []))
    results["has_description"] = bool(profile.get("description"))
    results["description_length"] = len(profile.get("description", ""))

    # Pathways
    pathways = profile.get("key_pathways", [])
    results["n_pathways"] = len(pathways)
    pathway_genes = set()
    pathway_pmids = set()
    for p in pathways:
        for g in p.get("key_genes", []):
            pathway_genes.add(g)
        for pmid in extract_pmids(p.get("evidence_summary", "")):
            pathway_pmids.add(pmid)
    results["pathway_unique_genes"] = len(pathway_genes)
    results["pathway_unique_pmids"] = len(pathway_pmids)
    results["pathway_pmids_per_entry"] = (
        len(pathway_pmids) / max(len(pathways), 1)
    )
    results["pathway_details"] = [
        {
            "name": p.get("name", ""),
            "n_genes": len(p.get("key_genes", [])),
            "n_pmids": len(extract_pmids(p.get("evidence_summary", ""))),
            "evidence_length": len(p.get("evidence_summary", "")),
        }
        for p in pathways
    ]

    # Genetic associations (supports both old and new schema)
    somatic = profile.get("somatic_genomics", [])
    germline = profile.get("germline_genetics", [])
    old_genetics = profile.get("genetic_associations", [])
    results["n_somatic_genomics"] = len(somatic)
    results["n_germline_genetics"] = len(germline)
    results["n_genetic_associations_legacy"] = len(old_genetics)
    results["germline_note"] = profile.get("germline_note", "")

    alteration_types = {}
    for g in somatic:
        t = g.get("alteration_type", "unknown")
        alteration_types[t] = alteration_types.get(t, 0) + 1
    results["alteration_types"] = alteration_types

    assoc_types = {}
    for g in germline:
        t = g.get("association_type", "unknown")
        assoc_types[t] = assoc_types.get(t, 0) + 1
    # Include legacy associations if new fields are empty
    if not somatic and not germline:
        for g in old_genetics:
            t = g.get("association_type", "unknown")
            assoc_types[t] = assoc_types.get(t, 0) + 1
    results["association_types"] = assoc_types

    genetic_pmids = set()
    for g in somatic + germline + old_genetics:
        for pmid in extract_pmids(g.get("evidence_summary", "")):
            genetic_pmids.add(pmid)
    results["genetic_unique_pmids"] = len(genetic_pmids)

    # Therapies
    therapies = profile.get("existing_therapies", [])
    results["n_therapies"] = len(therapies)
    statuses = {}
    for t in therapies:
        s = t.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    results["therapy_statuses"] = statuses
    results["therapy_details"] = [
        {
            "drug": t.get("drug_name", ""),
            "target": t.get("target", ""),
            "status": t.get("status", ""),
            "has_limitations": bool(t.get("limitations")),
        }
        for t in therapies
    ]

    # Unmet needs
    unmet = profile.get("unmet_needs", [])
    results["n_unmet_needs"] = len(unmet)
    results["unmet_needs_details"] = []
    for u in unmet:
        desc = u if isinstance(u, str) else u.get("description", str(u))
        results["unmet_needs_details"].append({
            "length": len(desc),
            "has_specific_target": bool(re.search(r"[A-Z]{2,}\d?", desc)),
        })

    # Literature summary
    lit = profile.get("literature_summary", "")
    results["literature_summary_length"] = len(lit)
    results["literature_summary_pmids"] = len(extract_pmids(lit))

    # Overall PMID coverage
    all_text = json.dumps(profile)
    all_pmids = set(extract_pmids(all_text))
    results["total_unique_pmids_cited"] = len(all_pmids)

    # All unique genes mentioned
    all_genes = set()
    for p in pathways:
        all_genes.update(p.get("key_genes", []))
    for g in somatic + germline + old_genetics:
        gene = g.get("gene_symbol", "")
        if gene:
            all_genes.add(gene)
    results["total_unique_genes"] = len(all_genes)

    return results


def print_report(analysis: dict) -> None:
    print("=" * 80)
    print("OUTPUT QUALITY ANALYSIS")
    print("=" * 80)

    print(f"\nDisease: {analysis['disease_name']}")
    print(f"Synonyms: {analysis['n_synonyms']}")
    print(f"Description: {analysis['description_length']} chars")

    print(f"\n--- Pathways ({analysis['n_pathways']}) ---")
    print(f"  Unique genes across pathways: {analysis['pathway_unique_genes']}")
    print(f"  Unique PMIDs cited: {analysis['pathway_unique_pmids']}")
    print(f"  Mean PMIDs per pathway: {analysis['pathway_pmids_per_entry']:.1f}")
    for p in analysis["pathway_details"]:
        print(f"    {p['name'][:50]:<52} genes={p['n_genes']:>2}  "
              f"PMIDs={p['n_pmids']:>2}  evidence={p['evidence_length']:>4} chars")

    print(f"\n--- Somatic Genomics ({analysis['n_somatic_genomics']}) ---")
    print(f"  Alteration types: {analysis['alteration_types']}")
    print(f"\n--- Germline Genetics ({analysis['n_germline_genetics']}) ---")
    print(f"  Association types: {analysis['association_types']}")
    if analysis.get("germline_note"):
        print(f"  Note: {analysis['germline_note']}")
    if analysis["n_genetic_associations_legacy"]:
        print(f"  (Legacy genetic_associations: {analysis['n_genetic_associations_legacy']})")
    print(f"  Unique PMIDs cited (all genetics): {analysis['genetic_unique_pmids']}")

    print(f"\n--- Therapies ({analysis['n_therapies']}) ---")
    print(f"  Status breakdown: {analysis['therapy_statuses']}")
    for t in analysis["therapy_details"]:
        lim = "✓" if t["has_limitations"] else "✗"
        print(f"    {t['drug']:<35} {t['target']:<10} {t['status']:<12} limitations={lim}")

    print(f"\n--- Unmet Needs ({analysis['n_unmet_needs']}) ---")
    specific = sum(1 for u in analysis["unmet_needs_details"] if u["has_specific_target"])
    print(f"  With specific molecular targets: {specific}/{analysis['n_unmet_needs']}")

    print(f"\n--- Literature Summary ---")
    print(f"  Length: {analysis['literature_summary_length']} chars")
    print(f"  PMIDs cited: {analysis['literature_summary_pmids']}")

    print(f"\n--- Overall ---")
    print(f"  Total unique PMIDs cited in profile: {analysis['total_unique_pmids_cited']}")
    print(f"  Total unique genes mentioned: {analysis['total_unique_genes']}")

    # Quality score heuristic
    score = 0
    score += min(analysis["n_pathways"], 10) * 1.0  # up to 10
    n_genetics = (analysis["n_somatic_genomics"] + analysis["n_germline_genetics"]
                  + analysis["n_genetic_associations_legacy"])
    score += min(n_genetics, 10) * 0.5  # up to 5
    score += min(analysis["n_therapies"], 12) * 0.5  # up to 6
    score += min(analysis["n_unmet_needs"], 10) * 0.5  # up to 5
    score += min(analysis["total_unique_pmids_cited"], 30) * 0.2  # up to 6
    score += min(analysis["pathway_unique_genes"], 30) * 0.1  # up to 3
    score += (1 if analysis["has_description"] else 0) * 2  # 2
    score += (1 if analysis["literature_summary_length"] > 500 else 0) * 3  # 3
    max_score = 10 + 5 + 6 + 5 + 6 + 3 + 2 + 3  # 40
    print(f"\n  Heuristic quality score: {score:.1f} / {max_score} "
          f"({score/max_score*100:.0f}%)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: analyze_output_quality.py <path-to-output.txt>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    profile = extract_profile_json(text)
    if not profile:
        print("ERROR: Could not extract DiseaseProfile JSON from output", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_profile(profile)
    print_report(analysis)

    out_path = path.with_name(path.stem + "_output_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n→ Analysis written to {out_path}")


if __name__ == "__main__":
    main()
