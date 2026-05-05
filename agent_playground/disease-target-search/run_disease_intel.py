"""Run the Disease Intel agent with the full paper pipeline.

Usage:
    .venv/bin/python run_disease_intel.py [disease_name]

Default disease: idiopathic pulmonary fibrosis
"""

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

from targetsearch.core.logging import setup_logging
from targetsearch.agents.disease_intel import DiseaseIntelAgent

disease = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "idiopathic pulmonary fibrosis"

run_id = setup_logging()

print(f"\n{'='*70}")
print(f"Disease Intel Agent (v2) — researching: {disease}")
print(f"Run ID: {run_id}  (logs in logs/)")
print(f"{'='*70}\n")

agent = DiseaseIntelAgent()
profile = agent.run(disease)

print(f"\n{'='*70}")
print("RESULT: DiseaseProfile")
print(f"{'='*70}\n")

# Print the profile without the bulky paper_summaries
profile_dict = profile.model_dump()
summaries = profile_dict.pop("paper_summaries", [])
print(json.dumps(profile_dict, indent=2, default=str))

print(f"\n{'='*70}")
print(f"PAPER SUMMARIES ({len(summaries)} papers processed)")
print(f"{'='*70}")
for i, ps in enumerate(summaries, 1):
    src = ps.get("source_type", "?")
    n_findings = len(ps.get("key_findings", []))
    genes = ps.get("genes_pathways_mentioned", [])
    print(f"  {i:2d}. PMID {ps.get('pmid', '?'):>10s} [{src:9s}] "
          f"{n_findings} findings, {len(genes)} genes — "
          f"{ps.get('title', '(no title)')[:60]}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Disease:        {profile.disease_name}")
print(f"Synonyms:       {', '.join(profile.synonyms) if profile.synonyms else '(none)'}")
print(f"Pathways:       {len(profile.key_pathways)}")
print(f"Somatic hits:   {len(profile.somatic_genomics)}")
print(f"Germline hits:  {len(profile.germline_genetics)}")
if profile.germline_note:
    print(f"Germline note:  {profile.germline_note}")
print(f"Therapies:      {len(profile.existing_therapies)}")
print(f"Unmet needs:    {len(profile.unmet_needs)}")
print(f"Papers:         {len(profile.paper_summaries)}")
full_text_count = sum(1 for ps in profile.paper_summaries if ps.source_type == "full_text")
print(f"  Full text:    {full_text_count}")
print(f"  Abstract:     {len(profile.paper_summaries) - full_text_count}")
