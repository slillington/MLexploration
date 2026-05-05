"""Prompt construction tools — LLM-callable tools that build prompt fragments.

These replace the static .txt template files in prompts/. The FeedbackAgent
and other agents call these tools to dynamically construct expert personas,
output schemas, and extraction guidelines.
"""

from __future__ import annotations

from targetsearch.core.registry import registry

# ── Output schema definitions ──────────────────────────────────────────

_OUTPUT_SCHEMAS: dict[str, str] = {
    "paper_summary": """\
{
  "paper_type": "primary research | meta-analysis | case study | case series | clinical trial",
  "study_design": "e.g. 'randomized phase II', 'retrospective cohort', 'CRISPR screen in cell lines', 'scRNA-seq of patient biopsies'. Be specific.",
  "objective": "What question the paper set out to answer (1-2 sentences)",
  "key_findings": [
    {
      "finding": "What was directly observed or measured — not author interpretation",
      "evidence_type": "in vivo | in vitro | clinical | computational | epidemiological | meta-analysis",
      "model_system": "e.g. 'bleomycin mouse model', 'primary human lung fibroblasts', 'IPF patient cohort (n=200)'",
      "effect_size": "Quantitative result if reported: OR, HR, fold-change, p-value, confidence interval. Leave empty if not reported.",
      "genes_proteins": ["GENE1", "GENE2"]
    }
  ],
  "evidence_strength": "strong | moderate | weak | insufficient — based on study design, sample size, and reproducibility",
  "methods_summary": "Brief description of experimental approach (2-3 sentences)",
  "limitations": "Authors' stated limitations or obvious methodological gaps",
  "target_relevance": "How these findings relate to potential drug targets — cite only what the evidence supports, not speculation",
  "genes_pathways_mentioned": ["GENE1", "PATHWAY1", "GENE2"]
}""",
    "review_mining": """\
{
  "review_title": "Title of the review being analyzed",
  "cited_papers": [
    {
      "pmid": "PubMed ID if identifiable from the citation, otherwise null",
      "description": "1-sentence description of what this cited paper showed",
      "relevance": "Why this paper matters for drug target discovery",
      "priority": "high | medium | low"
    }
  ]
}""",
    "disease_profile": """\
{
  "disease_name": "canonical disease name",
  "synonyms": ["alternative names"],
  "description": "1-2 sentence description of the disease",
  "key_pathways": [
    {
      "name": "pathway name",
      "description": "what this pathway does in the disease context",
      "key_genes": ["GENE1", "GENE2"],
      "evidence_summary": "brief summary citing specific papers by PMID"
    }
  ],
  "somatic_genomics": [
    {
      "gene_symbol": "GENE (HGNC symbol)",
      "alteration_type": "mutation | amplification | fusion | loss | overexpression",
      "frequency": "approximate frequency in the disease population, if known",
      "evidence_summary": "what the somatic evidence shows, citing PMIDs",
      "source": "specific paper or database"
    }
  ],
  "germline_genetics": [
    {
      "gene_symbol": "GENE (HGNC symbol)",
      "association_type": "GWAS | Mendelian | eQTL",
      "evidence_summary": "what the germline evidence shows, citing PMIDs",
      "source": "GWAS Catalog | ClinVar | Open Targets | specific paper"
    }
  ],
  "germline_note": "If no germline evidence exists, state that explicitly here. Leave empty otherwise.",
  "existing_therapies": [
    {
      "drug_name": "drug name",
      "target": "gene/protein target",
      "mechanism": "mechanism of action",
      "status": "approved | phase III | phase II | etc.",
      "limitations": "why this therapy is insufficient"
    }
  ],
  "unmet_needs": ["specific unmet medical needs"],
  "literature_summary": "2-3 paragraph synthesis highlighting convergent evidence, contradictions, and key gaps. Cite PMIDs."
}""",
    "evidence_audit": """\
{
  "coverage_by_bucket": {
    "disease_biology": "<int: number of papers>",
    "human_genetics": "<int: number of papers>",
    "preclinical_therapeutic": "<int: number of papers>",
    "clinical": "<int: number of papers>"
  },
  "contradictions": [
    {
      "topic": "<what the contradiction is about>",
      "paper_a": "<PMID and finding>",
      "paper_b": "<PMID and opposing finding>",
      "assessment": "<which is stronger and why>"
    }
  ],
  "unresolved_questions": [
    "<high-impact question the evidence raises but does not answer>"
  ]
}""",
    "quality_critique": """\
{
  "section_scores": {
    "pathways": "<float 0-10>",
    "somatic_genomics": "<float 0-10>",
    "germline_genetics": "<float 0-10>",
    "existing_therapies": "<float 0-10>",
    "unmet_needs": "<float 0-10>",
    "literature_summary": "<float 0-10>"
  },
  "hard_failures": [
    "<unsupported claim or factual error that must be fixed>"
  ],
  "weak_sections": [
    "<section name that scored below threshold>"
  ],
  "revision_instructions": [
    "<specific, actionable instruction for improving a weak section>"
  ]
}""",
}

# ── Extraction guidelines ──────────────────────────────────────────────

_EXTRACTION_GUIDELINES: dict[str, str] = {
    "full_text": """\
## Extraction guidelines (full text available)

- Extract findings from Results sections. Use Discussion only to
  contextualize, not as a source of new findings.
- For each key finding, be specific about the evidence type and model
  system. A finding without a model system is incomplete.
- Always include effect sizes when the paper reports them (OR, HR,
  fold-change, p-value, CI). Leave effect_size empty if not reported.
- The genes_pathways_mentioned list should include ALL genes, proteins,
  and pathways discussed, not just those in key_findings.
- If the paper is a meta-analysis, treat pooled effect sizes as key
  findings and note the number of studies/patients pooled.
- For preclinical papers: note the model system specifics (cell line
  name, mouse strain, organoid source). Note whether findings were
  validated in more than one model.
- For clinical papers: note trial phase, cohort size, primary endpoint,
  and comparator arm. Distinguish investigator-assessed from
  independent-review outcomes if both are reported.""",
    "abstract": """\
## Extraction guidelines (abstract only)

- Extract ONLY what the abstract explicitly states. Do not infer
  methods, controls, or effect sizes that are not written.
- Note in methods_summary that this extraction is from abstract only —
  findings may be incomplete and evidence_strength should reflect this.
- Effect sizes in abstracts are often the headline result; include them
  but note the limited context.
- If the abstract mentions specific genes or pathways, include them
  even if the evidence description is brief.
- Set evidence_strength to "insufficient" if the abstract provides no
  methods detail at all.""",
    "review": """\
## Extraction guidelines (review article)

A review article synthesizes evidence from many primary sources. Your
job is to extract the review's key conclusions as structured findings,
not to catalog every paper it cites.

### What to extract

- Extract the review's major conclusions as key_findings. Each finding
  should be a synthesized claim that the review supports with evidence
  from multiple primary sources.
  Example: "EGFR exon 20 insertion mutations occur in 2-3% of NSCLC
  and are resistant to first- and second-generation EGFR TKIs but
  respond to amivantamab and mobocertinib (PMID 34911336, 35534623)."

- For each finding, include the PMIDs of the primary sources the review
  cites as evidence, in the genes_proteins field list the relevant
  genes, and in evidence_type use "review synthesis".

- Include effect sizes when the review reports pooled or comparative
  statistics (e.g., "pooled ORR 40% across 3 trials").

- For model_system, describe the scope of evidence the review covers
  (e.g., "meta-analysis of 12 phase II-III trials", "narrative review
  of preclinical and clinical data").

### What NOT to extract

- Do not extract background statements that provide no actionable
  insight (e.g., "Lung cancer is the leading cause of cancer death").
- Do not extract every individual study the review mentions. Extract
  the review's synthesized conclusions, not a list of its references.
- Do not set evidence_type to "in vivo" or "in vitro" — the review
  itself is not performing experiments. Use "review synthesis" or
  "meta-analysis" as appropriate.

### Prioritization

Extract findings that address these dimensions of target assessment:

1. **Target biology and causal chain.** Pathway mechanisms linking
   target modulation to disease outcome. Feedback loops, compensatory
   pathways, or redundancies that could limit efficacy. On-target
   safety liabilities based on the target's normal physiological role
   (e.g., knockout phenotypes, tissue expression, known toxicities).

2. **Genetic support.** Human genetic evidence linking the target to
   disease: GWAS associations, rare variant studies, Mendelian
   randomization, eQTL data. Whether the genetic evidence supports
   inhibition vs. activation. Genetic signals defining patient
   subpopulations likely to respond.

3. **Druggability and modality.** Whether the target can be drugged
   and with what modality (antibody, ADC, bispecific, small molecule,
   PROTAC, cell therapy). Modality-specific challenges: PK/PD,
   tissue penetration, antigen density, internalization, half-life.
   Comparative efficacy across modalities when reviewed.

4. **Target expression and accessibility.** Cell-surface expression
   levels and tumor selectivity vs. normal tissue expression (safety
   window). For biologics: internalization rate (relevant for ADCs),
   shedding, antigen density. Expression heterogeneity across tumor
   subtypes or disease stages.

5. **Clinical relevance and competitive landscape.** Programs in the
   clinic targeting this pathway. Biomarker-defined patient subgroups
   and response predictors. Resistance mechanisms to existing
   therapies. Regulatory precedents and endpoints. Unmet therapeutic
   needs that a new agent could address.

6. **Differentiation from existing agents.** What distinguishes
   molecules targeting the same pathway: binding epitope, valency,
   Fc engineering, payload chemistry (ADCs), conditional activity,
   bispecific geometry. Head-to-head or cross-trial comparisons when
   available.

7. **Translational evidence.** Whether preclinical efficacy
   translated to clinical benefit — and when it did not, why (e.g.,
   insufficient tumor penetration, compensatory pathway activation,
   inadequate patient selection). Predictive biomarkers that emerged
   from translational studies.

8. **Combination rationale.** Mechanistic basis for combination
   therapies (e.g., anti-PD-1 + anti-TIGIT, EGFR + MET bispecific).
   Synergy data from preclinical or clinical studies. Whether
   combinations overcome resistance to monotherapy. Sequencing
   considerations.

9. **Patient stratification and biomarkers.** Companion diagnostic
   feasibility and availability. Prevalence of the biomarker-positive
   population. Whether the biomarker is prognostic vs. predictive.
   Co-occurrence patterns that define responder subgroups (e.g., EGFR
   mutation + TP53 co-mutation).

Deprioritize: historical context, epidemiology, staging/diagnosis,
and general disease biology that does not inform target selection,
druggability, or clinical strategy.

### Citation handling

- You will receive a REFERENCE LIST of valid PMIDs from this article's
  bibliography. ONLY cite PMIDs that appear in that list.
- If the review attributes a finding to a study whose PMID is in the
  reference list, include it as "(PMID XXXXXXXX)" in the finding text.
- If the review attributes a finding to a study whose PMID is NOT in
  the reference list, describe the finding without a citation. Do not
  guess or fabricate PMIDs.
- If no reference list is provided (abstract-only reviews), do not
  include any inline PMID citations.
- Format: "(PMID XXXXXXXX)" or "(PMID XXXXXXXX, YYYYYYYY)" for
  multiple sources.

### Fields

- paper_type: "narrative review", "systematic review", or "meta-analysis"
- study_design: describe the review's scope and methodology (e.g.,
  "systematic review of biologic therapies in NSCLC, 2018-2024,
  covering 45 clinical trials")
- evidence_strength: assess based on the review's rigor:
  - strong: systematic review or meta-analysis with clear methodology
  - moderate: narrative review by domain experts with comprehensive
    coverage
  - weak: brief or selective review, opinion-heavy
  - insufficient: abstract-only or commentary without systematic
    evidence assessment
- target_relevance: summarize the review's overall implications for
  drug target selection — which targets does it highlight as most
  promising and why?
- Aim for 8-15 key_findings. Fewer is acceptable if the review is
  narrow; more than 15 suggests you are extracting individual studies
  rather than synthesized conclusions.""",
}


# ── Registered tools ───────────────────────────────────────────────────


@registry.tool(
    description=(
        "Build an expert persona prompt section for a given disease area "
        "and list of expertise domains. Returns a system prompt fragment."
    ),
    tags=["prompts"],
    params={
        "disease_area": "The disease being investigated (e.g. 'idiopathic pulmonary fibrosis')",
        "expertise": "List of expertise domains (e.g. ['genetics', 'medicinal chemistry'])",
    },
    returns="System prompt fragment describing the expert persona",
)
def create_expert_persona(disease_area: str, expertise: list[str]) -> str:
    expertise_str = ", ".join(expertise)
    return (
        f"You are a critical evidence reviewer specializing in {disease_area}. "
        f"Your areas of expertise include: {expertise_str}.\n\n"
        f"Your extraction principles:\n"
        f"- Separate observed data from author interpretation. Report what "
        f"was measured, not what the authors speculate it means.\n"
        f"- Flag when a finding comes from a different disease context "
        f"(e.g., breast cancer data applied to NSCLC).\n"
        f"- Note when effect sizes, sample sizes, or statistical tests "
        f"are missing — do not fill gaps with assumptions.\n"
        f"- Use correct HGNC gene symbols. If the paper uses a non-standard "
        f"name, include both (e.g., 'TROP2/TACSTD2').\n"
        f"- Distinguish preclinical from clinical evidence. A cell-line "
        f"result is not equivalent to a patient cohort result."
    )


@registry.tool(
    description=(
        "Return the JSON output format instructions for a given schema type. "
        "Available schemas: paper_summary, review_mining, disease_profile."
    ),
    tags=["prompts"],
    params={
        "schema_name": "Schema type: 'paper_summary', 'review_mining', or 'disease_profile'",
    },
    returns="Output schema instructions as a formatted string",
)
def create_output_schema(schema_name: str) -> str:
    schema = _OUTPUT_SCHEMAS.get(schema_name)
    if schema is None:
        available = ", ".join(sorted(_OUTPUT_SCHEMAS.keys()))
        return f"ERROR: Unknown schema '{schema_name}'. Available: {available}"
    return f"## Output schema\n\nProduce a JSON object matching this exact schema:\n\n{schema}\n\nOutput ONLY the JSON object. No markdown fences, no explanation."


@registry.tool(
    description=(
        "Return extraction guidelines specific to a source type. "
        "Available types: full_text, abstract, review."
    ),
    tags=["prompts"],
    params={
        "source_type": "Source type: 'full_text', 'abstract', or 'review'",
    },
    returns="Extraction guidelines as a formatted string",
)
def create_extraction_guidelines(source_type: str) -> str:
    guidelines = _EXTRACTION_GUIDELINES.get(source_type)
    if guidelines is None:
        available = ", ".join(sorted(_EXTRACTION_GUIDELINES.keys()))
        return f"ERROR: Unknown source type '{source_type}'. Available: {available}"
    return guidelines
