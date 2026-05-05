"""Synthesis tools — coordination tools that read from ActionContext.

synthesize_disease_profile reads the full, uncompressed PaperSummary list
and Open Targets data from the context, calls the LLM, and writes the
resulting DiseaseProfile back to the context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from targetsearch.core.context import ActionContext
from targetsearch.core.llm import llm_text, parse_json_response
from targetsearch.core.registry import registry
from targetsearch.schemas.disease import DiseaseProfile
from targetsearch.tools.prompt_tools import create_output_schema

log = logging.getLogger(__name__)


def _build_synthesis_prompt() -> str:
    """Build the system prompt for disease profile synthesis."""
    schema = create_output_schema("disease_profile")
    return f"""\
You are a disease intelligence analyst specializing in drug target discovery.

You are given structured summaries of research papers and target database results for a disease. Your job is to synthesize these into a disease profile that downstream agents will use to generate novel drug target hypotheses.

## Your input

You will receive:
1. Open Targets data: genetically associated targets with scores
2. Paper summaries: structured extractions from primary research papers, each containing key findings, methods, effect sizes, and genes/pathways mentioned

{schema}

## Guidelines

- Ground every claim in specific paper summaries. Reference PMIDs.
- When multiple papers report findings about the same gene/pathway, note the convergence — this strengthens the evidence.
- When papers contradict each other, note the contradiction and which evidence is stronger (larger cohort, better model system, more recent).
- For pathways, name the specific genes involved and cite the papers that implicate them.
- **somatic_genomics**: Only include somatic tumor alterations (mutations, amplifications, fusions, losses, overexpression). Use correct HGNC gene symbols. Include frequency if reported.
- **germline_genetics**: Only include germline/population genetic evidence (GWAS, Mendelian, eQTL). If no germline evidence exists in the papers, leave the list empty and set germline_note to explain why (e.g., "No germline associations reported in the reviewed literature").
- Do NOT put expression data or somatic alterations in germline_genetics, and do NOT put germline variants in somatic_genomics.
- The literature_summary should emphasize open questions and controversies, not settled science. This is what the hypothesis agent will use to find novel angles."""


def _serialize_summaries(
    summaries_list: list[Any] | None = None,
    ctx: ActionContext | None = None,
) -> str:
    """Serialize paper summaries for the synthesis LLM call.

    Drops ``methods_summary`` and ``authors`` to reduce per-paper token
    cost.  All other fields (key_findings, limitations, target_relevance,
    genes_pathways_mentioned, identifiers) are preserved.

    Accepts either an explicit list of PaperSummary objects or reads from
    ``ctx.paper_state.summaries``.
    """
    if summaries_list is None:
        if ctx is None:
            raise ValueError("Provide summaries_list or ctx")
        summaries_list = ctx.paper_state.summaries

    serialized = []
    for ps in summaries_list:
        summary_dict = {
            "pmid": ps.pmid,
            "title": ps.title,
            "year": ps.year,
            "paper_type": ps.paper_type,
            "objective": ps.objective,
            "key_findings": [
                {
                    "finding": kf.finding,
                    "evidence_type": kf.evidence_type,
                    "model_system": kf.model_system,
                    "effect_size": kf.effect_size,
                    "genes_proteins": kf.genes_proteins,
                }
                for kf in ps.key_findings
            ],
            "limitations": ps.limitations,
            "target_relevance": ps.target_relevance,
            "genes_pathways_mentioned": ps.genes_pathways_mentioned,
            "source_type": ps.source_type,
        }
        serialized.append(summary_dict)
    return json.dumps(serialized, indent=2, default=str)


def _build_audit_prompt() -> str:
    """Build the system prompt for Pass A: Evidence Audit."""
    schema = create_output_schema("evidence_audit")
    return f"""\
You are a disease intelligence analyst. You are given structured summaries \
of research papers and Open Targets data for a disease. Your job is to \
audit the evidence BEFORE synthesis — do NOT produce a disease profile.

## Guidelines

- Count each paper into exactly one evidence bucket based on its primary \
contribution.
- Only flag genuine contradictions — not papers studying different aspects \
of the same topic.
- Unresolved questions should be specific and actionable, not generic \
(e.g., "Is TROP2 expressed on the surface of driver-negative NSCLC cells?" \
not "More research is needed").

{schema}"""


def _run_audit(
    summaries_json: str,
    n_papers: int,
    disease_name: str,
    synonyms: list[str],
    ot_section: str,
    batch_label: str = "",
) -> dict[str, Any]:
    """Run Pass A: Evidence Audit on a set of paper summaries.

    Returns parsed audit dict with coverage_by_bucket, contradictions,
    and unresolved_questions.  Returns a degraded default on parse failure.
    """
    user_message = (
        f"Disease: {disease_name}\n"
        f"Synonyms: {', '.join(synonyms)}\n\n"
        f"## Open Targets Data\n\n{ot_section}\n\n"
        f"## Paper Summaries ({n_papers} papers)\n\n{summaries_json}"
    )

    caller = f"audit[{batch_label}]" if batch_label else "audit"
    raw = llm_text(
        [
            {"role": "system", "content": _build_audit_prompt()},
            {"role": "user", "content": user_message},
        ],
        caller=caller,
    )

    try:
        data = parse_json_response(raw)
        return {
            "coverage_by_bucket": data.get("coverage_by_bucket", {}),
            "contradictions": data.get("contradictions", []),
            "unresolved_questions": data.get("unresolved_questions", []),
        }
    except Exception as e:
        log.error("Failed to parse audit output (%s): %s", batch_label, e)
        return {
            "coverage_by_bucket": {},
            "contradictions": [],
            "unresolved_questions": [],
        }


def _merge_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge audit results from multiple batches."""
    merged_coverage: dict[str, int] = {}
    merged_contradictions: list[dict] = []
    merged_questions: list[str] = []

    for audit in audits:
        for bucket, count in audit.get("coverage_by_bucket", {}).items():
            merged_coverage[bucket] = merged_coverage.get(bucket, 0) + count
        merged_contradictions.extend(audit.get("contradictions", []))
        merged_questions.extend(audit.get("unresolved_questions", []))

    # Deduplicate questions
    seen: set[str] = set()
    unique_questions = []
    for q in merged_questions:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique_questions.append(q)

    return {
        "coverage_by_bucket": merged_coverage,
        "contradictions": merged_contradictions,
        "unresolved_questions": unique_questions,
    }


def _synthesize_batch(
    summaries_list: list[Any],
    disease_name: str,
    synonyms: list[str],
    ot_section: str,
    audit_result: dict[str, Any] | None = None,
    batch_label: str = "",
) -> DiseaseProfile:
    """Run Pass B: Draft Profile for a batch of paper summaries."""
    summaries_json = _serialize_summaries(summaries_list=summaries_list)
    n_papers = len(summaries_list)

    user_message = (
        f"Disease: {disease_name}\n"
        f"Synonyms: {', '.join(synonyms)}\n\n"
        f"## Open Targets Data\n\n{ot_section}\n\n"
        f"## Paper Summaries ({n_papers} papers)\n\n{summaries_json}"
    )

    if audit_result:
        user_message += (
            f"\n\n## Evidence Audit Results\n\n"
            f"{json.dumps(audit_result, indent=2, default=str)}"
        )

    system_prompt = _build_synthesis_prompt()
    caller = f"draft[{batch_label}]" if batch_label else "draft"
    raw = llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        caller=caller,
    )

    try:
        data = parse_json_response(raw)
        return DiseaseProfile.model_validate(data)
    except Exception as e:
        log.error("Failed to parse draft DiseaseProfile (%s): %s", batch_label, e)
        return DiseaseProfile(
            disease_name=disease_name,
            description=f"Draft synthesis failed: {e}",
            literature_summary=str(raw)[:3000],
        )


def _build_merge_prompt() -> str:
    """Build the system prompt for merging intermediate profiles."""
    schema = create_output_schema("disease_profile")
    return f"""\
You are a disease intelligence analyst. You are given multiple intermediate \
disease profiles that were synthesized from different batches of research \
papers for the same disease. Your job is to merge them into a single, \
unified disease profile.

## Merge guidelines

- **Pathways:** Deduplicate by name. When the same pathway appears in \
multiple profiles, merge their key_genes lists and combine evidence \
summaries, citing all relevant PMIDs.
- **Genetic associations:** Deduplicate by gene_symbol. Merge evidence \
from multiple profiles for the same gene.
- **Existing therapies:** Deduplicate by drug_name. Keep the most \
complete entry (most fields filled).
- **Unmet needs:** Deduplicate semantically — combine similar needs \
into a single statement.
- **Literature summary:** Write a unified narrative that integrates \
findings from all profiles. Do not simply concatenate — synthesize \
the key themes, convergences, and contradictions across all batches.

{schema}"""


def _merge_profiles(
    profiles: list[DiseaseProfile],
    disease_name: str,
) -> DiseaseProfile:
    """Merge multiple intermediate DiseaseProfiles into a final one."""
    profiles_json = []
    for i, p in enumerate(profiles):
        profiles_json.append(
            json.dumps(
                p.model_dump(exclude={"paper_summaries"}),
                indent=2,
                default=str,
            )
        )

    user_message = (
        f"Disease: {disease_name}\n\n"
        f"## Intermediate Profiles ({len(profiles)} batches)\n\n"
        + "\n\n---\n\n".join(
            f"### Batch {i + 1}\n\n{pj}"
            for i, pj in enumerate(profiles_json)
        )
    )

    system_prompt = _build_merge_prompt()
    raw = llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        caller="merge_profiles",
    )

    try:
        data = parse_json_response(raw)
        return DiseaseProfile.model_validate(data)
    except Exception as e:
        log.error("Failed to parse merged DiseaseProfile: %s", e)
        # Fall back to the first profile
        return profiles[0] if profiles else DiseaseProfile(
            disease_name=disease_name,
            description=f"Merge failed: {e}",
        )


def _build_evidence_index(summaries_list: list[Any]) -> str:
    """Build a compact evidence index for the critique pass.

    Instead of sending full paper summaries (~2-3K chars each), produce a
    one-line-per-paper index with PMID, title, finding count, gene list,
    and source type.  The critique scores the *profile*, not the papers —
    it only needs enough to verify that cited PMIDs exist and roughly match
    the claimed topic.
    """
    lines = []
    for ps in summaries_list:
        pmid = getattr(ps, "pmid", None) or "?"
        title = (getattr(ps, "title", "") or "")[:80]
        n_findings = len(getattr(ps, "key_findings", []))
        genes = getattr(ps, "genes_pathways_mentioned", []) or []
        genes_str = ", ".join(genes[:8])
        if len(genes) > 8:
            genes_str += f" (+{len(genes) - 8} more)"
        source = getattr(ps, "source_type", "?")
        paper_type = getattr(ps, "paper_type", "")
        strength = getattr(ps, "evidence_strength", "") or ""
        design = getattr(ps, "study_design", "") or ""
        design_str = f" | design={design}" if design else ""
        strength_str = f" | strength={strength}" if strength else ""
        lines.append(
            f"PMID {pmid}: \"{title}\" | {n_findings} findings | "
            f"genes=[{genes_str}] | {paper_type} | {source}"
            f"{design_str}{strength_str}"
        )
    return "\n".join(lines)


def _build_critique_prompt() -> str:
    """Build the system prompt for Pass C: Quality Critique."""
    schema = create_output_schema("quality_critique")
    return f"""\
You are a quality reviewer for disease intelligence profiles used in \
drug target discovery. You are given a draft disease profile and a \
compact evidence index listing each paper's PMID, title, finding count, \
and gene coverage. Your job is to critique the profile's quality.

## What you receive

- The full draft disease profile (JSON).
- An evidence index: one line per paper with PMID, title, finding count, \
genes, paper type, and source type. Use this to verify that cited PMIDs \
exist and roughly match the claimed topic.
- Optionally, an evidence audit with coverage counts, contradictions, \
and unresolved questions.

## Score each profile section from 0-10

- **pathways**: Are pathways well-defined with specific genes and PMID citations? \
Are key_genes populated with correct HGNC symbols? Is the evidence_summary grounded?
- **somatic_genomics**: Are alterations correctly typed (mutation, amplification, fusion, \
loss, overexpression)? Are gene symbols correct HGNC? Is frequency cited where available? \
Do not penalize an empty section if the disease has no relevant somatic alterations.
- **germline_genetics**: Are associations correctly typed (GWAS, Mendelian, eQTL)? \
Is evidence cited? If the section is empty but germline_note explains why, score 7+. \
An empty section with no explanation scores 3 or below.
- **existing_therapies**: Are drugs named with targets, mechanisms, and development status? \
Are limitations noted?
- **unmet_needs**: Are needs specific and actionable, not generic? Do they follow from \
the evidence?
- **literature_summary**: Does it synthesize (not just list) findings? Does it highlight \
contradictions and open questions? Are PMIDs cited?

Be strict. A score of 7+ means the section is publication-ready. \
A score below 5 means the section has significant gaps or errors.

{schema}"""


def _run_critique(
    profile: DiseaseProfile,
    evidence_index: str,
    n_papers: int,
    audit_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Pass C: Quality Critique on a draft profile.

    Args:
        profile: The draft DiseaseProfile to critique.
        evidence_index: Compact per-paper index (from ``_build_evidence_index``),
            not the full serialized summaries.  The critique scores the profile's
            structure and labeling — it doesn't need to re-read paper findings.
        n_papers: Total number of papers in the evidence base.
        audit_result: Optional audit dict with coverage, contradictions, and
            unresolved questions.

    Returns parsed critique dict with section_scores, hard_failures,
    weak_sections, and revision_instructions.
    """
    profile_json = json.dumps(
        profile.model_dump(exclude={"paper_summaries"}),
        indent=2,
        default=str,
    )

    user_message = (
        f"## Draft Disease Profile\n\n{profile_json}\n\n"
        f"## Evidence Index ({n_papers} papers)\n\n{evidence_index}"
    )

    if audit_result:
        user_message += (
            f"\n\n## Evidence Audit Results\n\n"
            f"{json.dumps(audit_result, indent=2, default=str)}"
        )

    raw = llm_text(
        [
            {"role": "system", "content": _build_critique_prompt()},
            {"role": "user", "content": user_message},
        ],
        caller="critique",
    )

    try:
        data = parse_json_response(raw)
        return {
            "section_scores": data.get("section_scores", {}),
            "hard_failures": data.get("hard_failures", []),
            "weak_sections": data.get("weak_sections", []),
            "revision_instructions": data.get("revision_instructions", []),
        }
    except Exception as e:
        log.error("Failed to parse critique output: %s", e)
        return {
            "section_scores": {},
            "hard_failures": [],
            "weak_sections": [],
            "revision_instructions": [],
        }


def _build_refinement_prompt() -> str:
    """Build the system prompt for Pass D: Targeted Refinement."""
    schema = create_output_schema("disease_profile")
    return f"""\
You are a disease intelligence analyst. You are given a draft disease \
profile, a quality critique identifying weak sections, and the original \
evidence. Your job is to revise the profile to address the critique.

## Guidelines

- Only revise sections flagged as weak or containing hard failures.
- Do not remove content from strong sections.
- Address each revision instruction specifically.
- Hard failures are numbered (#1, #2, etc.). Fix each one. Do not \
introduce new errors in sections you are not revising.
- Maintain all PMID citations and add new ones where needed.
- Return the complete revised profile (not just the changed sections).

{schema}"""


def _run_refinement(
    profile: DiseaseProfile,
    critique: dict[str, Any],
    summaries_json: str,
    n_papers: int,
    pass_number: int,
) -> DiseaseProfile:
    """Run Pass D: Targeted Refinement on a draft profile.

    Returns a revised DiseaseProfile.
    """
    profile_json = json.dumps(
        profile.model_dump(exclude={"paper_summaries"}),
        indent=2,
        default=str,
    )

    # Number hard failures so the model can reference them explicitly
    numbered_critique = dict(critique)
    raw_failures = numbered_critique.get("hard_failures", [])
    if raw_failures:
        numbered_critique["hard_failures"] = [
            f"#{i + 1}: {f}" for i, f in enumerate(raw_failures)
        ]
    critique_json = json.dumps(numbered_critique, indent=2, default=str)

    user_message = (
        f"## Draft Disease Profile\n\n{profile_json}\n\n"
        f"## Quality Critique\n\n{critique_json}\n\n"
        f"## Evidence Base ({n_papers} papers)\n\n{summaries_json}"
    )

    raw = llm_text(
        [
            {"role": "system", "content": _build_refinement_prompt()},
            {"role": "user", "content": user_message},
        ],
        caller=f"refine[pass {pass_number}]",
    )

    try:
        data = parse_json_response(raw)
        return DiseaseProfile.model_validate(data)
    except Exception as e:
        log.error("Failed to parse refined DiseaseProfile (pass %d): %s", pass_number, e)
        return profile  # keep the draft on failure


def _coerce_scores(raw: dict[str, Any]) -> dict[str, float]:
    """Coerce section scores to float — LLM may return strings like "7"."""
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            log.warning("Non-numeric score for %s: %r, defaulting to 0", k, v)
            out[k] = 0.0
    return out


def _classify_hard_failures(
    current: list[str],
    previous: list[str],
) -> tuple[list[str], list[str]]:
    """Classify hard failures as new or persistent.

    A failure is "persistent" if a previous failure contains it as a
    substring or vice versa (fuzzy match on topic).  Everything else is
    "new".

    Returns (new_failures, persistent_failures).
    """
    if not previous:
        return list(current), []

    prev_lower = [f.lower() for f in previous]
    new: list[str] = []
    persistent: list[str] = []

    for failure in current:
        fl = failure.lower()
        is_persistent = any(
            fl in p or p in fl
            for p in prev_lower
        )
        if is_persistent:
            persistent.append(failure)
        else:
            new.append(failure)

    return new, persistent


def _assess_quality(
    critique: dict[str, Any],
    threshold: float,
    previous_failures: list[str] | None = None,
    max_new_hard_failures: int = 0,
) -> tuple[str, bool]:
    """Determine quality status from critique scores.

    Args:
        critique: Critique dict with section_scores and hard_failures.
        threshold: Minimum average score to pass.
        previous_failures: Hard failures from the prior critique round.
            Used to distinguish new failures from persistent ones.
        max_new_hard_failures: Tolerate up to this many *new* hard
            failures without triggering refinement.  Persistent failures
            (same as previous round) are logged but don't block passing.

    Returns (status, needs_refinement) where status is "pass", "fail",
    or "degraded".
    """
    raw_scores = critique.get("section_scores", {})
    hard_failures = critique.get("hard_failures", [])

    if not raw_scores:
        return "degraded", False

    scores = _coerce_scores(raw_scores)
    avg_score = sum(scores.values()) / len(scores)

    new_failures, persistent = _classify_hard_failures(
        hard_failures, previous_failures or [],
    )

    if new_failures and len(new_failures) > max_new_hard_failures:
        return "fail", True
    if avg_score < threshold:
        return "fail", True
    if persistent:
        log.info(
            "quality: %d persistent hard failure(s) tolerated (avg=%.1f)",
            len(persistent), avg_score,
        )
    return "pass", False


def _run_multi_pass_pipeline(
    summaries_list: list[Any],
    disease_name: str,
    synonyms: list[str],
    ot_section: str,
    ctx: ActionContext,
    batch_label: str = "",
) -> DiseaseProfile:
    """Run the full multi-pass pipeline: audit → draft → critique → refine.

    Returns the final DiseaseProfile. Writes diagnostics to ctx.synthesis_state.
    """
    from targetsearch.core.config import config

    summaries_json = _serialize_summaries(summaries_list=summaries_list)
    evidence_index = _build_evidence_index(summaries_list)
    n_papers = len(summaries_list)
    passes = 0

    # Pass A: Evidence Audit
    ctx.synthesis_state.synthesis_stage = "audit"
    log.info("synthesis[%s]: Pass A — evidence audit (%d papers)", batch_label, n_papers)
    audit_result = _run_audit(
        summaries_json, n_papers, disease_name, synonyms, ot_section,
        batch_label=batch_label,
    )
    passes += 1

    # Store audit diagnostics
    ctx.synthesis_state.coverage_by_bucket = audit_result.get("coverage_by_bucket", {})
    ctx.synthesis_state.contradiction_notes = [
        f"{c.get('topic', '?')}: {c.get('assessment', '')}"
        for c in audit_result.get("contradictions", [])
    ]
    ctx.synthesis_state.unresolved_claims = audit_result.get("unresolved_questions", [])

    # Pass B: Draft Profile
    ctx.synthesis_state.synthesis_stage = "draft"
    log.info("synthesis[%s]: Pass B — draft profile", batch_label)
    profile = _synthesize_batch(
        summaries_list, disease_name, synonyms, ot_section,
        audit_result=audit_result,
        batch_label=batch_label,
    )
    passes += 1

    # Pass C: Quality Critique (uses compact evidence index, not full summaries)
    ctx.synthesis_state.synthesis_stage = "critique"
    log.info("synthesis[%s]: Pass C — quality critique", batch_label)
    critique = _run_critique(profile, evidence_index, n_papers, audit_result)
    passes += 1

    scores = _coerce_scores(critique.get("section_scores", {}))
    ctx.synthesis_state.quality_scores = scores
    previous_failures: list[str] = []
    quality_status, needs_refinement = _assess_quality(
        critique, config.synthesis_quality_threshold,
        previous_failures=previous_failures,
        max_new_hard_failures=config.max_new_hard_failures_for_pass,
    )
    ctx.synthesis_state.quality_status = quality_status

    current_failures = critique.get("hard_failures", [])
    if scores:
        avg = sum(scores.values()) / len(scores)
        log.info(
            "synthesis[%s]: critique scores — avg=%.1f, status=%s, "
            "hard_failures=%d (all new)",
            batch_label, avg, quality_status, len(current_failures),
        )

    # Pass D: Bounded Refinement (refine uses full summaries, re-critique uses index)
    if needs_refinement and config.synthesis_refinement_enabled:
        for i in range(config.synthesis_max_internal_passes):
            ctx.synthesis_state.synthesis_stage = "refine"
            log.info(
                "synthesis[%s]: Pass D — refinement %d/%d",
                batch_label, i + 1, config.synthesis_max_internal_passes,
            )
            profile = _run_refinement(
                profile, critique, summaries_json, n_papers,
                pass_number=i + 1,
            )
            passes += 1

            previous_failures = current_failures

            # Re-critique after refinement (compact index)
            critique = _run_critique(profile, evidence_index, n_papers, audit_result)
            passes += 1

            scores = _coerce_scores(critique.get("section_scores", {}))
            ctx.synthesis_state.quality_scores = scores
            current_failures = critique.get("hard_failures", [])
            new_failures, persistent = _classify_hard_failures(
                current_failures, previous_failures,
            )
            quality_status, still_needs = _assess_quality(
                critique, config.synthesis_quality_threshold,
                previous_failures=previous_failures,
                max_new_hard_failures=config.max_new_hard_failures_for_pass,
            )
            ctx.synthesis_state.quality_status = quality_status

            if scores:
                avg = sum(scores.values()) / len(scores)
                log.info(
                    "synthesis[%s]: post-refinement %d — avg=%.1f, status=%s, "
                    "hard_failures=%d (new=%d, persistent=%d)",
                    batch_label, i + 1, avg, quality_status,
                    len(current_failures), len(new_failures), len(persistent),
                )

            if not still_needs:
                log.info("synthesis[%s]: quality threshold met, stopping refinement", batch_label)
                break
    elif needs_refinement:
        log.info("synthesis[%s]: refinement needed but disabled by config", batch_label)

    ctx.synthesis_state.synthesis_stage = "done"
    ctx.synthesis_state.synthesis_passes_run = passes
    return profile


def _format_compact_result(profile: DiseaseProfile, ctx: ActionContext) -> str:
    """Format a compact summary string for the orchestrator.

    Includes profile shape, diagnostics snapshot, and literature summary.
    The full profile is in ctx.synthesis_state.profile.
    """
    pathway_names = ", ".join(p.name for p in profile.key_pathways) or "none"
    somatic_genes = ", ".join(
        g.gene_symbol for g in profile.somatic_genomics
    ) or "none"
    germline_genes = ", ".join(
        g.gene_symbol for g in profile.germline_genetics
    ) or "none"
    drug_names = ", ".join(
        t.drug_name for t in profile.existing_therapies
    ) or "none"
    unmet = "\n".join(f"  - {u}" for u in profile.unmet_needs) or "  none"

    sy = ctx.synthesis_state
    lines = [
        f"Synthesis complete for {profile.disease_name}.",
        "",
        f"Pathways ({len(profile.key_pathways)}): {pathway_names}",
        f"Somatic genomics ({len(profile.somatic_genomics)}): {somatic_genes}",
        f"Germline genetics ({len(profile.germline_genetics)}): {germline_genes}",
        f"Existing therapies ({len(profile.existing_therapies)}): {drug_names}",
        f"Unmet needs ({len(profile.unmet_needs)}):",
        unmet,
    ]

    # Diagnostics
    lines.append("")
    lines.append(f"Quality: {sy.quality_status or 'not assessed'}")
    if sy.quality_scores:
        scores_str = ", ".join(f"{k}: {v:.1f}" for k, v in sy.quality_scores.items())
        lines.append(f"Section scores: {scores_str}")
    if sy.contradiction_notes:
        lines.append(f"Contradictions ({len(sy.contradiction_notes)}):")
        for c in sy.contradiction_notes:
            lines.append(f"  - {c}")
    if sy.unresolved_claims:
        lines.append(f"Unresolved claims ({len(sy.unresolved_claims)}):")
        for u in sy.unresolved_claims:
            lines.append(f"  - {u}")
    lines.append(f"Internal passes: {sy.synthesis_passes_run}")

    lines.append("")
    lines.append(f"Literature summary:\n{profile.literature_summary}")

    return "\n".join(lines)


def _incremental_synthesis(
    existing_profile: DiseaseProfile,
    new_summaries: list[Any],
    all_summaries: list[Any],
    disease_name: str,
    synonyms: list[str],
    ot_section: str,
    ctx: ActionContext,
) -> DiseaseProfile:
    """Incremental synthesis: audit+draft new papers, merge, critique+refine.

    Only the new papers go through audit and draft. The merged result
    (existing + delta) goes through critique+refine with the compact
    evidence index from all papers.
    """
    from targetsearch.core.config import config

    n_new = len(new_summaries)
    n_all = len(all_summaries)
    passes = 0

    # Audit new papers only
    ctx.synthesis_state.synthesis_stage = "audit"
    log.info("incremental_synthesis: auditing %d new papers", n_new)
    new_summaries_json = _serialize_summaries(summaries_list=new_summaries)
    audit_result = _run_audit(
        new_summaries_json, n_new, disease_name, synonyms, ot_section,
        batch_label="incremental",
    )
    passes += 1

    # Draft from new papers only
    ctx.synthesis_state.synthesis_stage = "draft"
    log.info("incremental_synthesis: drafting delta profile from %d new papers", n_new)
    delta_profile = _synthesize_batch(
        new_summaries, disease_name, synonyms, ot_section,
        audit_result=audit_result,
        batch_label="incremental",
    )
    passes += 1

    # Merge delta into existing profile
    log.info("incremental_synthesis: merging delta into existing profile")
    profile = _merge_profiles([existing_profile, delta_profile], disease_name)
    passes += 1

    # Critique + refine the merged result using all papers' evidence index
    all_evidence_index = _build_evidence_index(all_summaries)
    all_summaries_json = _serialize_summaries(summaries_list=all_summaries)

    # Merge audit results: combine previous coverage with new audit
    prev_coverage = ctx.synthesis_state.coverage_by_bucket
    new_coverage = audit_result.get("coverage_by_bucket", {})
    merged_coverage = dict(prev_coverage)
    for k, v in new_coverage.items():
        merged_coverage[k] = merged_coverage.get(k, 0) + v
    merged_audit = {
        "coverage_by_bucket": merged_coverage,
        "contradictions": audit_result.get("contradictions", []),
        "unresolved_questions": audit_result.get("unresolved_questions", []),
    }

    ctx.synthesis_state.synthesis_stage = "critique"
    log.info("incremental_synthesis: critiquing merged profile (%d total papers)", n_all)
    critique = _run_critique(profile, all_evidence_index, n_all, merged_audit)
    passes += 1

    scores = _coerce_scores(critique.get("section_scores", {}))
    ctx.synthesis_state.quality_scores = scores
    previous_failures: list[str] = []
    quality_status, needs_refinement = _assess_quality(
        critique, config.synthesis_quality_threshold,
        previous_failures=previous_failures,
        max_new_hard_failures=config.max_new_hard_failures_for_pass,
    )
    ctx.synthesis_state.quality_status = quality_status
    current_failures = critique.get("hard_failures", [])

    if needs_refinement and config.synthesis_refinement_enabled:
        for i in range(config.synthesis_max_internal_passes):
            ctx.synthesis_state.synthesis_stage = "refine"
            log.info(
                "incremental_synthesis: refinement %d/%d",
                i + 1, config.synthesis_max_internal_passes,
            )
            profile = _run_refinement(
                profile, critique, all_summaries_json, n_all,
                pass_number=i + 1,
            )
            passes += 1

            previous_failures = current_failures

            critique = _run_critique(profile, all_evidence_index, n_all, merged_audit)
            passes += 1

            scores = _coerce_scores(critique.get("section_scores", {}))
            ctx.synthesis_state.quality_scores = scores
            current_failures = critique.get("hard_failures", [])
            new_failures, persistent = _classify_hard_failures(
                current_failures, previous_failures,
            )
            quality_status, still_needs = _assess_quality(
                critique, config.synthesis_quality_threshold,
                previous_failures=previous_failures,
                max_new_hard_failures=config.max_new_hard_failures_for_pass,
            )
            ctx.synthesis_state.quality_status = quality_status

            if not still_needs:
                log.info("incremental_synthesis: quality threshold met")
                break

    ctx.synthesis_state.synthesis_stage = "done"
    ctx.synthesis_state.synthesis_passes_run += passes
    return profile


@registry.tool(
    description=(
        "Synthesize all accumulated paper summaries and Open Targets data "
        "into a DiseaseProfile. Reads from ActionContext, writes the profile "
        "back to ctx.synthesis_state. Returns a compact summary for the "
        "orchestrator — the full profile is in ctx.synthesis_state.profile."
    ),
    tags=["synthesis"],
    params={},
    returns="Compact summary of the synthesized profile",
)
def synthesize_disease_profile(ctx: ActionContext) -> str:
    """Synthesize paper summaries + target data into a DiseaseProfile.

    Runs a multi-pass pipeline: audit → draft → critique → refine.
    Uses map-reduce when paper count exceeds ``config.synthesis_batch_size``.

    On re-invocation (after feedback-driven re-search), uses incremental
    synthesis: only audit+draft new papers, merge into the existing
    profile, then critique+refine the merged result.

    Returns a compact summary string for the orchestrator. The full
    profile is written to ``ctx.synthesis_state.profile``.
    """
    from targetsearch.core.config import config

    disease_name = ctx.disease_info.name or "unknown disease"
    synonyms = ctx.disease_info.synonyms
    all_summaries = list(ctx.paper_state.summaries)
    n_papers = len(all_summaries)
    batch_size = config.synthesis_batch_size

    # Precondition: paper summaries must exist
    if n_papers == 0:
        log.warning("synthesize_disease_profile: called with 0 paper summaries")
        return (
            "ERROR: No paper summaries available. The search agent must call "
            "batch_summarize_papers before synthesis can run. Call "
            "run_search_agent first, then retry synthesize_disease_profile."
        )

    ot_section = json.dumps(
        ctx.target_state.opentargets_results,
        indent=2,
        default=str,
    )

    # Check for incremental synthesis
    existing_profile = ctx.synthesis_state.profile
    previously_synthesized = ctx.synthesis_state.synthesized_pmids

    if existing_profile and previously_synthesized:
        new_summaries = [
            s for s in all_summaries
            if getattr(s, "pmid", None) not in previously_synthesized
        ]
        if not new_summaries:
            log.info("synthesize_disease_profile: no new papers, returning existing profile")
            return _format_compact_result(existing_profile, ctx)

        log.info(
            "synthesize_disease_profile: incremental — %d new papers "
            "(of %d total), merging into existing profile",
            len(new_summaries), n_papers,
        )
        profile = _incremental_synthesis(
            existing_profile, new_summaries, all_summaries,
            disease_name, synonyms, ot_section, ctx,
        )

        # Record all PMIDs as synthesized
        ctx.synthesis_state.synthesized_pmids = {
            getattr(s, "pmid", "") for s in all_summaries
        }
        profile.paper_summaries = all_summaries
        ctx.synthesis_state.has_been_run = True
        ctx.synthesis_state.profile = profile
        return _format_compact_result(profile, ctx)

    log.info(
        "synthesize_disease_profile: %s, %d papers, %d OT targets",
        disease_name,
        n_papers,
        len(ctx.target_state.opentargets_results),
    )

    if n_papers <= batch_size:
        # Single-batch: run full multi-pass pipeline
        profile = _run_multi_pass_pipeline(
            all_summaries, disease_name, synonyms, ot_section, ctx,
            batch_label="single",
        )
    else:
        # Map-reduce: audit + draft each batch, merge, then
        # critique + refine the merged profile
        batches = [
            all_summaries[i:i + batch_size]
            for i in range(0, n_papers, batch_size)
        ]
        log.info(
            "synthesize_disease_profile: map-reduce with %d batches of ~%d papers",
            len(batches),
            batch_size,
        )

        # Per-batch audit
        batch_audits = []
        for idx, batch in enumerate(batches):
            label = f"{idx + 1}/{len(batches)}"
            log.info("synthesize_disease_profile: auditing batch %s (%d papers)", label, len(batch))
            summaries_json = _serialize_summaries(summaries_list=batch)
            audit = _run_audit(
                summaries_json, len(batch), disease_name, synonyms, ot_section,
                batch_label=label,
            )
            batch_audits.append(audit)

        # Merge audits
        merged_audit = _merge_audits(batch_audits)
        ctx.synthesis_state.coverage_by_bucket = merged_audit.get("coverage_by_bucket", {})
        ctx.synthesis_state.contradiction_notes = [
            f"{c.get('topic', '?')}: {c.get('assessment', '')}"
            for c in merged_audit.get("contradictions", [])
        ]
        ctx.synthesis_state.unresolved_claims = merged_audit.get("unresolved_questions", [])

        # Per-batch draft (with merged audit context)
        intermediate_profiles = []
        for idx, batch in enumerate(batches):
            label = f"{idx + 1}/{len(batches)}"
            log.info("synthesize_disease_profile: drafting batch %s (%d papers)", label, len(batch))
            p = _synthesize_batch(
                batch, disease_name, synonyms, ot_section,
                audit_result=merged_audit,
                batch_label=label,
            )
            intermediate_profiles.append(p)

        # Merge intermediate profiles
        log.info("synthesize_disease_profile: merging %d intermediate profiles", len(intermediate_profiles))
        profile = _merge_profiles(intermediate_profiles, disease_name)

        # Critique + refine the merged profile
        all_summaries_json = _serialize_summaries(summaries_list=all_summaries)
        all_evidence_index = _build_evidence_index(all_summaries)

        ctx.synthesis_state.synthesis_stage = "critique"
        log.info("synthesize_disease_profile: critiquing merged profile")
        critique = _run_critique(profile, all_evidence_index, n_papers, merged_audit)

        scores = _coerce_scores(critique.get("section_scores", {}))
        ctx.synthesis_state.quality_scores = scores
        previous_failures: list[str] = []
        quality_status, needs_refinement = _assess_quality(
            critique, config.synthesis_quality_threshold,
            previous_failures=previous_failures,
            max_new_hard_failures=config.max_new_hard_failures_for_pass,
        )
        ctx.synthesis_state.quality_status = quality_status
        current_failures = critique.get("hard_failures", [])

        passes = len(batches) * 2 + 2  # audits + drafts + merge + critique

        if needs_refinement and config.synthesis_refinement_enabled:
            for i in range(config.synthesis_max_internal_passes):
                ctx.synthesis_state.synthesis_stage = "refine"
                log.info(
                    "synthesize_disease_profile: refinement %d/%d",
                    i + 1, config.synthesis_max_internal_passes,
                )
                profile = _run_refinement(
                    profile, critique, all_summaries_json, n_papers,
                    pass_number=i + 1,
                )
                passes += 1

                previous_failures = current_failures

                critique = _run_critique(profile, all_evidence_index, n_papers, merged_audit)
                passes += 1

                scores = _coerce_scores(critique.get("section_scores", {}))
                ctx.synthesis_state.quality_scores = scores
                current_failures = critique.get("hard_failures", [])
                new_failures, persistent = _classify_hard_failures(
                    current_failures, previous_failures,
                )
                quality_status, still_needs = _assess_quality(
                    critique, config.synthesis_quality_threshold,
                    previous_failures=previous_failures,
                    max_new_hard_failures=config.max_new_hard_failures_for_pass,
                )
                ctx.synthesis_state.quality_status = quality_status

                if not still_needs:
                    if persistent:
                        log.info(
                            "synthesize_disease_profile: quality threshold met "
                            "(%d persistent failures tolerated)",
                            len(persistent),
                        )
                    else:
                        log.info("synthesize_disease_profile: quality threshold met")
                    break

        ctx.synthesis_state.synthesis_stage = "done"
        ctx.synthesis_state.synthesis_passes_run = passes

    # Attach paper summaries to the profile
    profile.paper_summaries = all_summaries

    # Record which PMIDs were synthesized (for incremental synthesis)
    ctx.synthesis_state.synthesized_pmids = {
        getattr(s, "pmid", "") for s in all_summaries
    }

    # Write back to context
    ctx.synthesis_state.has_been_run = True
    ctx.synthesis_state.profile = profile

    return _format_compact_result(profile, ctx)
