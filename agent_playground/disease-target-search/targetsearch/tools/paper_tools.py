"""Paper processing tools — stateless leaf tools for paper summarization and review mining.

These replace PaperAgent and ReviewMinerAgent. They are leaf tools (no
ActionContext) — the coordination tool batch_summarize_papers handles
writing results to context.
"""

from __future__ import annotations

import logging
from typing import Any

from targetsearch.core.config import config
from targetsearch.core.llm import llm_text, parse_json_response
from targetsearch.core.registry import registry
from targetsearch.schemas.paper import PaperSummary
from targetsearch.tools.prompt_tools import (
    create_expert_persona,
    create_extraction_guidelines,
    create_output_schema,
)

log = logging.getLogger(__name__)

_MAX_PAPER_CHARS = 50_000


def _build_paper_header(metadata: dict[str, Any]) -> str:
    """Build a header block from paper metadata."""
    parts = []
    if metadata.get("title"):
        parts.append(f"Title: {metadata['title']}")
    if metadata.get("authors"):
        authors = metadata["authors"]
        if isinstance(authors, list):
            authors = ", ".join(authors[:5])
            if len(metadata["authors"]) > 5:
                authors += " et al."
        parts.append(f"Authors: {authors}")
    if metadata.get("journal"):
        parts.append(f"Journal: {metadata['journal']}")
    if metadata.get("year"):
        parts.append(f"Year: {metadata['year']}")
    if metadata.get("pmid"):
        parts.append(f"PMID: {metadata['pmid']}")
    source_type = metadata.get("source_type", "abstract")
    parts.append(f"Source: {source_type}")
    return "\n".join(parts)


@registry.tool(
    description=(
        "Summarize a single research paper into a structured PaperSummary. "
        "Accepts paper text and metadata, returns extracted findings as a dict."
    ),
    tags=["paper"],
    params={
        "paper_text": "Full text or abstract of the paper",
        "disease_area": "Disease being investigated (e.g. 'idiopathic pulmonary fibrosis')",
        "metadata": "Dict with keys: pmid, doi, title, authors, year, journal, source_type",
    },
    returns="PaperSummary as a dict",
)
def summarize_paper(
    paper_text: str,
    disease_area: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize a paper into a PaperSummary.

    Stateless leaf tool — no ActionContext. Called in parallel by
    batch_summarize_papers.
    """
    metadata = metadata or {}
    source_type = metadata.get("source_type", "abstract")

    # Truncate long papers
    if len(paper_text) > _MAX_PAPER_CHARS:
        paper_text = paper_text[:_MAX_PAPER_CHARS] + "\n\n[... text truncated ...]"

    # Build prompt dynamically using prompt tools
    persona = create_expert_persona(disease_area, ["drug target discovery"])
    schema = create_output_schema("paper_summary")
    guidelines = create_extraction_guidelines(
        "review" if "review" in source_type.lower() else source_type
    )

    system_prompt = f"""{persona}

## Your task

Read the paper text and produce a structured JSON summary.

## Extraction rules

1. **Classify the domain first.** Determine whether this is preclinical
   biology, clinical/translational, computational, or epidemiological.
   Apply domain-appropriate extraction logic — a cell-line IC50 is not
   the same kind of evidence as a clinical ORR.

2. **Report observations, not interpretations.** Extract what was
   directly measured or observed. If the authors speculate about
   mechanism or clinical relevance beyond their data, do not include
   that speculation as a finding.

3. **Leave fields empty when data is missing.** If effect_size is not
   reported, leave it as an empty string. Do not write "Not reported"
   or invent approximate values.

4. **Flag cross-disease evidence.** If the paper studies a different
   disease (e.g., breast cancer) but is being summarized for
   {disease_area}, note the disease context in the finding text.

5. **Constrain target_relevance to extracted evidence.** Only state
   target relevance that is directly supported by the paper's own
   findings. Do not add general knowledge about the target.

6. **Use correct HGNC gene symbols.** If the paper uses a non-standard
   name, include both (e.g., "TROP2/TACSTD2").

7. **Assess evidence_strength** based on study design, sample size,
   controls, and reproducibility:
   - strong: large clinical cohort, randomized trial, or multiple
     independent preclinical validations
   - moderate: single well-controlled preclinical study, or small
     clinical cohort with clear signal
   - weak: single cell-line experiment, no controls described, or
     very small sample size
   - insufficient: abstract-only with no methods detail, or
     commentary/opinion without original data

{schema}

{guidelines}"""

    header = _build_paper_header(metadata)
    user_message = f"{header}\n\n---\n\n{paper_text}"

    # For reviews, append the structured reference list as a PMID whitelist
    if "review" in source_type.lower() and metadata.get("references"):
        refs = metadata["references"]
        ref_block = (
            "\n\nREFERENCE LIST (valid PMIDs from this article's bibliography):\n"
            + ", ".join(str(r) for r in refs)
            + "\n\nOnly cite PMIDs from this list. If a finding's source is "
            "not in this list, describe the finding without a PMID citation."
        )
        user_message += ref_block

    log.info(
        "summarize_paper: PMID %s (%s, %d chars)",
        metadata.get("pmid", "?"),
        source_type,
        len(paper_text),
    )

    raw = llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        caller=f"summarize_paper[{metadata.get('pmid', '?')}]",
        model=config.summarization_model,
    )

    return _parse_paper_summary(raw, metadata)


def _parse_paper_summary(raw: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Parse LLM output into a PaperSummary dict, merging metadata."""
    try:
        data = parse_json_response(raw)
    except Exception as e:
        log.error(
            "Failed to parse PaperSummary for PMID %s: %s",
            metadata.get("pmid", "?"),
            e,
        )
        data = {}

    # Merge metadata — these come from PubMed, not the LLM
    data.setdefault("pmid", metadata.get("pmid"))
    data.setdefault("doi", metadata.get("doi"))
    data.setdefault("title", metadata.get("title", ""))
    data.setdefault("authors", metadata.get("authors", []))
    data.setdefault("year", metadata.get("year"))
    data.setdefault("journal", metadata.get("journal", ""))
    data["source_type"] = metadata.get("source_type", "abstract")

    # Validate through Pydantic and return as dict
    summary = PaperSummary.model_validate(data)
    return summary.model_dump()


@registry.tool(
    description=(
        "Extract relevant cited PMIDs from a review article. "
        "Returns a list of cited papers with priorities and PMIDs."
    ),
    tags=["paper"],
    params={
        "review_text": "Full text of the review article",
        "disease_area": "Disease being investigated",
        "structured_refs": "PMIDs already extracted from PMC XML reference list (optional)",
    },
    returns="Dict with review_title and cited_papers list",
)
def mine_review_references(
    review_text: str,
    disease_area: str,
    structured_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Extract relevant cited papers from a review.

    Stateless leaf tool — no ActionContext.
    """
    # Truncate long reviews
    if len(review_text) > _MAX_PAPER_CHARS:
        review_text = review_text[:_MAX_PAPER_CHARS] + "\n\n[... text truncated ...]"

    # Build prompt dynamically
    persona = create_expert_persona(disease_area, ["drug target discovery", "literature analysis"])
    schema = create_output_schema("review_mining")
    guidelines = create_extraction_guidelines("review")

    system_prompt = f"{persona}\n\n## Your task\n\nRead the review text and extract a list of the primary source papers it cites that are most relevant to drug target discovery for {disease_area}.\n\n{schema}\n\n{guidelines}"

    # Append structured references if available
    user_message = review_text
    if structured_refs:
        ref_block = "\n\nREFERENCE LIST (PMIDs from this article's bibliography):\n"
        ref_block += ", ".join(structured_refs)
        user_message += ref_block

    log.info(
        "mine_review_references: %d chars, %d structured refs",
        len(review_text),
        len(structured_refs or []),
    )

    raw = llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        caller="mine_review_references",
        model=config.summarization_model,
    )

    return _parse_review_mining(raw)


def _parse_review_mining(raw: str) -> dict[str, Any]:
    """Parse LLM output into a review mining result."""
    try:
        data = parse_json_response(raw)
    except Exception as e:
        log.error("Failed to parse review mining result: %s", e)
        return {"review_title": "", "cited_papers": []}

    # Normalize PMIDs
    cited = []
    for paper in data.get("cited_papers", []):
        pmid = paper.get("pmid")
        if pmid is not None:
            pmid = str(pmid).strip()
            if not pmid.isdigit():
                pmid = None
        cited.append({
            "pmid": pmid,
            "description": paper.get("description", ""),
            "relevance": paper.get("relevance", ""),
            "priority": paper.get("priority", "medium"),
        })

    return {
        "review_title": data.get("review_title", ""),
        "cited_papers": cited,
    }
