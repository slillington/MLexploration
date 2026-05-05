"""Ontology tools — Disease Ontology and Gene Ontology lookups.

These tools query the OLS (Ontology Lookup Service) at EMBL-EBI, which
provides a unified REST API over hundreds of ontologies including:
  - MONDO (Monarch Disease Ontology)
  - DO (Disease Ontology)
  - GO (Gene Ontology)
  - HP (Human Phenotype Ontology)

OLS API docs: https://www.ebi.ac.uk/ols4/api
"""

from __future__ import annotations

import logging

import httpx

from targetsearch.core.config import config
from targetsearch.core.registry import registry

log = logging.getLogger(__name__)

_OLS_BASE = "https://www.ebi.ac.uk/ols4/api"


@registry.tool(
    description=(
        "Search for a disease in the MONDO ontology. Returns the disease "
        "definition, synonyms, parent/child relationships, and cross-references "
        "to other ontologies (EFO, DOID, OMIM, Orphanet). Useful for "
        "understanding disease classification and finding related conditions."
    ),
    tags=["ontology", "disease"],
    cache=True,
    params={
        "disease_name": "Disease name to search for (e.g. 'idiopathic pulmonary fibrosis')",
        "max_results": "Maximum number of results (default 5)",
    },
    returns="List of dicts with keys: id, label, description, synonyms, xrefs, parents",
)
def disease_ontology_search(disease_name: str, max_results: int = 5) -> list[dict]:
    """Search MONDO for a disease and return ontology information."""
    resp = httpx.get(
        f"{_OLS_BASE}/search",
        params={
            "q": disease_name,
            "ontology": "mondo",
            "rows": str(max_results),
            "exact": "false",
        },
        timeout=config.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for doc in data.get("response", {}).get("docs", []):
        # Extract cross-references from annotation
        annotation = doc.get("annotation", {})
        xrefs = annotation.get("database_cross_reference", [])

        results.append({
            "id": doc.get("obo_id", doc.get("short_form", "")),
            "label": doc.get("label", ""),
            "description": (doc.get("description") or [""])[0],
            "synonyms": doc.get("synonyms", []),
            "xrefs": xrefs,
            "ontology": doc.get("ontology_name", ""),
            "is_defining_ontology": doc.get("is_defining_ontology", False),
        })

    return results


@registry.tool(
    description=(
        "Look up Gene Ontology (GO) terms for a gene. Returns biological "
        "processes, molecular functions, and cellular components associated "
        "with the gene. Useful for understanding what a potential target does."
    ),
    tags=["ontology", "gene"],
    cache=True,
    params={
        "gene_symbol": "HGNC gene symbol (e.g. 'EGFR', 'TP53')",
    },
    returns="Dict with keys: gene_symbol, go_terms (list of dicts with id, name, aspect, evidence)",
)
def gene_ontology_lookup(gene_symbol: str) -> dict:
    """Look up GO annotations for a gene via QuickGO.

    QuickGO API: https://www.ebi.ac.uk/QuickGO/api
    """
    resp = httpx.get(
        "https://www.ebi.ac.uk/QuickGO/services/annotation/search",
        params={
            "geneProductId": gene_symbol,
            "geneProductType": "protein",
            "taxonId": "9606",  # Human
            "limit": "50",
        },
        headers={"Accept": "application/json"},
        timeout=config.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    go_terms = []
    seen = set()
    for result in data.get("results", []):
        go_id = result.get("goId", "")
        if go_id in seen:
            continue
        seen.add(go_id)

        go_terms.append({
            "id": go_id,
            "name": result.get("goName", ""),
            "aspect": result.get("goAspect", ""),  # biological_process, molecular_function, cellular_component
            "evidence": result.get("goEvidence", ""),
            "qualifier": result.get("qualifier", ""),
        })

    return {
        "gene_symbol": gene_symbol,
        "go_terms": go_terms,
    }
