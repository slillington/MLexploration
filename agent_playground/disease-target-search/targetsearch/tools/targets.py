"""Target database tools — Open Targets Platform.

Open Targets uses a GraphQL API. We send POST requests with GraphQL queries
and parse the JSON responses into flat dicts the LLM can reason over.

API docs: https://platform-docs.opentargets.org/data-access/graphql-api
"""

from __future__ import annotations

import logging

import httpx

from targetsearch.core.config import config
from targetsearch.core.registry import registry

log = logging.getLogger(__name__)

_OT_BASE = "https://api.platform.opentargets.org/api/v4/graphql"


def _ot_query(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against Open Targets."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    resp = httpx.post(
        _OT_BASE,
        json=payload,
        timeout=config.request_timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Disease → associated targets
# ---------------------------------------------------------------------------


@registry.tool(
    description=(
        "Search Open Targets for targets associated with a disease. "
        "Accepts a disease name or EFO ID and returns targets ranked by "
        "association score, including genetic and literature evidence."
    ),
    tags=["targets", "open_targets", "disease"],
    cache=True,
    params={
        "disease_query": "Disease name (e.g. 'pulmonary fibrosis') or EFO ID (e.g. 'EFO_0000768')",
        "max_results": "Maximum number of associated targets to return (default 20)",
    },
    returns="Dict with keys: disease_id, disease_name, targets (list of dicts with gene_symbol, score, evidence_types)",
)
def opentargets_disease_targets(
    disease_query: str, max_results: int = 20
) -> dict:
    """Find targets associated with a disease via Open Targets.

    If disease_query looks like an EFO ID (starts with EFO/MONDO/etc.),
    query directly. Otherwise, search by name first.
    """
    # Resolve disease name to ID if needed
    disease_id = disease_query
    if not any(disease_query.upper().startswith(p) for p in ["EFO_", "MONDO_", "OTAR_", "HP_", "ORPHANET_"]):
        disease_id = _search_disease_id(disease_query)
        if not disease_id:
            return {"disease_id": None, "disease_name": disease_query, "targets": [],
                    "error": f"No disease found for '{disease_query}'"}

    # Query associated targets
    query = """
    query DiseaseTargets($diseaseId: String!, $size: Int!) {
      disease(efoId: $diseaseId) {
        id
        name
        description
        associatedTargets(page: {size: $size, index: 0}) {
          rows {
            target {
              id
              approvedSymbol
              approvedName
              biotype
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """
    data = _ot_query(query, {"diseaseId": disease_id, "size": max_results})
    disease_data = data.get("data", {}).get("disease")
    if not disease_data:
        return {"disease_id": disease_id, "disease_name": disease_query, "targets": [],
                "error": "Disease ID not found in Open Targets"}

    targets = []
    for row in disease_data.get("associatedTargets", {}).get("rows", []):
        target = row.get("target", {})
        datatype_scores = {
            dt["id"]: round(dt["score"], 3)
            for dt in row.get("datatypeScores", [])
        }
        targets.append({
            "ensembl_id": target.get("id", ""),
            "gene_symbol": target.get("approvedSymbol", ""),
            "gene_name": target.get("approvedName", ""),
            "biotype": target.get("biotype", ""),
            "overall_score": round(row.get("score", 0), 3),
            "evidence_scores": datatype_scores,
        })

    return {
        "disease_id": disease_data.get("id", disease_id),
        "disease_name": disease_data.get("name", disease_query),
        "description": disease_data.get("description", ""),
        "targets": targets,
    }


def _search_disease_id(name: str) -> str | None:
    """Search Open Targets for a disease by name and return its EFO ID."""
    query = """
    query SearchDisease($queryString: String!) {
      search(queryString: $queryString, entityNames: ["disease"]) {
        hits {
          id
          name
          entity
        }
      }
    }
    """
    data = _ot_query(query, {"queryString": name})
    hits = data.get("data", {}).get("search", {}).get("hits", [])
    for hit in hits:
        if hit.get("entity") == "disease":
            return hit["id"]
    return hits[0]["id"] if hits else None


# ---------------------------------------------------------------------------
# Target → known drugs
# ---------------------------------------------------------------------------


@registry.tool(
    description=(
        "Look up known drugs for a target in Open Targets. "
        "Returns approved and clinical-stage drugs with their mechanisms, "
        "indications, and clinical trial phases."
    ),
    tags=["targets", "open_targets", "drugs"],
    cache=True,
    params={
        "ensembl_id": "Ensembl gene ID (e.g. 'ENSG00000146648' for EGFR)",
    },
    returns="List of dicts with keys: drug_name, molecule_type, mechanism, phase, indications",
)
def opentargets_target_drugs(ensembl_id: str) -> list[dict]:
    """Get known drugs targeting a gene from Open Targets."""
    query = """
    query TargetDrugs($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        knownDrugs(size: 25) {
          rows {
            drug {
              name
              drugType
              mechanismOfAction
              isApproved
            }
            phase
            disease {
              name
            }
          }
        }
      }
    }
    """
    data = _ot_query(query, {"ensemblId": ensembl_id})
    target_data = data.get("data", {}).get("target")
    if not target_data:
        return []

    drugs = []
    seen = set()
    for row in target_data.get("knownDrugs", {}).get("rows", []):
        drug = row.get("drug", {})
        drug_name = drug.get("name", "")
        # Deduplicate by drug name
        if drug_name.lower() in seen:
            continue
        seen.add(drug_name.lower())

        disease = row.get("disease", {})
        drugs.append({
            "drug_name": drug_name,
            "molecule_type": drug.get("drugType", ""),
            "mechanism": drug.get("mechanismOfAction", ""),
            "is_approved": drug.get("isApproved", False),
            "max_phase": row.get("phase", 0),
            "indication": disease.get("name", ""),
        })

    return drugs
