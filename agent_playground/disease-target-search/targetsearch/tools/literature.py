"""Literature search tools — PubMed and Semantic Scholar.

These tools wrap public REST APIs to search for and retrieve research papers.
They return plain dicts so the LLM can reason over the results directly.

PubMed E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
Semantic Scholar API docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET

import httpx

from targetsearch.core.config import config
from targetsearch.core.registry import registry

log = logging.getLogger(__name__)

_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_S2_BASE = "https://api.semanticscholar.org/graph/v1"

# Retry settings — mirrors the NCBI wrapper in fulltext.py
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 3.0, 6.0]


def _http_get_with_retry(
    url: str,
    params: dict,
    headers: dict | None = None,
    timeout: float | None = None,
) -> httpx.Response:
    """HTTP GET with retry on 429 and 5xx errors.

    Waits with exponential backoff before retrying.  Non-retryable 4xx
    errors are raised immediately.
    """
    timeout = timeout or config.request_timeout
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF[attempt]
                    log.warning(
                        "%d for %s, retrying in %.1fs (attempt %d/%d)",
                        resp.status_code,
                        url.split("/")[-1],
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF[attempt]
                log.warning(
                    "Request failed: %s, retrying in %.1fs (attempt %d/%d)",
                    e,
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------


@registry.tool(
    description=(
        "Search PubMed for biomedical literature. Returns article metadata "
        "including PMID, title, abstract, authors, journal, and publication year. "
        "Use specific MeSH terms or Boolean queries for best results."
    ),
    tags=["literature", "pubmed"],
    cache=True,
    params={
        "query": "PubMed search query (supports Boolean operators and MeSH terms)",
        "max_results": "Maximum number of articles to return (default 5, max 5)",
    },
    returns="List of dicts with keys: pmid, title, abstract, authors, journal, year",
)
def pubmed_search(query: str, max_results: int = 5) -> list[dict]:
    """Search PubMed and return structured article metadata.

    Two-step process:
      1. esearch — get PMIDs matching the query
      2. efetch  — get full metadata for those PMIDs
    """
    max_results = min(max_results, 5)

    # Step 1: Search for PMIDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "sort": "relevance",
        "email": config.ncbi_email,
    }
    if config.ncbi_api_key:
        search_params["api_key"] = config.ncbi_api_key

    resp = _http_get_with_retry(
        f"{_PUBMED_BASE}/esearch.fcgi",
        params=search_params,
    )
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])

    if not id_list:
        return []

    # Step 2: Fetch article details
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        "email": config.ncbi_email,
    }
    if config.ncbi_api_key:
        fetch_params["api_key"] = config.ncbi_api_key

    resp = _http_get_with_retry(
        f"{_PUBMED_BASE}/efetch.fcgi",
        params=fetch_params,
    )

    return _parse_pubmed_xml(resp.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed efetch XML into a list of article dicts."""
    root = ET.fromstring(xml_text)
    articles = []

    for article_el in root.findall(".//PubmedArticle"):
        medline = article_el.find("MedlineCitation")
        if medline is None:
            continue

        pmid_el = medline.find("PMID")
        art = medline.find("Article")
        if art is None:
            continue

        # Title
        title_el = art.find("ArticleTitle")
        title = _text(title_el)

        # Abstract — may have multiple AbstractText elements (structured abstract)
        abstract_parts = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for at in abstract_el.findall("AbstractText"):
                label = at.get("Label", "")
                text = _text(at)
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author_el in author_list.findall("Author"):
                last = _text(author_el.find("LastName"))
                fore = _text(author_el.find("ForeName"))
                if last:
                    authors.append(f"{last} {fore}".strip())

        # Journal
        journal_el = art.find("Journal/Title")
        journal = _text(journal_el)

        # Year — try multiple locations
        year = None
        for year_path in [
            "Journal/JournalIssue/PubDate/Year",
            "Journal/JournalIssue/PubDate/MedlineDate",
        ]:
            year_el = art.find(year_path)
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text[:4])
                except ValueError:
                    pass
                break

        articles.append({
            "pmid": _text(pmid_el),
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
        })

    return articles


def _text(el: ET.Element | None) -> str:
    """Extract text from an XML element, handling mixed content."""
    if el is None:
        return ""
    # itertext() captures text from child elements too (e.g. <i>, <sup>)
    return "".join(el.itertext()).strip()


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


@registry.tool(
    description=(
        "Search Semantic Scholar for academic papers. Returns paper metadata "
        "including title, abstract, authors, year, citation count, and external IDs. "
        "Good for finding highly-cited papers and exploring citation graphs."
    ),
    tags=["literature", "semantic_scholar"],
    cache=True,
    params={
        "query": "Search query (natural language or keywords)",
        "max_results": "Maximum number of papers to return (default 5, max 5)",
    },
    returns="List of dicts with keys: paper_id, title, abstract, authors, year, citation_count, doi, pmid",
)
def semantic_scholar_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Semantic Scholar and return structured paper metadata."""
    max_results = min(max_results, 5)

    headers = {}
    if config.s2_api_key:
        headers["x-api-key"] = config.s2_api_key

    resp = _http_get_with_retry(
        f"{_S2_BASE}/paper/search",
        params={
            "query": query,
            "limit": str(max_results),
            "fields": "title,abstract,authors,year,citationCount,externalIds",
        },
        headers=headers or None,
    )
    data = resp.json()

    papers = []
    for paper in data.get("data", []):
        ext_ids = paper.get("externalIds") or {}
        authors = [
            a.get("name", "") for a in (paper.get("authors") or [])
        ]
        papers.append({
            "paper_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract") or "",
            "authors": authors,
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount", 0),
            "doi": ext_ids.get("DOI"),
            "pmid": ext_ids.get("PubMed"),
        })

    return papers
