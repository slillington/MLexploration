"""Full-text retrieval tools — PMC Open Access and PubMed metadata.

PMC provides free full-text XML for open-access articles. For papers not
in PMC, we fall back to the abstract from PubMed. The pipeline uses these
tools to get the best available text for each paper before handing it to
the PaperAgent.

PMC OA docs: https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
ID converter: https://www.ncbi.nlm.nih.gov/pmc/pmctopmid/
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET

import httpx

from targetsearch.core.config import config
from targetsearch.core.registry import registry

log = logging.getLogger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Retry settings for NCBI API calls
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 3.0, 6.0]  # seconds to wait after 1st, 2nd, 3rd failure


def _ncbi_get(url: str, params: dict, timeout: float | None = None) -> httpx.Response:
    """HTTP GET with retry on 429 (Too Many Requests) and 5xx errors.

    NCBI rate-limits to 3 req/s without an API key, 10 req/s with one.
    On 429 or server errors, waits with exponential backoff and retries.
    """
    timeout = timeout or config.request_timeout
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = httpx.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF[attempt]
                    log.warning(
                        "NCBI %d for %s, retrying in %.1fs (attempt %d/%d)",
                        resp.status_code,
                        url.split("/")[-1],
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                # Final attempt — raise
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError:
            # Non-retryable status errors (4xx other than 429) — the
            # retryable statuses (429, 5xx) are handled above before
            # raise_for_status() is called, so anything reaching here
            # is a genuine client error.
            raise
        except httpx.HTTPError as e:
            # Retryable transport errors (connection refused, timeout, etc.)
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF[attempt]
                log.warning(
                    "NCBI request failed: %s, retrying in %.1fs (attempt %d/%d)",
                    e,
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise

    raise last_exc  # should not reach here, but satisfies type checker
_PMC_IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"


# ---------------------------------------------------------------------------
# PMID → PMC ID conversion
# ---------------------------------------------------------------------------


def pmids_to_pmcids(pmids: list[str]) -> dict[str, str | None]:
    """Convert a list of PMIDs to PMC IDs via the NCBI ID converter.

    Returns a dict mapping PMID → PMCID (or None if not in PMC).
    Handles batches — the API accepts comma-separated IDs.
    """
    if not pmids:
        return {}

    result: dict[str, str | None] = {p: None for p in pmids}

    # API accepts up to 200 IDs per request
    for i in range(0, len(pmids), 200):
        batch = pmids[i : i + 200]
        resp = _ncbi_get(
            _PMC_IDCONV,
            params={
                "ids": ",".join(batch),
                "format": "json",
                "tool": "targetsearch",
                "email": config.ncbi_email,
            },
        )
        data = resp.json()

        for record in data.get("records", []):
            pmid = str(record.get("pmid", record.get("requested-id", "")))
            pmcid = record.get("pmcid")
            if pmid in result and pmcid:
                result[pmid] = pmcid

    return result


# ---------------------------------------------------------------------------
# PMC full-text fetch
# ---------------------------------------------------------------------------


@registry.tool(
    description=(
        "Fetch the full text of a paper from PMC Open Access. "
        "Returns the body text as plain text extracted from PMC XML. "
        "If the paper is not in PMC, returns None."
    ),
    tags=["fulltext", "pipeline"],
    cache=True,
    params={
        "pmcid": "PMC ID (e.g. 'PMC8142468')",
    },
    returns="Dict with keys: pmcid, body_text, references (list of cited PMIDs)",
)
def pmc_fulltext_fetch(pmcid: str) -> dict | None:
    """Fetch full text and references from a PMC article."""
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "xml",
        "email": config.ncbi_email,
    }
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key

    try:
        resp = _ncbi_get(f"{_EUTILS_BASE}/efetch.fcgi", params=params)
    except httpx.HTTPError as e:
        log.warning("Failed to fetch PMC %s: %s", pmcid, e)
        return None

    return _parse_pmc_xml(resp.text, pmcid)


def _parse_pmc_xml(xml_text: str, pmcid: str) -> dict | None:
    """Parse PMC XML into body text and cited PMIDs.

    Uses section-aware extraction to prioritize Results/Discussion over
    figure captions and supplementary content. Falls back to flat
    paragraph extraction for articles without <sec> structure.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.warning("Failed to parse XML for %s", pmcid)
        return None

    body = root.find(".//body")
    if body is None:
        return None

    # Try section-aware extraction first
    sections = _extract_sections(body)
    if sections:
        body_text = _prioritize_and_assemble(sections)
    else:
        # Fallback: flat paragraph extraction (no <sec> structure)
        paragraphs = []
        for p in body.findall(".//p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)
        if not paragraphs:
            return None
        body_text = "\n\n".join(paragraphs)

    if not body_text.strip():
        return None

    # Extract cited PMIDs from the reference list
    cited_pmids = _extract_reference_pmids(root)

    return {
        "pmcid": pmcid,
        "body_text": body_text,
        "references": cited_pmids,
    }


# Section priority: lower number = higher priority
_SECTION_PRIORITY: dict[str, int] = {
    "results": 1,
    "findings": 1,
    "results and discussion": 1,
    "discussion": 2,
    "introduction": 3,
    "background": 3,
    "methods": 4,
    "materials and methods": 4,
    "experimental": 4,
    "experimental procedures": 4,
    "study design": 4,
    "patients and methods": 4,
    "conclusions": 5,
    "conclusion": 5,
    "summary": 5,
}

# Sections to skip entirely — low value, waste of budget
_SKIP_SECTIONS = {
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "acknowledgement",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "declaration of competing interest",
    "declaration of interest",
    "funding",
    "funding sources",
    "author contributions",
    "authors' contributions",
    "credit authorship contribution statement",
    "data availability",
    "data availability statement",
    "supplementary material",
    "supplementary data",
    "supporting information",
    "abbreviations",
    "ethics",
    "ethics statement",
    "ethical approval",
}


def _classify_section(title: str) -> int | None:
    """Map a section title to a priority number, or None to skip.

    Returns None for sections that should be excluded entirely.
    """
    normalized = title.strip().lower()

    if normalized in _SKIP_SECTIONS:
        return None

    if normalized in _SECTION_PRIORITY:
        return _SECTION_PRIORITY[normalized]

    # Partial matching for common variants
    for key, priority in _SECTION_PRIORITY.items():
        if key in normalized or normalized in key:
            return priority

    # Check skip patterns
    for skip in _SKIP_SECTIONS:
        if skip in normalized or normalized in skip:
            return None

    # Unknown section — assign priority 6 (after conclusions, before unnamed)
    return 6


def _extract_sections(body: ET.Element) -> list[tuple[str, str]]:
    """Extract named sections from a PMC <body> element.

    Returns a list of (section_title, section_text) tuples.
    Nested <sec> elements are flattened under their parent section name.
    Returns an empty list if the body has no <sec> children (triggers
    fallback to flat extraction).
    """
    top_level_secs = [child for child in body if child.tag == "sec"]
    if not top_level_secs:
        return []

    sections: list[tuple[str, str]] = []

    for sec in top_level_secs:
        title_el = sec.find("title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Collect all paragraphs in this section (including nested subsections)
        paragraphs = []
        for p in sec.findall(".//p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        if paragraphs:
            sections.append((title, "\n\n".join(paragraphs)))

    return sections


def _prioritize_and_assemble(
    sections: list[tuple[str, str]],
    char_limit: int = 50_000,
) -> str:
    """Assemble sections in priority order within a character budget.

    High-value sections (Results, Discussion) are included first.
    Low-value sections (figure captions, supplementary) are included
    only if budget remains.
    """
    # Classify and sort by priority
    classified: list[tuple[int, str, str]] = []
    for title, text in sections:
        priority = _classify_section(title) if title else 7  # unnamed = lowest
        if priority is None:
            continue  # skip excluded sections
        classified.append((priority, title, text))

    classified.sort(key=lambda x: x[0])

    # Assemble within budget
    parts: list[str] = []
    total_chars = 0

    for priority, title, text in classified:
        header = f"## {title}\n\n" if title else ""
        section_text = header + text
        section_len = len(section_text)

        if total_chars + section_len <= char_limit:
            parts.append(section_text)
            total_chars += section_len
        else:
            # Include as much of this section as fits
            remaining = char_limit - total_chars
            if remaining > 200:  # only worth including if meaningful content fits
                truncated = section_text[:remaining] + "\n\n[... section truncated ...]"
                parts.append(truncated)
                total_chars += len(truncated)
            break  # budget exhausted

    return "\n\n".join(parts)


def _extract_reference_pmids(root: ET.Element) -> list[str]:
    """Extract PMIDs from the reference list of a PMC article."""
    pmids = []
    # PMC XML stores references in <ref-list><ref>...<pub-id pub-id-type="pmid">
    for ref in root.findall(".//ref-list//ref"):
        for pub_id in ref.findall(".//pub-id"):
            if pub_id.get("pub-id-type") == "pmid" and pub_id.text:
                pmids.append(pub_id.text.strip())
    return pmids


# ---------------------------------------------------------------------------
# Batch PMID metadata fetch
# ---------------------------------------------------------------------------


@registry.tool(
    description=(
        "Fetch metadata for a list of PMIDs from PubMed. "
        "Returns title, abstract, authors, journal, year, and publication type "
        "for each PMID. Use this to get details for PMIDs extracted from "
        "review article reference lists."
    ),
    tags=["fulltext", "pipeline"],
    cache=True,
    params={
        "pmids": "List of PubMed IDs to fetch metadata for",
    },
    returns="List of dicts with keys: pmid, title, abstract, authors, journal, year, pub_types",
)
def pubmed_fetch_by_pmids(pmids: list[str]) -> list[dict]:
    """Fetch PubMed metadata for a batch of PMIDs.

    Similar to the efetch in pubmed_search, but takes PMIDs directly
    and also extracts publication type (Review, Research Article, etc.).
    """
    if not pmids:
        return []

    all_articles = []

    # PubMed efetch handles up to ~200 IDs per request reliably
    for i in range(0, len(pmids), 200):
        batch = pmids[i : i + 200]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "email": config.ncbi_email,
        }
        if config.ncbi_api_key:
            params["api_key"] = config.ncbi_api_key

        resp = _ncbi_get(f"{_EUTILS_BASE}/efetch.fcgi", params=params)
        all_articles.extend(_parse_pubmed_xml_with_types(resp.text))

    return all_articles


def _parse_pubmed_xml_with_types(xml_text: str) -> list[dict]:
    """Parse PubMed XML, including publication types."""
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

        title_el = art.find("ArticleTitle")
        title = _text(title_el)

        # Abstract
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
        journal = _text(art.find("Journal/Title"))

        # Year
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

        # Publication types — this is how we distinguish reviews from primary
        pub_types = []
        for pt in art.findall("PublicationTypeList/PublicationType"):
            if pt.text:
                pub_types.append(pt.text.strip())

        articles.append({
            "pmid": _text(pmid_el),
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "pub_types": pub_types,
        })

    return articles


def _text(el: ET.Element | None) -> str:
    """Extract text from an XML element, handling mixed content."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


# ---------------------------------------------------------------------------
# Convenience: get best available text for a paper
# ---------------------------------------------------------------------------


def fetch_paper_text(
    pmid: str,
    pmcid: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Get the best available text for a paper.

    Tries PMC full text first, falls back to PubMed abstract.
    Returns a dict with keys: pmid, title, text, source_type, references.

    If *metadata* is supplied (a dict with title, authors, journal, year,
    pub_types, abstract) the per-paper ``pubmed_fetch_by_pmids`` call is
    skipped, avoiding a redundant NCBI round-trip when the caller already
    fetched metadata in bulk.
    """
    meta = metadata  # may be None

    # Try PMC full text
    if pmcid:
        fulltext = pmc_fulltext_fetch(pmcid)
        if fulltext and fulltext.get("body_text"):
            if meta is None:
                fetched = pubmed_fetch_by_pmids([pmid])
                meta = fetched[0] if fetched else {}
            return {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "journal": meta.get("journal", ""),
                "year": meta.get("year"),
                "pub_types": meta.get("pub_types", []),
                "text": fulltext["body_text"],
                "source_type": "full_text",
                "references": fulltext.get("references", []),
            }

    # Fall back to abstract
    if meta is None:
        fetched = pubmed_fetch_by_pmids([pmid])
        meta = fetched[0] if fetched else None
    if not meta:
        return {
            "pmid": pmid,
            "title": "",
            "text": "",
            "source_type": "not_found",
            "references": [],
        }

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "title": meta.get("title", ""),
        "authors": meta.get("authors", []),
        "journal": meta.get("journal", ""),
        "year": meta.get("year"),
        "pub_types": meta.get("pub_types", []),
        "text": meta.get("abstract", ""),
        "source_type": "abstract",
        "references": [],
    }
