"""Tests for fulltext tools — PMC XML parsing and metadata extraction."""

from targetsearch.tools.fulltext import (
    _classify_section,
    _extract_reference_pmids,
    _extract_sections,
    _parse_pmc_xml,
    _parse_pubmed_xml_with_types,
    _prioritize_and_assemble,
)
import xml.etree.ElementTree as ET


SAMPLE_PMC_XML = """\
<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <body>
      <sec>
        <title>Introduction</title>
        <p>Pulmonary fibrosis is a progressive disease characterized by scarring.</p>
        <p>TGF-beta signaling plays a central role in fibrogenesis.</p>
      </sec>
      <sec>
        <title>Results</title>
        <p>We found that SMAD3 phosphorylation was increased 3-fold in IPF tissue.</p>
        <p>Inhibition of SMAD3 reduced collagen deposition by 45%.</p>
      </sec>
    </body>
    <back>
      <ref-list>
        <ref id="R1">
          <mixed-citation>
            Smith et al. (2020)
            <pub-id pub-id-type="pmid">32000001</pub-id>
          </mixed-citation>
        </ref>
        <ref id="R2">
          <mixed-citation>
            Jones et al. (2019)
            <pub-id pub-id-type="pmid">31000002</pub-id>
            <pub-id pub-id-type="doi">10.1234/test</pub-id>
          </mixed-citation>
        </ref>
        <ref id="R3">
          <mixed-citation>
            No PMID for this one.
            <pub-id pub-id-type="doi">10.5678/other</pub-id>
          </mixed-citation>
        </ref>
      </ref-list>
    </back>
  </article>
</pmc-articleset>
"""

SAMPLE_PUBMED_WITH_TYPES = """\
<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <ArticleTitle>A review of IPF pathobiology.</ArticleTitle>
        <Abstract>
          <AbstractText>This review covers recent advances.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
        </AuthorList>
        <Journal>
          <Title>Chest</Title>
          <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
        </Journal>
        <PublicationTypeList>
          <PublicationType>Review</PublicationType>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <ArticleTitle>SMAD3 in lung fibrosis.</ArticleTitle>
        <Abstract>
          <AbstractText>We investigated SMAD3 phosphorylation.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Research Support, N.I.H., Extramural</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33333333</PMID>
      <Article>
        <ArticleTitle>Meta-analysis of anti-fibrotic therapies.</ArticleTitle>
        <Abstract>
          <AbstractText>Pooled analysis of 12 RCTs.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Garcia</LastName><ForeName>Maria</ForeName></Author>
        </AuthorList>
        <Journal>
          <Title>Lancet</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <PublicationTypeList>
          <PublicationType>Meta-Analysis</PublicationType>
          <PublicationType>Systematic Review</PublicationType>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestPMCXMLParser:
    def test_extracts_body_text(self):
        result = _parse_pmc_xml(SAMPLE_PMC_XML, "PMC_TEST")
        assert result is not None
        assert "Pulmonary fibrosis" in result["body_text"]
        assert "SMAD3 phosphorylation" in result["body_text"]
        assert result["pmcid"] == "PMC_TEST"

    def test_extracts_paragraphs_separated(self):
        result = _parse_pmc_xml(SAMPLE_PMC_XML, "PMC_TEST")
        # Paragraphs should be separated by double newlines
        assert "\n\n" in result["body_text"]

    def test_extracts_reference_pmids(self):
        result = _parse_pmc_xml(SAMPLE_PMC_XML, "PMC_TEST")
        refs = result["references"]
        assert "32000001" in refs
        assert "31000002" in refs
        # R3 has no PMID, should not appear
        assert len(refs) == 2

    def test_no_body_returns_none(self):
        xml = '<?xml version="1.0"?><pmc-articleset><article></article></pmc-articleset>'
        result = _parse_pmc_xml(xml, "PMC_EMPTY")
        assert result is None

    def test_invalid_xml_returns_none(self):
        result = _parse_pmc_xml("not xml at all", "PMC_BAD")
        assert result is None


class TestExtractReferencePmids:
    def test_extracts_from_ref_list(self):
        root = ET.fromstring(SAMPLE_PMC_XML)
        pmids = _extract_reference_pmids(root)
        assert pmids == ["32000001", "31000002"]

    def test_empty_ref_list(self):
        root = ET.fromstring(
            '<?xml version="1.0"?><article><back><ref-list></ref-list></back></article>'
        )
        pmids = _extract_reference_pmids(root)
        assert pmids == []


class TestPubMedWithTypes:
    def test_parses_three_articles(self):
        articles = _parse_pubmed_xml_with_types(SAMPLE_PUBMED_WITH_TYPES)
        assert len(articles) == 3

    def test_review_has_pub_types(self):
        articles = _parse_pubmed_xml_with_types(SAMPLE_PUBMED_WITH_TYPES)
        review = articles[0]
        assert review["pmid"] == "11111111"
        assert "Review" in review["pub_types"]
        assert "Journal Article" in review["pub_types"]

    def test_primary_article_pub_types(self):
        articles = _parse_pubmed_xml_with_types(SAMPLE_PUBMED_WITH_TYPES)
        primary = articles[1]
        assert primary["pmid"] == "22222222"
        assert "Review" not in primary["pub_types"]
        assert "Journal Article" in primary["pub_types"]

    def test_meta_analysis_pub_types(self):
        articles = _parse_pubmed_xml_with_types(SAMPLE_PUBMED_WITH_TYPES)
        meta = articles[2]
        assert "Meta-Analysis" in meta["pub_types"]
        assert "Systematic Review" in meta["pub_types"]

    def test_metadata_fields(self):
        articles = _parse_pubmed_xml_with_types(SAMPLE_PUBMED_WITH_TYPES)
        art = articles[0]
        assert art["title"] == "A review of IPF pathobiology."
        assert art["authors"] == ["Smith John"]
        assert art["journal"] == "Chest"
        assert art["year"] == 2023


# ── Section-aware extraction tests ─────────────────────────────────────

MULTI_SECTION_XML = """\
<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <body>
      <sec>
        <title>Introduction</title>
        <p>Background information about the disease.</p>
      </sec>
      <sec>
        <title>Methods</title>
        <p>We used a bleomycin mouse model.</p>
        <sec>
          <title>Cell culture</title>
          <p>Primary fibroblasts were isolated.</p>
        </sec>
      </sec>
      <sec>
        <title>Results</title>
        <p>SMAD3 phosphorylation was increased 3-fold.</p>
        <p>Collagen deposition was reduced by 45%.</p>
        <sec>
          <title>Subgroup analysis</title>
          <p>The effect was stronger in older mice.</p>
        </sec>
      </sec>
      <sec>
        <title>Discussion</title>
        <p>These findings suggest SMAD3 is a viable target.</p>
      </sec>
      <sec>
        <title>Conclusions</title>
        <p>SMAD3 inhibition warrants clinical investigation.</p>
      </sec>
      <sec>
        <title>Acknowledgments</title>
        <p>We thank the NIH for funding.</p>
      </sec>
      <sec>
        <title>Author Contributions</title>
        <p>JS designed the study. AD performed experiments.</p>
      </sec>
    </body>
    <back>
      <ref-list>
        <ref><mixed-citation><pub-id pub-id-type="pmid">99999</pub-id></mixed-citation></ref>
      </ref-list>
    </back>
  </article>
</pmc-articleset>
"""

NO_SEC_XML = """\
<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <body>
      <p>First paragraph without sections.</p>
      <p>Second paragraph without sections.</p>
    </body>
  </article>
</pmc-articleset>
"""

EMPTY_BODY_XML = """\
<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <body>
      <sec>
        <title>Acknowledgments</title>
        <p>Thanks to everyone.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>
"""


class TestClassifySection:
    def test_results_priority(self):
        assert _classify_section("Results") == 1
        assert _classify_section("results") == 1
        assert _classify_section("Findings") == 1
        assert _classify_section("Results and discussion") == 1

    def test_discussion_priority(self):
        assert _classify_section("Discussion") == 2

    def test_introduction_priority(self):
        assert _classify_section("Introduction") == 3
        assert _classify_section("Background") == 3

    def test_methods_priority(self):
        assert _classify_section("Methods") == 4
        assert _classify_section("Materials and Methods") == 4
        assert _classify_section("Experimental") == 4
        assert _classify_section("Patients and methods") == 4

    def test_conclusions_priority(self):
        assert _classify_section("Conclusions") == 5
        assert _classify_section("Conclusion") == 5

    def test_unknown_section(self):
        assert _classify_section("Supplementary Tables") == 6

    def test_skip_sections(self):
        assert _classify_section("Acknowledgments") is None
        assert _classify_section("Acknowledgements") is None
        assert _classify_section("Conflict of Interest") is None
        assert _classify_section("Funding") is None
        assert _classify_section("Author Contributions") is None
        assert _classify_section("Data Availability") is None
        assert _classify_section("Declaration of Competing Interest") is None
        assert _classify_section("Ethics Statement") is None

    def test_case_insensitive(self):
        assert _classify_section("RESULTS") == 1
        assert _classify_section("ACKNOWLEDGMENTS") is None

    def test_partial_match_skip(self):
        assert _classify_section("CRediT authorship contribution statement") is None


class TestExtractSections:
    def test_extracts_named_sections(self):
        root = ET.fromstring(MULTI_SECTION_XML)
        body = root.find(".//body")
        sections = _extract_sections(body)

        titles = [title for title, _ in sections]
        assert "Introduction" in titles
        assert "Methods" in titles
        assert "Results" in titles
        assert "Discussion" in titles
        assert "Conclusions" in titles
        assert "Acknowledgments" in titles  # extracted, filtered later by prioritize

    def test_nested_sections_flattened(self):
        root = ET.fromstring(MULTI_SECTION_XML)
        body = root.find(".//body")
        sections = _extract_sections(body)

        # Methods section should include nested "Cell culture" paragraph
        methods = next(text for title, text in sections if title == "Methods")
        assert "bleomycin mouse model" in methods
        assert "Primary fibroblasts" in methods

        # Results section should include nested "Subgroup analysis" paragraph
        results = next(text for title, text in sections if title == "Results")
        assert "SMAD3 phosphorylation" in results
        assert "stronger in older mice" in results

    def test_no_sec_returns_empty(self):
        root = ET.fromstring(NO_SEC_XML)
        body = root.find(".//body")
        sections = _extract_sections(body)
        assert sections == []

    def test_empty_paragraphs_skipped(self):
        xml = """\
<?xml version="1.0"?>
<article><body>
  <sec><title>Empty</title></sec>
  <sec><title>Has content</title><p>Real text.</p></sec>
</body></article>"""
        root = ET.fromstring(xml)
        body = root.find(".//body")
        sections = _extract_sections(body)
        assert len(sections) == 1
        assert sections[0][0] == "Has content"


class TestPrioritizeAndAssemble:
    def test_results_before_methods(self):
        sections = [
            ("Methods", "Methods text here."),
            ("Results", "Results text here."),
            ("Introduction", "Intro text here."),
        ]
        result = _prioritize_and_assemble(sections)
        results_pos = result.index("Results text")
        methods_pos = result.index("Methods text")
        intro_pos = result.index("Intro text")
        assert results_pos < intro_pos < methods_pos

    def test_skips_acknowledgments(self):
        sections = [
            ("Results", "Important findings."),
            ("Acknowledgments", "Thanks to the NIH."),
        ]
        result = _prioritize_and_assemble(sections)
        assert "Important findings" in result
        assert "Thanks to the NIH" not in result

    def test_section_headers_included(self):
        sections = [("Results", "Some results.")]
        result = _prioritize_and_assemble(sections)
        assert "## Results" in result
        assert "Some results." in result

    def test_char_limit_truncates_low_priority(self):
        sections = [
            ("Results", "R" * 300),
            ("Discussion", "D" * 300),
            ("Methods", "M" * 300),
        ]
        # Limit that fits Results + Discussion but not Methods
        result = _prioritize_and_assemble(sections, char_limit=700)
        assert "R" * 300 in result
        assert "D" * 300 in result
        assert "M" * 300 not in result

    def test_partial_section_included(self):
        sections = [
            ("Results", "R" * 500),
            ("Discussion", "D" * 1000),
        ]
        # Limit that fits Results fully but Discussion only partially
        result = _prioritize_and_assemble(sections, char_limit=800)
        assert "R" * 500 in result
        assert "D" in result  # some Discussion content included
        assert "D" * 1000 not in result  # but not all of it
        assert "[... section truncated ...]" in result

    def test_very_small_remainder_skipped(self):
        sections = [
            ("Results", "R" * 490),
            ("Discussion", "D" * 500),
        ]
        # Only ~10 chars remaining after Results — not worth including
        result = _prioritize_and_assemble(sections, char_limit=510)
        assert "R" * 490 in result
        assert "section truncated" not in result  # too small to bother

    def test_unnamed_sections_lowest_priority(self):
        sections = [
            ("", "Unnamed content."),
            ("Results", "Results content."),
        ]
        result = _prioritize_and_assemble(sections)
        results_pos = result.index("Results content")
        unnamed_pos = result.index("Unnamed content")
        assert results_pos < unnamed_pos

    def test_empty_sections_list(self):
        result = _prioritize_and_assemble([])
        assert result == ""


class TestParsePMCXMLSectionAware:
    def test_multi_section_prioritized(self):
        result = _parse_pmc_xml(MULTI_SECTION_XML, "PMC_TEST")
        assert result is not None
        body = result["body_text"]

        # Results should appear before Methods
        assert "SMAD3 phosphorylation" in body
        assert "bleomycin mouse model" in body
        results_pos = body.index("SMAD3 phosphorylation")
        methods_pos = body.index("bleomycin mouse model")
        assert results_pos < methods_pos

    def test_acknowledgments_excluded(self):
        result = _parse_pmc_xml(MULTI_SECTION_XML, "PMC_TEST")
        assert "thank the NIH" not in result["body_text"]

    def test_author_contributions_excluded(self):
        result = _parse_pmc_xml(MULTI_SECTION_XML, "PMC_TEST")
        assert "designed the study" not in result["body_text"]

    def test_references_still_extracted(self):
        result = _parse_pmc_xml(MULTI_SECTION_XML, "PMC_TEST")
        assert "99999" in result["references"]

    def test_no_sec_fallback(self):
        result = _parse_pmc_xml(NO_SEC_XML, "PMC_NOSEC")
        assert result is not None
        assert "First paragraph" in result["body_text"]
        assert "Second paragraph" in result["body_text"]

    def test_only_skippable_sections_returns_none(self):
        result = _parse_pmc_xml(EMPTY_BODY_XML, "PMC_EMPTY")
        # Only has Acknowledgments which gets skipped, so body_text is empty
        assert result is None

    def test_existing_sample_still_works(self):
        """The original SAMPLE_PMC_XML should still parse correctly."""
        result = _parse_pmc_xml(SAMPLE_PMC_XML, "PMC_TEST")
        assert result is not None
        assert "Pulmonary fibrosis" in result["body_text"]
        assert "SMAD3 phosphorylation" in result["body_text"]
        assert result["pmcid"] == "PMC_TEST"
        assert "32000001" in result["references"]
