"""Tests for literature tools — verifies registration and XML parsing.

These tests don't make live API calls. They test the registry integration
and the PubMed XML parser with a fixture.
"""

from targetsearch.core.registry import registry

# Import the module to trigger tool registration
import targetsearch.tools.literature  # noqa: F401


SAMPLE_PUBMED_XML = """\
<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN"
 "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>TGF-beta signaling in pulmonary fibrosis: a review.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Pulmonary fibrosis is a progressive disease.</AbstractText>
          <AbstractText Label="CONCLUSIONS">TGF-beta is a central mediator.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John A</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane B</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Journal of Respiratory Research</Title>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article>
        <ArticleTitle>Novel targets in IPF.</ArticleTitle>
        <Abstract>
          <AbstractText>We identified several novel targets.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Garcia</LastName>
            <ForeName>Maria</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestToolRegistration:
    def test_pubmed_search_registered(self):
        spec = registry.get_tool("pubmed_search")
        assert spec.name == "pubmed_search"
        assert "literature" in spec.tags
        assert "pubmed" in spec.tags
        assert spec.cache is True

    def test_semantic_scholar_registered(self):
        spec = registry.get_tool("semantic_scholar_search")
        assert "literature" in spec.tags
        assert spec.cache is True

    def test_literature_tools_filterable(self):
        lit_tools = registry.get_tools(tags=["literature"])
        names = {t.name for t in lit_tools}
        assert "pubmed_search" in names
        assert "semantic_scholar_search" in names

    def test_openai_schema_generated(self):
        schema = registry.get_tool("pubmed_search").to_openai_schema()
        fn = schema["function"]
        assert fn["name"] == "pubmed_search"
        assert "query" in fn["parameters"]["properties"]
        assert "query" in fn["parameters"]["required"]


class TestPubMedXMLParser:
    def test_parse_two_articles(self):
        from targetsearch.tools.literature import _parse_pubmed_xml

        articles = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
        assert len(articles) == 2

    def test_first_article_fields(self):
        from targetsearch.tools.literature import _parse_pubmed_xml

        articles = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
        art = articles[0]
        assert art["pmid"] == "12345678"
        assert "TGF-beta" in art["title"]
        assert art["year"] == 2023
        assert len(art["authors"]) == 2
        assert art["authors"][0] == "Smith John A"
        assert art["journal"] == "Journal of Respiratory Research"

    def test_structured_abstract(self):
        from targetsearch.tools.literature import _parse_pubmed_xml

        articles = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
        abstract = articles[0]["abstract"]
        # Structured abstract should include labels
        assert "BACKGROUND:" in abstract
        assert "CONCLUSIONS:" in abstract

    def test_simple_abstract(self):
        from targetsearch.tools.literature import _parse_pubmed_xml

        articles = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
        abstract = articles[1]["abstract"]
        assert "novel targets" in abstract

    def test_empty_xml(self):
        from targetsearch.tools.literature import _parse_pubmed_xml

        articles = _parse_pubmed_xml(
            '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
        )
        assert articles == []
