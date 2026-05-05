"""Tests for paper classification logic (ported from paper_pipeline).

These tests don't make LLM or API calls. They test the deterministic
classification logic now in coordination_tools.
"""

from targetsearch.tools.coordination_tools import _classify_papers


class TestClassifyPapers:
    def test_review_detected(self):
        papers = [
            {"pmid": "1", "pub_types": ["Review", "Journal Article"]},
            {"pmid": "2", "pub_types": ["Journal Article"]},
        ]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 1
        assert reviews[0]["pmid"] == "1"
        assert len(primaries) == 1
        assert primaries[0]["pmid"] == "2"

    def test_meta_analysis_is_review(self):
        papers = [
            {"pmid": "1", "pub_types": ["Meta-Analysis", "Journal Article"]},
        ]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 1
        assert len(primaries) == 0

    def test_systematic_review_is_review(self):
        papers = [
            {"pmid": "1", "pub_types": ["Systematic Review", "Journal Article"]},
        ]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 1

    def test_no_pub_types_is_primary(self):
        papers = [
            {"pmid": "1", "pub_types": []},
            {"pmid": "2"},  # no pub_types key at all
        ]
        reviews, primaries = _classify_papers(papers)
        assert len(reviews) == 0
        assert len(primaries) == 2

    def test_empty_list(self):
        reviews, primaries = _classify_papers([])
        assert reviews == []
        assert primaries == []

    def test_mixed_batch(self):
        papers = [
            {"pmid": "1", "pub_types": ["Review"]},
            {"pmid": "2", "pub_types": ["Journal Article"]},
            {"pmid": "3", "pub_types": ["Meta-Analysis", "Systematic Review"]},
            {"pmid": "4", "pub_types": ["Case Reports"]},
            {"pmid": "5", "pub_types": ["Practice Guideline"]},
        ]
        reviews, primaries = _classify_papers(papers)
        review_pmids = {r["pmid"] for r in reviews}
        primary_pmids = {p["pmid"] for p in primaries}
        assert review_pmids == {"1", "3", "5"}
        assert primary_pmids == {"2", "4"}
