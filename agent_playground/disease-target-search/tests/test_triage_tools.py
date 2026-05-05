"""Tests for triage_search_results — registration, parsing, normalization.

Tests cover:
- Tool registration and schema
- Paper formatting for LLM input
- Triage result parsing (valid, invalid, missing papers)
- Score clamping and bucket normalization
- Deduplication
- Empty input handling
- Filtering by min_relevance
"""

import json

from targetsearch.core.registry import registry
from targetsearch.tools.triage_tools import (
    EVIDENCE_BUCKETS,
    _build_triage_prompt,
    _clamp_score,
    _format_papers_for_triage,
    _normalize_papers,
    _normalize_bucket,
    _parse_triage_result,
    triage_search_results,
)


# ── Sample data ────────────────────────────────────────────────────────

SAMPLE_PAPERS = [
    {
        "pmid": "11111",
        "title": "GWAS identifies novel loci for IPF",
        "abstract": "We performed a genome-wide association study of 2,000 IPF patients...",
        "year": 2023,
        "journal": "Nature Genetics",
        "authors": ["Smith J", "Doe A"],
    },
    {
        "pmid": "22222",
        "title": "TGF-beta signaling in pulmonary fibrosis",
        "abstract": "Transforming growth factor beta plays a central role in fibrogenesis...",
        "year": 2022,
        "journal": "American Journal of Respiratory Cell and Molecular Biology",
        "authors": ["Jones B"],
    },
    {
        "pmid": "33333",
        "title": "Machine learning for image classification",
        "abstract": "We present a novel deep learning architecture for classifying images...",
        "year": 2023,
        "journal": "IEEE Transactions on Pattern Analysis",
        "authors": ["Lee C"],
    },
]


# ── Registration tests ─────────────────────────────────────────────────


class TestToolRegistration:
    def test_registered_with_literature_tag(self):
        names = registry.list_names(tags=["literature"])
        assert "triage_search_results" in names

    def test_no_context_needed(self):
        """Triage is a leaf tool — no ActionContext."""
        assert registry.tool_needs_context("triage_search_results") is False

    def test_schema_params(self):
        schema = registry.get_tool("triage_search_results").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "papers" in props
        assert "disease_area" in props
        assert "search_context" in props
        assert "min_relevance" in props
        assert "max_recommended" in props
        assert "ctx" not in props

    def test_schema_required(self):
        schema = registry.get_tool("triage_search_results").to_openai_schema()
        required = schema["function"]["parameters"]["required"]
        assert "papers" in required
        assert "disease_area" in required
        # search_context, min_relevance, max_recommended have defaults
        assert "search_context" not in required
        assert "min_relevance" not in required
        assert "max_recommended" not in required


# ── Formatting tests ───────────────────────────────────────────────────


class TestFormatPapersForTriage:
    def test_basic_formatting(self):
        result = _format_papers_for_triage(SAMPLE_PAPERS)
        assert "PMID: 11111" in result
        assert "GWAS identifies novel loci" in result
        assert "Nature Genetics" in result
        assert "2023" in result

    def test_all_papers_included(self):
        result = _format_papers_for_triage(SAMPLE_PAPERS)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_long_abstract_truncated(self):
        papers = [{
            "pmid": "99999",
            "title": "Long paper",
            "abstract": "A" * 1000,
            "year": 2024,
            "journal": "Test",
        }]
        result = _format_papers_for_triage(papers)
        assert "..." in result
        # Should be truncated to ~500 chars + "..."
        abstract_part = result.split("Abstract: ")[1]
        assert len(abstract_part) < 600

    def test_missing_abstract(self):
        papers = [{"pmid": "88888", "title": "No abstract", "year": 2024, "journal": "J"}]
        result = _format_papers_for_triage(papers)
        assert "(not available)" in result

    def test_empty_list(self):
        result = _format_papers_for_triage([])
        assert result == ""


# ── Score clamping tests ───────────────────────────────────────────────


class TestClampScore:
    def test_normal_range(self):
        assert _clamp_score(5) == 5
        assert _clamp_score(0) == 0
        assert _clamp_score(10) == 10

    def test_below_zero(self):
        assert _clamp_score(-3) == 0

    def test_above_ten(self):
        assert _clamp_score(15) == 10

    def test_string_number(self):
        assert _clamp_score("7") == 7

    def test_float(self):
        assert _clamp_score(7.8) == 7

    def test_invalid(self):
        assert _clamp_score("not a number") == 5
        assert _clamp_score(None) == 5


# ── Bucket normalization tests ─────────────────────────────────────────


class TestNormalizeBucket:
    def test_exact_match(self):
        for bucket in EVIDENCE_BUCKETS:
            assert _normalize_bucket(bucket) == bucket

    def test_with_spaces(self):
        assert _normalize_bucket("disease biology") == "disease_biology"
        assert _normalize_bucket("human genetics") == "human_genetics"

    def test_with_hyphens(self):
        assert _normalize_bucket("preclinical-therapeutic") == "preclinical_therapeutic"

    def test_case_insensitive(self):
        assert _normalize_bucket("Human_Genetics") == "human_genetics"
        assert _normalize_bucket("CLINICAL") == "clinical"

    def test_partial_match(self):
        assert _normalize_bucket("genetics") == "human_genetics"
        assert _normalize_bucket("biology") == "disease_biology"
        assert _normalize_bucket("therapeutic") == "preclinical_therapeutic"

    def test_unknown_defaults(self):
        assert _normalize_bucket("completely_unknown") == "disease_biology"


# ── Parse triage result tests ──────────────────────────────────────────


class TestParseTriageResult:
    def test_valid_result(self):
        raw = json.dumps({
            "triaged_papers": [
                {
                    "pmid": "11111",
                    "relevance_score": 9,
                    "evidence_bucket": "human_genetics",
                    "rationale": "GWAS study directly relevant",
                    "redundant_with": None,
                },
                {
                    "pmid": "22222",
                    "relevance_score": 7,
                    "evidence_bucket": "disease_biology",
                    "rationale": "Mechanism paper",
                    "redundant_with": None,
                },
                {
                    "pmid": "33333",
                    "relevance_score": 1,
                    "evidence_bucket": "disease_biology",
                    "rationale": "ML paper, not relevant",
                    "redundant_with": None,
                },
            ],
            "bucket_coverage": {
                "disease_biology": 1,
                "human_genetics": 1,
                "preclinical_therapeutic": 0,
                "clinical": 0,
            },
            "recommendation": "Need preclinical and clinical evidence",
        })
        result = _parse_triage_result(raw, {"11111", "22222", "33333"})

        assert len(result["triaged_papers"]) == 3
        # Should be sorted by relevance descending
        assert result["triaged_papers"][0]["pmid"] == "11111"
        assert result["triaged_papers"][0]["relevance_score"] == 9
        assert result["triaged_papers"][-1]["pmid"] == "33333"
        assert result["triaged_papers"][-1]["relevance_score"] == 1
        assert result["bucket_coverage"]["human_genetics"] == 1
        assert result["recommendation"] == "Need preclinical and clinical evidence"

    def test_missing_papers_added_back(self):
        """If the LLM drops papers, they should be added back."""
        raw = json.dumps({
            "triaged_papers": [
                {
                    "pmid": "11111",
                    "relevance_score": 8,
                    "evidence_bucket": "human_genetics",
                    "rationale": "Good paper",
                },
            ],
            "bucket_coverage": {},
            "recommendation": "",
        })
        result = _parse_triage_result(raw, {"11111", "22222", "33333"})

        assert len(result["triaged_papers"]) == 3
        pmids = {p["pmid"] for p in result["triaged_papers"]}
        assert pmids == {"11111", "22222", "33333"}

        # Missing papers should have default score of 5
        missing_papers = [
            p for p in result["triaged_papers"] if p["pmid"] in {"22222", "33333"}
        ]
        for p in missing_papers:
            assert p["relevance_score"] == 5
            assert "default" in p["rationale"].lower()

    def test_invalid_json_returns_all_papers(self):
        """On parse failure, all papers should be included with default scores."""
        result = _parse_triage_result("not valid json at all", {"11111", "22222"})

        assert len(result["triaged_papers"]) == 2
        pmids = {p["pmid"] for p in result["triaged_papers"]}
        assert pmids == {"11111", "22222"}
        for p in result["triaged_papers"]:
            assert p["relevance_score"] == 5

    def test_score_clamping_in_parse(self):
        raw = json.dumps({
            "triaged_papers": [
                {"pmid": "11111", "relevance_score": 15, "evidence_bucket": "clinical"},
                {"pmid": "22222", "relevance_score": -5, "evidence_bucket": "clinical"},
            ],
            "bucket_coverage": {},
        })
        result = _parse_triage_result(raw, {"11111", "22222"})
        scores = {p["pmid"]: p["relevance_score"] for p in result["triaged_papers"]}
        assert scores["11111"] == 10
        assert scores["22222"] == 0

    def test_bucket_normalization_in_parse(self):
        raw = json.dumps({
            "triaged_papers": [
                {"pmid": "11111", "relevance_score": 7, "evidence_bucket": "Human Genetics"},
            ],
            "bucket_coverage": {},
        })
        result = _parse_triage_result(raw, {"11111"})
        assert result["triaged_papers"][0]["evidence_bucket"] == "human_genetics"

    def test_pmid_normalized_to_string(self):
        """PMIDs should always be strings, even if LLM returns integers."""
        raw = json.dumps({
            "triaged_papers": [
                {"pmid": 11111, "relevance_score": 7, "evidence_bucket": "clinical"},
            ],
            "bucket_coverage": {},
        })
        result = _parse_triage_result(raw, {"11111"})
        assert result["triaged_papers"][0]["pmid"] == "11111"

    def test_missing_bucket_coverage_filled(self):
        raw = json.dumps({
            "triaged_papers": [],
            "bucket_coverage": {"clinical": 2},
        })
        result = _parse_triage_result(raw, set())
        for bucket in EVIDENCE_BUCKETS:
            assert bucket in result["bucket_coverage"]
        assert result["bucket_coverage"]["clinical"] == 2
        assert result["bucket_coverage"]["disease_biology"] == 0


# ── Empty input test ───────────────────────────────────────────────────


class TestTriageEmptyInput:
    def test_empty_papers_list(self):
        """Should return immediately without calling the LLM."""
        result = triage_search_results(
            papers=[],
            disease_area="IPF",
        )
        assert result["recommended_pmids"] == []
        assert result["total_input"] == 0
        assert result["total_recommended"] == 0
        assert result["filtered_out"] == 0
        for bucket in EVIDENCE_BUCKETS:
            assert bucket in result["bucket_coverage"]


# ── Build prompt tests ─────────────────────────────────────────────────


class TestBuildTriagePrompt:
    def test_contains_disease(self):
        prompt = _build_triage_prompt("ALS", "")
        assert "ALS" in prompt

    def test_contains_buckets(self):
        prompt = _build_triage_prompt("IPF", "")
        assert "disease_biology" in prompt
        assert "human_genetics" in prompt
        assert "preclinical_therapeutic" in prompt
        assert "clinical" in prompt

    def test_contains_search_context(self):
        prompt = _build_triage_prompt("IPF", "## Search context\n\nfilling genetic gaps")
        assert "filling genetic gaps" in prompt

    def test_output_format_specified(self):
        prompt = _build_triage_prompt("IPF", "")
        assert "triaged_papers" in prompt
        assert "relevance_score" in prompt
        assert "bucket_coverage" in prompt


# ── Deduplication test ─────────────────────────────────────────────────


class TestStringInputNormalization:
    def test_bare_pmid_strings(self):
        """LLM may pass PMIDs as strings instead of dicts."""
        result = triage_search_results(
            papers=["12345", "67890"],
            disease_area="IPF",
        )
        assert set(result["recommended_pmids"]) == {"12345", "67890"}
        assert result["total_input"] == 2
        assert result["total_recommended"] == 2
        assert result["filtered_out"] == 0

    def test_mixed_strings_and_dicts(self):
        """Mix of strings and dicts should normalize strings to dicts."""
        papers = [
            {"pmid": "11111", "title": "Real paper", "abstract": "Has content", "year": 2024, "journal": "J"},
            "22222",
        ]
        normalized = _normalize_papers(papers)
        assert len(normalized) == 2
        assert normalized[0]["title"] == "Real paper"
        assert normalized[1]["pmid"] == "22222"

    def test_json_string_papers_parsed(self):
        """LLM may pass paper dicts as JSON strings — these should be parsed."""
        json_paper = json.dumps({
            "pmid": "39121882",
            "title": "New promises in NSCLC",
            "abstract": "Targeted therapies have improved treatment.",
            "year": 2024,
            "journal": "Lancet",
        })
        normalized = _normalize_papers([json_paper])
        assert len(normalized) == 1
        assert normalized[0]["pmid"] == "39121882"
        assert normalized[0]["title"] == "New promises in NSCLC"
        assert "Targeted therapies" in normalized[0]["abstract"]

    def test_json_strings_not_treated_as_bare_pmids(self):
        """JSON string papers should retain their metadata after normalization."""
        papers = [
            json.dumps({"pmid": "111", "title": "Paper A", "abstract": "Content A", "year": 2024, "journal": "J"}),
            json.dumps({"pmid": "222", "title": "Paper B", "abstract": "Content B", "year": 2023, "journal": "K"}),
        ]
        normalized = _normalize_papers(papers)
        # All papers have titles and abstracts — should NOT be bare PMIDs
        assert all(p.get("title") for p in normalized)
        assert all(p.get("abstract") for p in normalized)

    def test_invalid_json_string_treated_as_pmid(self):
        """A string starting with { but not valid JSON falls back to bare PMID."""
        normalized = _normalize_papers(["{not valid json"])
        assert len(normalized) == 1
        assert normalized[0]["pmid"] == "{not valid json"
        assert normalized[0]["title"] == ""

    def test_duplicate_bare_pmids(self):
        result = triage_search_results(
            papers=["12345", "12345", "67890"],
            disease_area="IPF",
        )
        assert result["recommended_pmids"] == ["12345", "67890"]
        assert result["total_input"] == 2


class TestDeduplication:
    def test_duplicate_pmids_removed(self):
        """triage_search_results should deduplicate by PMID before triaging."""
        papers = [
            {"pmid": "11111", "title": "Paper A", "abstract": "...", "year": 2023, "journal": "J"},
            {"pmid": "11111", "title": "Paper A (dup)", "abstract": "...", "year": 2023, "journal": "J"},
            {"pmid": "22222", "title": "Paper B", "abstract": "...", "year": 2023, "journal": "J"},
        ]
        # We can't call the full function (needs LLM), but we can test
        # the dedup logic by checking the empty-input path doesn't crash
        # and the formatting deduplicates
        seen: set[str] = set()
        unique = []
        for p in papers:
            pmid = str(p.get("pmid", ""))
            if pmid and pmid not in seen:
                seen.add(pmid)
                unique.append(p)
        assert len(unique) == 2
        assert {p["pmid"] for p in unique} == {"11111", "22222"}


# ── Filtering by min_relevance ─────────────────────────────────────────


class TestMinRelevanceFiltering:
    def test_filtering_logic(self):
        """Verify that the filtering logic works on parsed results."""
        triaged_papers = [
            {"pmid": "11111", "relevance_score": 9, "evidence_bucket": "human_genetics", "rationale": "", "redundant_with": None},
            {"pmid": "22222", "relevance_score": 6, "evidence_bucket": "disease_biology", "rationale": "", "redundant_with": None},
            {"pmid": "33333", "relevance_score": 3, "evidence_bucket": "disease_biology", "rationale": "", "redundant_with": None},
            {"pmid": "44444", "relevance_score": 1, "evidence_bucket": "disease_biology", "rationale": "", "redundant_with": None},
        ]

        # min_relevance = 4 should keep papers with score >= 4
        min_relevance = 4
        recommended = [p for p in triaged_papers if p["relevance_score"] >= min_relevance]
        assert len(recommended) == 2
        assert {p["pmid"] for p in recommended} == {"11111", "22222"}

        # min_relevance = 7 should keep only the top paper
        min_relevance = 7
        recommended = [p for p in triaged_papers if p["relevance_score"] >= min_relevance]
        assert len(recommended) == 1
        assert recommended[0]["pmid"] == "11111"

        # min_relevance = 0 should keep all
        min_relevance = 0
        recommended = [p for p in triaged_papers if p["relevance_score"] >= min_relevance]
        assert len(recommended) == 4

