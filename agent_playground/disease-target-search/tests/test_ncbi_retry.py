"""Tests for _ncbi_get retry logic and fetch_paper_text metadata passthrough."""

from unittest.mock import patch, MagicMock
import httpx
import pytest

from targetsearch.tools.fulltext import _ncbi_get, fetch_paper_text


# ---------------------------------------------------------------------------
# _ncbi_get retry logic
# ---------------------------------------------------------------------------


class TestNcbiGetRetry:
    """Verify retry behaviour on 429 and 5xx responses."""

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_success_on_first_try(self, mock_get, mock_sleep):
        resp = MagicMock(status_code=200)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = _ncbi_get("https://example.com", {"q": "1"})

        assert result is resp
        mock_get.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_retry_on_429_then_success(self, mock_get, mock_sleep):
        fail = MagicMock(status_code=429)
        ok = MagicMock(status_code=200)
        ok.raise_for_status = MagicMock()
        mock_get.side_effect = [fail, ok]

        result = _ncbi_get("https://example.com/efetch", {"q": "1"})

        assert result is ok
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # first backoff

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_retry_on_500_then_success(self, mock_get, mock_sleep):
        fail = MagicMock(status_code=500)
        ok = MagicMock(status_code=200)
        ok.raise_for_status = MagicMock()
        mock_get.side_effect = [fail, ok]

        result = _ncbi_get("https://example.com/esearch", {"q": "1"})

        assert result is ok
        assert mock_get.call_count == 2

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_exhausts_retries_on_persistent_429(self, mock_get, mock_sleep):
        fail = MagicMock(status_code=429)
        fail.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "429", request=MagicMock(), response=fail
            )
        )
        mock_get.return_value = fail

        with pytest.raises(httpx.HTTPStatusError):
            _ncbi_get("https://example.com/efetch", {"q": "1"})

        # 1 initial + 3 retries = 4 calls
        assert mock_get.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_backoff_schedule(self, mock_get, mock_sleep):
        """Verify the exact backoff delays: 1.0, 3.0, 6.0."""
        fail = MagicMock(status_code=503)
        fail.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "503", request=MagicMock(), response=fail
            )
        )
        mock_get.return_value = fail

        with pytest.raises(httpx.HTTPStatusError):
            _ncbi_get("https://example.com/efetch", {"q": "1"})

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 3.0, 6.0]

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_retry_on_connection_error(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            httpx.ConnectError("connection refused"),
            MagicMock(status_code=200, raise_for_status=MagicMock()),
        ]

        result = _ncbi_get("https://example.com/efetch", {"q": "1"})

        assert result.status_code == 200
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch("targetsearch.tools.fulltext.time.sleep")
    @patch("targetsearch.tools.fulltext.httpx.get")
    def test_no_retry_on_4xx_other_than_429(self, mock_get, mock_sleep):
        """Client errors other than 429 should fail immediately."""
        fail = MagicMock(status_code=400)
        fail.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "400", request=MagicMock(), response=fail
            )
        )
        mock_get.return_value = fail

        with pytest.raises(httpx.HTTPStatusError):
            _ncbi_get("https://example.com/efetch", {"q": "1"})

        mock_get.assert_called_once()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_paper_text metadata passthrough
# ---------------------------------------------------------------------------


_SAMPLE_META = {
    "pmid": "12345",
    "title": "Test Paper Title",
    "authors": ["Author A", "Author B"],
    "journal": "J Test",
    "year": 2024,
    "pub_types": ["Journal Article"],
    "abstract": "This is the abstract text.",
}


class TestFetchPaperTextMetadataPassthrough:
    """Verify that pre-supplied metadata skips the pubmed_fetch_by_pmids call."""

    @patch("targetsearch.tools.fulltext.pmc_fulltext_fetch")
    @patch("targetsearch.tools.fulltext.pubmed_fetch_by_pmids")
    def test_fulltext_with_metadata_skips_fetch(self, mock_pubmed, mock_pmc):
        """When metadata is provided and PMC full text is available,
        pubmed_fetch_by_pmids should NOT be called."""
        mock_pmc.return_value = {
            "body_text": "Full text body here.",
            "references": ["99999"],
            "pmcid": "PMC111",
        }

        result = fetch_paper_text("12345", pmcid="PMC111", metadata=_SAMPLE_META)

        mock_pubmed.assert_not_called()
        assert result["source_type"] == "full_text"
        assert result["title"] == "Test Paper Title"
        assert result["text"] == "Full text body here."
        assert result["authors"] == ["Author A", "Author B"]

    @patch("targetsearch.tools.fulltext.pmc_fulltext_fetch")
    @patch("targetsearch.tools.fulltext.pubmed_fetch_by_pmids")
    def test_abstract_with_metadata_skips_fetch(self, mock_pubmed, mock_pmc):
        """When metadata is provided but no PMC full text, should use
        abstract from metadata without calling pubmed_fetch_by_pmids."""
        result = fetch_paper_text("12345", pmcid=None, metadata=_SAMPLE_META)

        mock_pubmed.assert_not_called()
        mock_pmc.assert_not_called()
        assert result["source_type"] == "abstract"
        assert result["text"] == "This is the abstract text."
        assert result["title"] == "Test Paper Title"

    @patch("targetsearch.tools.fulltext.pmc_fulltext_fetch")
    @patch("targetsearch.tools.fulltext.pubmed_fetch_by_pmids")
    def test_fulltext_without_metadata_calls_fetch(self, mock_pubmed, mock_pmc):
        """Without pre-supplied metadata, pubmed_fetch_by_pmids IS called."""
        mock_pmc.return_value = {
            "body_text": "Full text body.",
            "references": [],
            "pmcid": "PMC111",
        }
        mock_pubmed.return_value = [_SAMPLE_META]

        result = fetch_paper_text("12345", pmcid="PMC111")

        mock_pubmed.assert_called_once_with(["12345"])
        assert result["source_type"] == "full_text"
        assert result["title"] == "Test Paper Title"

    @patch("targetsearch.tools.fulltext.pmc_fulltext_fetch")
    @patch("targetsearch.tools.fulltext.pubmed_fetch_by_pmids")
    def test_abstract_without_metadata_calls_fetch(self, mock_pubmed, mock_pmc):
        """Without pre-supplied metadata and no PMC, falls back to abstract
        and calls pubmed_fetch_by_pmids."""
        mock_pubmed.return_value = [_SAMPLE_META]

        result = fetch_paper_text("12345", pmcid=None)

        mock_pubmed.assert_called_once_with(["12345"])
        assert result["source_type"] == "abstract"
        assert result["text"] == "This is the abstract text."

    @patch("targetsearch.tools.fulltext.pmc_fulltext_fetch")
    @patch("targetsearch.tools.fulltext.pubmed_fetch_by_pmids")
    def test_pmc_fails_falls_back_to_abstract_with_metadata(
        self, mock_pubmed, mock_pmc
    ):
        """PMC returns empty body → falls back to abstract from metadata."""
        mock_pmc.return_value = {"body_text": "", "references": []}

        result = fetch_paper_text("12345", pmcid="PMC111", metadata=_SAMPLE_META)

        mock_pubmed.assert_not_called()
        assert result["source_type"] == "abstract"
        assert result["text"] == "This is the abstract text."

    @patch("targetsearch.tools.fulltext.pubmed_fetch_by_pmids")
    def test_no_metadata_no_pubmed_result(self, mock_pubmed):
        """No metadata provided and PubMed returns nothing → not_found."""
        mock_pubmed.return_value = []

        result = fetch_paper_text("99999", pmcid=None)

        assert result["source_type"] == "not_found"
        assert result["text"] == ""
