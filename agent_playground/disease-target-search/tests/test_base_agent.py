"""Tests for Agent base class — history compaction."""

from targetsearch.agents.base import Agent, _compact_tool_result
from targetsearch.core.config import config


class TestCompactToolResult:
    def test_json_list(self):
        content = '[{"pmid": "123", "title": "Paper A"}, {"pmid": "456", "title": "Paper B"}]'
        result = _compact_tool_result(content)
        assert "[compacted]" in result
        assert "2 items" in result

    def test_json_dict(self):
        content = '{"disease_name": "NSCLC", "pathways": [1, 2, 3]}'
        result = _compact_tool_result(content)
        assert "[compacted]" in result
        assert "Dict with keys" in result

    def test_pmid_extraction(self):
        content = "Found papers: PMID 12345678, PMID 23456789, PMID 34567890. " + "x" * 2000
        result = _compact_tool_result(content)
        assert "[compacted]" in result
        assert "3 PMIDs" in result
        assert "12345678" in result

    def test_generic_truncation(self):
        content = "A" * 5000
        result = _compact_tool_result(content)
        assert "[compacted]" in result
        assert len(result) < 500

    def test_short_content_unchanged_structure(self):
        content = "Short result"
        result = _compact_tool_result(content)
        assert "[compacted]" in result
        assert "Short result" in result


class TestHistoryCompaction:
    def _make_agent(self):
        agent = Agent(system_prompt="You are a test agent.")
        agent.name = "test"
        return agent

    def test_no_compaction_below_threshold(self):
        agent = self._make_agent()
        agent._messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "system", "content": "[context_summary]\nState"},
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "1", "content": "Short result"},
        ]
        original = [dict(m) for m in agent._messages]
        agent._compact_history()
        # Nothing should change — total chars well below threshold
        assert agent._messages == original

    def test_compaction_above_threshold(self):
        agent = self._make_agent()
        old_threshold = config.history_compaction_threshold
        try:
            config.history_compaction_threshold = 500  # low threshold for testing

            big_result = "X" * 2000
            agent._messages = [
                {"role": "system", "content": "System prompt"},
                {"role": "system", "content": "[context_summary]\nState"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": big_result},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
                {"role": "tool", "tool_call_id": "2", "content": big_result},
                # Recent messages (last 4) — should not be compacted
                {"role": "assistant", "content": None, "tool_calls": [{"id": "3"}]},
                {"role": "tool", "tool_call_id": "3", "content": big_result},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "4"}]},
                {"role": "tool", "tool_call_id": "4", "content": big_result},
            ]

            agent._compact_history()

            # Old tool results (indices 4, 6) should be compacted
            assert "[compacted]" in agent._messages[4]["content"]
            assert "[compacted]" in agent._messages[6]["content"]
            # Recent tool results (indices 8, 10) should NOT be compacted
            assert agent._messages[8]["content"] == big_result
            assert agent._messages[10]["content"] == big_result
        finally:
            config.history_compaction_threshold = old_threshold

    def test_compaction_preserves_system_messages(self):
        agent = self._make_agent()
        old_threshold = config.history_compaction_threshold
        try:
            config.history_compaction_threshold = 100

            agent._messages = [
                {"role": "system", "content": "System prompt " + "X" * 500},
                {"role": "system", "content": "[context_summary]\n" + "Y" * 500},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": "Z" * 2000},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
                {"role": "tool", "tool_call_id": "2", "content": "W" * 2000},
                # Recent (last 4)
                {"role": "assistant", "content": None, "tool_calls": [{"id": "3"}]},
                {"role": "tool", "tool_call_id": "3", "content": "recent"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "4"}]},
                {"role": "tool", "tool_call_id": "4", "content": "recent"},
            ]

            agent._compact_history()

            # System messages untouched
            assert "System prompt" in agent._messages[0]["content"]
            assert "[context_summary]" in agent._messages[1]["content"]
            # Old tool results compacted
            assert "[compacted]" in agent._messages[4]["content"]
            assert "[compacted]" in agent._messages[6]["content"]
        finally:
            config.history_compaction_threshold = old_threshold

    def test_disabled_when_threshold_zero(self):
        agent = self._make_agent()
        old_threshold = config.history_compaction_threshold
        try:
            config.history_compaction_threshold = 0

            big_result = "X" * 10000
            agent._messages = [
                {"role": "system", "content": "System"},
                {"role": "system", "content": "[context_summary]"},
                {"role": "user", "content": "Hello"},
                {"role": "tool", "tool_call_id": "1", "content": big_result},
            ]

            agent._compact_history()

            # Nothing compacted
            assert agent._messages[3]["content"] == big_result
        finally:
            config.history_compaction_threshold = old_threshold


class TestToolTagsFiltering:
    """Verify that tool_tags=[] means no tools, not all tools."""

    def test_empty_tool_tags_returns_no_tools(self):
        agent = Agent(system_prompt="test")
        agent.tool_tags = []
        assert agent.tools == []

    def test_none_tool_tags_returns_all_tools(self):
        agent = Agent(system_prompt="test")
        agent.tool_tags = None
        # None means "all tools" — should be non-empty
        assert len(agent.tools) > 0

    def test_feedback_agent_has_no_tools(self):
        from targetsearch.agents.feedback import FeedbackAgent
        fa = FeedbackAgent()
        assert fa.tools == []

    def test_searcher_agent_has_tools(self):
        from targetsearch.agents.searcher import SearcherAgent
        sa = SearcherAgent()
        assert len(sa.tools) > 0


class TestConfigPaperBudget:
    """Verify split paper budget config."""

    def test_max_papers_initial(self):
        assert config.max_papers_initial == 12

    def test_max_papers_gap_fill(self):
        assert config.max_papers_gap_fill == 8

    def test_max_papers_is_sum(self):
        assert config.max_papers == config.max_papers_initial + config.max_papers_gap_fill

    def test_max_papers_value(self):
        assert config.max_papers == 20

