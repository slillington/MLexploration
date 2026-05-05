"""Agent base class with a tool-calling loop and ActionContext integration.

This is the core abstraction: an agent is an LLM with a system prompt and
access to a subset of tools. The run() method implements a loop where the
LLM can call tools, observe results, and continue reasoning until it
produces a final text answer.

The loop uses OpenAI's function-calling protocol:
  1. Send messages (including tool schemas) to the LLM.
  2. If the LLM responds with tool_calls, execute them and append results.
  3. Repeat until the LLM responds with a text message (no tool_calls).

ActionContext flows through the loop: coordination tools that declare an
ActionContext parameter receive it via auto-injection. The context summary
is injected as a system message on each turn so the LLM can make informed
decisions about what to do next.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from targetsearch.core.config import config
from targetsearch.core.context import ActionContext
from targetsearch.core.llm import llm_call
from targetsearch.core.registry import ToolRegistry, registry

log = logging.getLogger(__name__)

# Name used for the injected context summary message so we can replace it
_CONTEXT_SUMMARY_MARKER = "[context_summary]"


def _compact_tool_result(content: str, max_len: int = 300) -> str:
    """Produce a compact summary of a tool result for history compaction.

    Tries to extract useful metadata (PMIDs, counts) from the content,
    then truncates to ``max_len`` chars.
    """
    import re

    # Try to parse as JSON for structured results
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return f"[compacted] List of {len(data)} items. First: {json.dumps(data[0], default=str)[:200]}"
        if isinstance(data, dict):
            keys = list(data.keys())[:6]
            return f"[compacted] Dict with keys: {keys}. Preview: {content[:200]}"
    except (json.JSONDecodeError, TypeError, IndexError):
        pass

    # Extract PMIDs if present
    pmids = re.findall(r"PMID[:\s]*(\d{7,8})", content)
    if pmids:
        pmid_str = ", ".join(pmids[:10])
        if len(pmids) > 10:
            pmid_str += f" (+{len(pmids) - 10} more)"
        return f"[compacted] {len(pmids)} PMIDs: {pmid_str}. {content[:150]}"

    # Generic truncation
    return f"[compacted] {content[:max_len]}"


class Agent:
    """Base agent with tool-calling capabilities and ActionContext support.

    Subclasses override:
      - system_prompt: str or property returning the system prompt
      - tool_tags: list of tags to filter the tool registry
      - parse_output(): optional post-processing of the final LLM text
    """

    name: str = "agent"
    tool_tags: list[str] = []

    def __init__(
        self,
        system_prompt: str = "",
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.registry = tool_registry or registry
        self._messages: list[dict] = []
        self._tool_call_count = 0
        self.context: ActionContext | None = None

    @property
    def tools(self) -> list[dict]:
        """OpenAI-compatible tool schemas for this agent's tool subset."""
        tags = self.tool_tags if self.tool_tags is not None else None
        return self.registry.tool_schemas(tags=tags)

    def run(
        self,
        user_message: str,
        context: ActionContext | None = None,
    ) -> str:
        """Execute the agent's reasoning loop.

        Args:
            user_message: The task or question for the agent.
            context: Shared ActionContext. Created automatically if not provided.

        Returns:
            The agent's final text response after all tool calls are complete.
        """
        self.context = context or ActionContext()
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        self._tool_call_count = 0

        log.info("[%s] Starting run with message: %s", self.name, user_message[:100])
        t0 = time.time()

        while True:
            # Inject/update context summary before each LLM call
            self._inject_context_summary()

            # Call the LLM with tool schemas
            tool_schemas = self.tools
            resp = llm_call(
                self._messages,
                tools=tool_schemas if tool_schemas else None,
                caller=self.name,
            )
            choice = resp.choices[0]
            msg = choice.message

            # Case 1: LLM wants to call tools
            if msg.tool_calls:
                # Append the assistant message (with tool_calls) to history
                self._messages.append(msg.model_dump())

                for tool_call in msg.tool_calls:
                    self._tool_call_count += 1
                    self.context.metadata.tool_call_count += 1

                    if self._tool_call_count > config.max_tool_calls_per_turn:
                        log.warning(
                            "[%s] Hit tool call limit (%d)",
                            self.name,
                            config.max_tool_calls_per_turn,
                        )
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "ERROR: Tool call limit reached. Produce your final answer now.",
                        })
                        continue

                    result = self._execute_tool_call(tool_call)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                n_tool_results = len(msg.tool_calls)
                self.context.metadata.iteration_count += 1

                # Compact old tool results if history is getting large
                if config.history_compaction_threshold > 0:
                    self._compact_history(n_tool_results)

                continue  # Loop back to let the LLM process tool results

            # Case 2: LLM produced a text response — we're done
            elapsed = time.time() - t0
            log.info(
                "[%s] Finished in %.1fs (%d tool calls)",
                self.name,
                elapsed,
                self._tool_call_count,
            )
            final_text = msg.content or ""
            return self.parse_output(final_text)

    def _inject_context_summary(self) -> None:
        """Insert or update the context summary as a system message.

        Placed right after the initial system prompt so the LLM always
        has current state awareness. Replaces any previous summary.
        """
        if self.context is None:
            return

        summary_content = (
            f"{_CONTEXT_SUMMARY_MARKER}\n"
            f"Current pipeline state:\n{self.context.summarize()}"
        )

        # Look for an existing summary message to replace
        for i, msg in enumerate(self._messages):
            if (
                msg.get("role") == "system"
                and _CONTEXT_SUMMARY_MARKER in msg.get("content", "")
            ):
                self._messages[i] = {"role": "system", "content": summary_content}
                return

        # Insert after the first system message
        self._messages.insert(1, {"role": "system", "content": summary_content})

    def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute a single tool call and return the result as a string."""
        fn_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return f"ERROR: Could not parse arguments for {fn_name}"

        log.info("[%s] Calling tool: %s(%s)", self.name, fn_name, arguments)

        try:
            t0 = time.time()
            result = self.registry.call_tool(
                fn_name, arguments, context=self.context
            )
            elapsed = time.time() - t0

            # Record search queries for deduplication across passes
            if self.context is not None and "query" in arguments:
                try:
                    tool_spec = self.registry.get_tool(fn_name)
                    if "literature" in tool_spec.tags:
                        query_entry = f"{fn_name}: {arguments['query']}"
                        self.context.search_state.queries_executed.append(
                            query_entry
                        )
                except KeyError:
                    pass

            # Serialize the result for the LLM
            if isinstance(result, str):
                serialized = result
            else:
                serialized = json.dumps(result, indent=2, default=str)
            log.debug(
                "[%s] Tool %s returned in %.1fs (%d chars)",
                self.name, fn_name, elapsed, len(serialized),
            )
            return serialized
        except KeyError:
            return f"ERROR: Unknown tool '{fn_name}'"
        except Exception as e:
            log.exception("[%s] Tool %s failed", self.name, fn_name)
            return f"ERROR: {type(e).__name__}: {e}"

    def _compact_history(self, n_latest_tool_results: int = 1) -> None:
        """Compact old tool results when message history exceeds the threshold.

        Keeps the system prompt(s) and the most recent messages intact,
        then replaces older tool result messages with a one-line summary.

        ``n_latest_tool_results`` is the number of tool results just
        appended in the current turn.  We keep at least that many + 1
        (for the preceding assistant message) uncompacted so the model
        can read results it hasn't seen yet.
        """
        # Protect the assistant message + all its tool results from this turn
        keep_recent = max(4, n_latest_tool_results + 1)

        total_chars = sum(len(m.get("content", "") or "") for m in self._messages)
        if total_chars <= config.history_compaction_threshold:
            return

        # Find the boundary: keep system messages and the last N messages
        # Everything between system messages and the tail is eligible
        compactable_end = len(self._messages) - keep_recent
        if compactable_end <= 2:  # nothing worth compacting
            return

        compacted = 0
        for i in range(2, compactable_end):  # skip system + context summary
            msg = self._messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "") or ""
            if len(content) <= 500:  # already small
                continue
            self._messages[i] = {
                **msg,
                "content": _compact_tool_result(content),
            }
            compacted += 1

        if compacted:
            new_total = sum(len(m.get("content", "") or "") for m in self._messages)
            log.debug(
                "[%s] Compacted %d tool results: %d → %d chars",
                self.name, compacted, total_chars, new_total,
            )

    def parse_output(self, text: str) -> str:
        """Post-process the LLM's final text output.

        Override in subclasses to parse structured output (e.g., JSON → Pydantic).
        Default implementation returns the text as-is.
        """
        return text

    @property
    def message_history(self) -> list[dict]:
        """Access the full message history (useful for debugging)."""
        return list(self._messages)
