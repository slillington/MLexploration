"""Provider-agnostic LLM client.

Wraps litellm so the rest of the codebase never imports it directly.
Swap models by changing config.model — everything else stays the same.

LLM inputs and outputs are logged at DEBUG to the ``targetsearch.llm_io``
logger (configured by ``setup_logging``).  Each log entry is a JSON object
so you can parse the .llm.log file programmatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

import litellm

from targetsearch.core.config import config

# Keep litellm quiet
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")

log = logging.getLogger(__name__)
_llm_log = logging.getLogger("targetsearch.llm_io")

# Running counter so you can correlate request → response in the log.
# Thread-safe because summarize_paper calls run in parallel.
_call_seq = 0
_call_seq_lock = threading.Lock()


def _log_llm_request(
    seq: int,
    messages: list[dict],
    tools: list[dict] | None,
    caller: str,
    model: str = "",
) -> None:
    """Log the prompt sent to the LLM."""
    # Truncate large tool-result messages to keep the log readable
    compact_msgs = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + f"... [{len(content)} chars total]"
        compact_msgs.append({**m, "content": content})

    entry = {
        "event": "llm_request",
        "seq": seq,
        "caller": caller,
        "model": model,
        "n_messages": len(messages),
        "has_tools": bool(tools),
        "n_tools": len(tools) if tools else 0,
        "messages": compact_msgs,
    }
    _llm_log.debug("LLM_REQUEST %s", json.dumps(entry, default=str))


def _log_llm_response(
    seq: int,
    resp: litellm.ModelResponse,
    elapsed: float,
    caller: str,
) -> None:
    """Log the LLM response including token usage."""
    choice = resp.choices[0]
    msg = choice.message

    # Extract token usage from the response
    usage = {}
    if hasattr(resp, "usage") and resp.usage:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

    # Capture tool calls if present
    tool_calls_summary = None
    if msg.tool_calls:
        tool_calls_summary = [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in msg.tool_calls
        ]

    # Capture text content
    content = msg.content
    if isinstance(content, str) and len(content) > 5000:
        content = content[:5000] + f"... [{len(content)} chars total]"

    entry = {
        "event": "llm_response",
        "seq": seq,
        "caller": caller,
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "finish_reason": choice.finish_reason,
        "has_tool_calls": bool(msg.tool_calls),
        "tool_calls": tool_calls_summary,
        "content": content,
    }
    _llm_log.debug("LLM_RESPONSE %s", json.dumps(entry, default=str))


def llm_call(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    caller: str = "",
) -> litellm.ModelResponse:
    """Make a single LLM call and return the full response object.

    When `tools` is provided, the response may contain tool_calls that the
    caller is responsible for dispatching.

    Args:
        model:  Override the default model for this call.  Falls back to
                ``config.model`` when *None*.
        caller: Free-form label identifying who made this call (e.g.
                agent name or tool name).  Appears in the LLM I/O log.
    """
    global _call_seq
    with _call_seq_lock:
        _call_seq += 1
        seq = _call_seq

    effective_model = model or config.model

    kwargs: dict[str, Any] = {
        "model": effective_model,
        "messages": messages,
        "max_tokens": max_tokens or config.max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    _log_llm_request(seq, messages, tools, caller, model=effective_model)

    t0 = time.time()
    resp = litellm.completion(**kwargs)
    elapsed = time.time() - t0

    _log_llm_response(seq, resp, elapsed, caller)

    return resp


def llm_text(
    messages: list[dict], caller: str = "", model: str | None = None,
) -> str:
    """Convenience wrapper that returns just the text content."""
    resp = llm_call(messages, caller=caller, model=model)
    return resp.choices[0].message.content.strip()


def parse_json_response(text: str) -> Any:
    """Extract JSON from an LLM response, tolerating common issues.

    Handles: markdown fences, trailing commas, truncated output.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before } or ] (common LLM mistake)
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Handle truncated JSON — try to close open braces/brackets
    repaired = _repair_truncated_json(fixed)
    return json.loads(repaired)


def _repair_truncated_json(text: str) -> str:
    """Attempt to close unclosed braces/brackets in truncated JSON."""
    # Track open delimiters, ignoring those inside strings
    open_stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            open_stack.append(ch)
        elif ch == "}" and open_stack and open_stack[-1] == "{":
            open_stack.pop()
        elif ch == "]" and open_stack and open_stack[-1] == "[":
            open_stack.pop()

    # Strip any trailing partial value (e.g., a truncated string)
    result = text.rstrip()
    if open_stack:
        # Remove trailing partial tokens: incomplete strings, trailing commas
        result = re.sub(r',?\s*"[^"]*$', "", result)  # partial string value
        result = re.sub(r",\s*$", "", result)  # trailing comma

    # Close remaining open delimiters
    closers = {"[": "]", "{": "}"}
    for opener in reversed(open_stack):
        result += closers[opener]

    return result
