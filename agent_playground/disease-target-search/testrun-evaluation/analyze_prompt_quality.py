"""Analyze prompt and response quality from parsed LLM call JSONL.

Examines:
- System prompt length and structure
- User message growth across conversation turns
- Response format compliance (JSON vs free text)
- Tool call patterns per caller
- Prompt-to-completion ratio (efficiency)

Usage:
    uv run python testrun-evaluation/analyze_prompt_quality.py logs/20260418_042405_70af98_llm_calls.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_entries(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_caller(caller: str) -> str:
    if caller.startswith("summarize_paper"):
        return "summarize_paper"
    if caller.startswith("synthesize_batch"):
        return "synthesize_batch"
    return caller


def analyze_prompts(entries: list[dict]) -> dict:
    """Analyze prompt structure and growth patterns."""
    by_caller: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        g = normalize_caller(e["caller"])
        msgs = e.get("request_messages", [])
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]

        system_chars = sum(len(m.get("content", "") or "") for m in system_msgs)
        user_chars = sum(len(m.get("content", "") or "") for m in user_msgs)
        total_chars = sum(len(m.get("content", "") or "") for m in msgs)

        by_caller[g].append({
            "seq": e["seq"],
            "n_messages": e["n_messages"],
            "system_chars": system_chars,
            "user_chars": user_chars,
            "total_chars": total_chars,
            "n_system": len(system_msgs),
            "n_user": len(user_msgs),
            "n_assistant": len(assistant_msgs),
            "n_tool": len(tool_msgs),
            "prompt_tokens": e["prompt_tokens"],
            "completion_tokens": e["completion_tokens"],
        })

    results = {}
    for caller, calls in by_caller.items():
        prompt_tokens = [c["prompt_tokens"] for c in calls]
        completion_tokens = [c["completion_tokens"] for c in calls]
        total_chars = [c["total_chars"] for c in calls]
        n_messages = [c["n_messages"] for c in calls]

        results[caller] = {
            "count": len(calls),
            "prompt_token_range": [min(prompt_tokens), max(prompt_tokens)],
            "prompt_token_mean": sum(prompt_tokens) / len(prompt_tokens),
            "completion_token_mean": sum(completion_tokens) / len(completion_tokens),
            "prompt_to_completion_ratio": (
                sum(prompt_tokens) / max(sum(completion_tokens), 1)
            ),
            "message_count_range": [min(n_messages), max(n_messages)],
            "char_range": [min(total_chars), max(total_chars)],
            "context_growth": (
                total_chars[-1] / max(total_chars[0], 1)
                if len(total_chars) > 1 else 1.0
            ),
        }

    return results


def analyze_responses(entries: list[dict]) -> dict:
    """Analyze response format compliance and tool call patterns."""
    json_responses = 0
    text_responses = 0
    tool_call_responses = 0
    empty_responses = 0

    tool_call_counts: dict[str, list[int]] = defaultdict(list)

    for e in entries:
        content = e.get("response_content")
        tool_calls = e.get("response_tool_calls")
        g = normalize_caller(e["caller"])

        if tool_calls:
            tool_call_responses += 1
            tool_call_counts[g].append(len(tool_calls))

        if content:
            content = content.strip()
            if content.startswith("{") or content.startswith("["):
                json_responses += 1
            else:
                text_responses += 1
        elif not tool_calls:
            empty_responses += 1

    return {
        "json_responses": json_responses,
        "text_responses": text_responses,
        "tool_call_responses": tool_call_responses,
        "empty_responses": empty_responses,
        "tool_calls_per_caller": {
            caller: {
                "count": len(counts),
                "mean": sum(counts) / len(counts),
                "max": max(counts),
            }
            for caller, counts in tool_call_counts.items()
        },
    }


def analyze_caller_patterns(entries: list[dict]) -> dict:
    """Analyze conversation patterns per caller group."""
    by_caller: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        g = normalize_caller(e["caller"])
        by_caller[g].append(e)

    patterns = {}
    for caller, calls in by_caller.items():
        seqs = [c["seq"] for c in calls]
        total_prompt = sum(c["prompt_tokens"] for c in calls)
        total_completion = sum(c["completion_tokens"] for c in calls)
        total_elapsed = sum(c["elapsed_s"] for c in calls)

        # Detect multi-turn conversations (sequential seqs for same caller)
        turns = 1
        for i in range(1, len(seqs)):
            if seqs[i] - seqs[i - 1] <= 2:  # allow small gaps
                turns += 1

        patterns[caller] = {
            "total_calls": len(calls),
            "estimated_turns": turns,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_elapsed_s": total_elapsed,
            "tokens_per_second": total_completion / max(total_elapsed, 0.1),
            "efficiency_ratio": total_completion / max(total_prompt, 1),
        }

    return patterns


def print_report(prompt_analysis: dict, response_analysis: dict,
                 caller_patterns: dict) -> None:
    print("=" * 80)
    print("PROMPT & RESPONSE QUALITY ANALYSIS")
    print("=" * 80)

    print("\n--- Prompt Structure by Caller ---")
    print(f"  {'Caller':<25} {'Calls':>5} {'PromptTok':>12} {'ComplTok':>10} "
          f"{'P:C Ratio':>9} {'MsgRange':>10} {'CtxGrowth':>9}")
    print("  " + "-" * 86)
    for caller in sorted(prompt_analysis,
                         key=lambda c: prompt_analysis[c]["prompt_token_mean"],
                         reverse=True):
        d = prompt_analysis[caller]
        pr = d["prompt_token_range"]
        mr = d["message_count_range"]
        print(f"  {caller:<25} {d['count']:>5} "
              f"{d['prompt_token_mean']:>10,.0f}  {d['completion_token_mean']:>10,.0f} "
              f"{d['prompt_to_completion_ratio']:>8.1f}x "
              f"{mr[0]:>3}-{mr[1]:<3} "
              f"{d['context_growth']:>8.1f}x")

    print("\n--- Response Format ---")
    ra = response_analysis
    print(f"  JSON responses:      {ra['json_responses']}")
    print(f"  Text responses:      {ra['text_responses']}")
    print(f"  Tool-call responses: {ra['tool_call_responses']}")
    print(f"  Empty responses:     {ra['empty_responses']}")

    if ra["tool_calls_per_caller"]:
        print("\n  Tool calls per response by caller:")
        for caller, d in sorted(ra["tool_calls_per_caller"].items(),
                                key=lambda x: x[1]["mean"], reverse=True):
            print(f"    {caller:<25} mean={d['mean']:.1f}  max={d['max']}")

    print("\n--- Caller Efficiency ---")
    print(f"  {'Caller':<25} {'Calls':>5} {'Prompt':>10} {'Compl':>10} "
          f"{'Eff%':>6} {'Tok/s':>6} {'Time(s)':>8}")
    print("  " + "-" * 76)
    for caller in sorted(caller_patterns,
                         key=lambda c: caller_patterns[c]["total_prompt_tokens"],
                         reverse=True):
        d = caller_patterns[caller]
        eff = d["efficiency_ratio"] * 100
        print(f"  {caller:<25} {d['total_calls']:>5} "
              f"{d['total_prompt_tokens']:>10,} {d['total_completion_tokens']:>10,} "
              f"{eff:>5.1f}% {d['tokens_per_second']:>6.1f} {d['total_elapsed_s']:>8.1f}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: analyze_prompt_quality.py <path-to-jsonl>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    entries = load_entries(path)

    prompt_analysis = analyze_prompts(entries)
    response_analysis = analyze_responses(entries)
    caller_patterns = analyze_caller_patterns(entries)

    print_report(prompt_analysis, response_analysis, caller_patterns)

    # Write combined analysis JSON
    out_path = path.with_name(
        path.stem.replace("_llm_calls", "") + "_prompt_analysis.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "prompt_analysis": prompt_analysis,
            "response_analysis": response_analysis,
            "caller_patterns": caller_patterns,
        }, f, indent=2)
    print(f"\n→ Analysis written to {out_path}")


if __name__ == "__main__":
    main()
