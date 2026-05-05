"""Analyze token usage from parsed LLM call JSONL.

Usage:
    uv run python testrun-evaluation/analyze_token_usage.py logs/20260418_042405_70af98_llm_calls.jsonl
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


def analyze(entries: list[dict]) -> dict:
    total_prompt = sum(e["prompt_tokens"] for e in entries)
    total_completion = sum(e["completion_tokens"] for e in entries)
    total_elapsed = sum(e["elapsed_s"] for e in entries)

    # Per-caller-group aggregation
    by_caller: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "prompt": 0, "completion": 0, "elapsed": 0.0,
                 "max_prompt": 0, "max_completion": 0, "max_elapsed": 0.0}
    )
    for e in entries:
        g = normalize_caller(e["caller"])
        d = by_caller[g]
        d["count"] += 1
        d["prompt"] += e["prompt_tokens"]
        d["completion"] += e["completion_tokens"]
        d["elapsed"] += e["elapsed_s"]
        d["max_prompt"] = max(d["max_prompt"], e["prompt_tokens"])
        d["max_completion"] = max(d["max_completion"], e["completion_tokens"])
        d["max_elapsed"] = max(d["max_elapsed"], e["elapsed_s"])

    # Top-N largest calls
    by_prompt = sorted(entries, key=lambda e: e["prompt_tokens"], reverse=True)[:10]
    by_time = sorted(entries, key=lambda e: e["elapsed_s"], reverse=True)[:10]

    return {
        "total_calls": len(entries),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "total_elapsed_s": total_elapsed,
        "by_caller": dict(by_caller),
        "top_by_prompt": [
            {"seq": e["seq"], "caller": e["caller"],
             "prompt": e["prompt_tokens"], "completion": e["completion_tokens"],
             "elapsed": e["elapsed_s"]}
            for e in by_prompt
        ],
        "top_by_time": [
            {"seq": e["seq"], "caller": e["caller"],
             "prompt": e["prompt_tokens"], "completion": e["completion_tokens"],
             "elapsed": e["elapsed_s"]}
            for e in by_time
        ],
    }


def print_report(analysis: dict) -> None:
    print("=" * 80)
    print("TOKEN USAGE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total LLM calls:       {analysis['total_calls']}")
    print(f"Total prompt tokens:   {analysis['total_prompt_tokens']:>12,}")
    print(f"Total completion tokens:{analysis['total_completion_tokens']:>11,}")
    print(f"Total tokens:          {analysis['total_tokens']:>12,}")
    print(f"Total LLM time:        {analysis['total_elapsed_s']:>10.1f}s "
          f"({analysis['total_elapsed_s']/60:.1f} min)")
    print()

    # By caller
    print(f"{'Caller':<25} {'Calls':>5} {'Prompt':>10} {'Compl':>10} "
          f"{'Total':>10} {'Time(s)':>8} {'MaxPr':>8} {'MaxT(s)':>7}")
    print("-" * 93)
    for caller in sorted(analysis["by_caller"],
                         key=lambda c: analysis["by_caller"][c]["prompt"],
                         reverse=True):
        d = analysis["by_caller"][caller]
        total = d["prompt"] + d["completion"]
        print(f"{caller:<25} {d['count']:>5} {d['prompt']:>10,} {d['completion']:>10,} "
              f"{total:>10,} {d['elapsed']:>8.1f} {d['max_prompt']:>8,} {d['max_elapsed']:>7.1f}")

    print()
    print("Top 10 calls by prompt tokens:")
    for i, e in enumerate(analysis["top_by_prompt"], 1):
        print(f"  {i:>2}. seq={e['seq']:>3} {e['caller']:<35} "
              f"prompt={e['prompt']:>8,}  compl={e['completion']:>6,}  {e['elapsed']:>6.1f}s")

    print()
    print("Top 10 calls by elapsed time:")
    for i, e in enumerate(analysis["top_by_time"], 1):
        print(f"  {i:>2}. seq={e['seq']:>3} {e['caller']:<35} "
              f"{e['elapsed']:>6.1f}s  prompt={e['prompt']:>8,}  compl={e['completion']:>6,}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: analyze_token_usage.py <path-to-jsonl>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    entries = load_entries(path)
    analysis = analyze(entries)

    print_report(analysis)

    # Write JSON for downstream use
    out_path = path.with_name(path.stem.replace("_llm_calls", "") + "_token_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n→ Analysis written to {out_path}")


if __name__ == "__main__":
    main()
