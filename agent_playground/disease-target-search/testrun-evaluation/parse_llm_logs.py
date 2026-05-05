"""Parse .llm.log files into structured JSONL and summary CSV.

Usage:
    uv run python testrun-evaluation/parse_llm_logs.py logs/20260418_042405_70af98.llm.log
"""

import csv
import json
import re
import sys
from pathlib import Path

LOG_LINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\s+\[.*?\]\s+\w+\s+\w+\s+({.*})$")


def parse_log(path: Path) -> list[dict]:
    """Extract JSON payloads from timestamped log lines, pair requests with responses."""
    entries: list[dict] = []
    pending: dict[int, dict] = {}  # seq -> request

    with open(path) as f:
        for line in f:
            m = LOG_LINE_RE.match(line.strip())
            if not m:
                continue
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

            seq = d.get("seq")
            if d.get("event") == "llm_request":
                pending[seq] = d
            elif d.get("event") == "llm_response" and seq in pending:
                req = pending.pop(seq)
                entry = {
                    "seq": seq,
                    "caller": d["caller"],
                    "model": req.get("model", ""),
                    "n_messages": req.get("n_messages", 0),
                    "n_tools": req.get("n_tools", 0),
                    "prompt_tokens": d["usage"]["prompt_tokens"],
                    "completion_tokens": d["usage"]["completion_tokens"],
                    "total_tokens": d["usage"]["total_tokens"],
                    "elapsed_s": d["elapsed_s"],
                    "finish_reason": d.get("finish_reason", ""),
                    "has_tool_calls": d.get("has_tool_calls", False),
                    "n_tool_calls": len(d["tool_calls"]) if d.get("tool_calls") else 0,
                    # Keep full messages/content for downstream analysis
                    "request_messages": req.get("messages", []),
                    "response_content": d.get("content"),
                    "response_tool_calls": d.get("tool_calls"),
                }
                entries.append(entry)

    return entries


def normalize_caller(caller: str) -> str:
    if caller.startswith("summarize_paper"):
        return "summarize_paper"
    if caller.startswith("synthesize_batch"):
        return "synthesize_batch"
    return caller


def write_jsonl(entries: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")


def write_summary_csv(entries: list[dict], out_path: Path) -> None:
    rows = []
    for e in entries:
        rows.append({
            "seq": e["seq"],
            "caller": e["caller"],
            "caller_group": normalize_caller(e["caller"]),
            "model": e["model"],
            "n_messages": e["n_messages"],
            "n_tools": e["n_tools"],
            "prompt_tokens": e["prompt_tokens"],
            "completion_tokens": e["completion_tokens"],
            "total_tokens": e["total_tokens"],
            "elapsed_s": e["elapsed_s"],
            "finish_reason": e["finish_reason"],
            "has_tool_calls": e["has_tool_calls"],
            "n_tool_calls": e["n_tool_calls"],
        })

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_llm_logs.py <path-to-.llm.log>", file=sys.stderr)
        sys.exit(1)

    log_path = Path(sys.argv[1])
    out_dir = log_path.parent
    stem = log_path.stem.replace(".llm", "")

    entries = parse_log(log_path)
    print(f"Parsed {len(entries)} request/response pairs from {log_path.name}")

    jsonl_path = out_dir / f"{stem}_llm_calls.jsonl"
    write_jsonl(entries, jsonl_path)
    print(f"  → {jsonl_path}")

    csv_path = out_dir / f"{stem}_llm_calls_summary.csv"
    write_summary_csv(entries, csv_path)
    print(f"  → {csv_path}")


if __name__ == "__main__":
    main()
