"""Analyze pipeline timing and identify parallelization opportunities.

Usage:
    uv run python testrun-evaluation/analyze_timing.py logs/20260418_042405_70af98.log
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})")
TOOL_CALL_RE = re.compile(r"\[(\w+)\] Calling tool: (\w+)\(")
TOOL_DONE_RE = re.compile(r"\[(\w+)\] Tool (\w+) returned in ([\d.]+)s")
AGENT_START_RE = re.compile(r"\[(\w+)\] Starting run with message:")
AGENT_DONE_RE = re.compile(r"\[(\w+)\] Finished in ([\d.]+)s \((\d+) tool calls\)")
SYNTH_RE = re.compile(r"synthesize_disease_profile|run_feedback_agent|run_search_agent")


def parse_seconds(line: str) -> float | None:
    m = TIMESTAMP_RE.match(line)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


@dataclass
class ToolEvent:
    agent: str
    tool: str
    start_s: float
    duration_s: float = 0.0


@dataclass
class AgentEvent:
    name: str
    start_s: float
    end_s: float = 0.0
    duration_s: float = 0.0
    tool_calls: int = 0


def parse_log(path: Path) -> tuple[list[ToolEvent], list[AgentEvent]]:
    tools: list[ToolEvent] = []
    agents: list[AgentEvent] = []
    pending_tools: dict[tuple[str, str], ToolEvent] = {}
    pending_agents: dict[str, AgentEvent] = {}

    with open(path) as f:
        for line in f:
            ts = parse_seconds(line)
            if ts is None:
                continue

            m = TOOL_CALL_RE.search(line)
            if m:
                agent, tool = m.group(1), m.group(2)
                pending_tools[(agent, tool)] = ToolEvent(agent=agent, tool=tool, start_s=ts)
                continue

            m = TOOL_DONE_RE.search(line)
            if m:
                agent, tool, dur = m.group(1), m.group(2), float(m.group(3))
                key = (agent, tool)
                if key in pending_tools:
                    ev = pending_tools.pop(key)
                    ev.duration_s = dur
                    tools.append(ev)
                else:
                    tools.append(ToolEvent(agent=agent, tool=tool, start_s=ts, duration_s=dur))
                continue

            m = AGENT_START_RE.search(line)
            if m:
                name = m.group(1)
                pending_agents[name] = AgentEvent(name=name, start_s=ts)
                continue

            m = AGENT_DONE_RE.search(line)
            if m:
                name, dur, tc = m.group(1), float(m.group(2)), int(m.group(3))
                if name in pending_agents:
                    ev = pending_agents.pop(name)
                    ev.end_s = ts
                    ev.duration_s = dur
                    ev.tool_calls = tc
                    agents.append(ev)
                continue

    return tools, agents


def analyze_phases(agents: list[AgentEvent], tools: list[ToolEvent]) -> dict:
    """Identify pipeline phases from agent events."""
    phases = []
    for a in agents:
        phases.append({
            "agent": a.name,
            "duration_s": a.duration_s,
            "tool_calls": a.tool_calls,
        })

    # Tool timing by type
    tool_times: dict[str, list[float]] = defaultdict(list)
    for t in tools:
        tool_times[t.tool].append(t.duration_s)

    tool_summary = {}
    for tool, times in sorted(tool_times.items(), key=lambda x: sum(x[1]), reverse=True):
        tool_summary[tool] = {
            "count": len(times),
            "total_s": sum(times),
            "mean_s": sum(times) / len(times),
            "max_s": max(times),
            "min_s": min(times),
        }

    # Parallelization: paper summarization calls are independent
    paper_tools = [t for t in tools if t.tool in ("summarize_paper", "fetch_full_text")]
    paper_total = sum(t.duration_s for t in paper_tools)
    paper_max = max((t.duration_s for t in paper_tools), default=0)

    return {
        "phases": phases,
        "tool_summary": tool_summary,
        "parallelization": {
            "paper_processing_sequential_s": paper_total,
            "paper_processing_parallel_s": paper_max,
            "potential_savings_s": paper_total - paper_max if paper_max else 0,
            "paper_tool_count": len(paper_tools),
        },
    }


def print_report(analysis: dict, agents: list[AgentEvent]) -> None:
    print("=" * 80)
    print("TIMING ANALYSIS")
    print("=" * 80)

    total_wall = agents[0].duration_s if agents else 0
    for a in agents:
        if a.name == "disease_intel":
            total_wall = a.duration_s
            break

    print(f"\nTotal wall-clock time: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print()

    print("Pipeline phases:")
    print(f"  {'Agent':<20} {'Duration':>10} {'Tool calls':>10}")
    print("  " + "-" * 44)
    for p in analysis["phases"]:
        print(f"  {p['agent']:<20} {p['duration_s']:>9.1f}s {p['tool_calls']:>10}")

    print()
    print("Tool timing summary:")
    print(f"  {'Tool':<30} {'Count':>5} {'Total(s)':>9} {'Mean(s)':>8} {'Max(s)':>7}")
    print("  " + "-" * 63)
    for tool, d in analysis["tool_summary"].items():
        print(f"  {tool:<30} {d['count']:>5} {d['total_s']:>9.1f} "
              f"{d['mean_s']:>8.1f} {d['max_s']:>7.1f}")

    par = analysis["parallelization"]
    print()
    print("Parallelization opportunity (paper processing):")
    print(f"  Sequential total:  {par['paper_processing_sequential_s']:.1f}s")
    print(f"  Parallel estimate: {par['paper_processing_parallel_s']:.1f}s")
    print(f"  Potential savings: {par['potential_savings_s']:.1f}s "
          f"({par['paper_tool_count']} operations)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: analyze_timing.py <path-to-.log>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    tools, agents = parse_log(path)
    print(f"Parsed {len(tools)} tool events, {len(agents)} agent events from {path.name}")

    analysis = analyze_phases(agents, tools)
    print_report(analysis, agents)

    import json
    out_path = path.with_name(path.stem + "_timing_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n→ Analysis written to {out_path}")


if __name__ == "__main__":
    main()
