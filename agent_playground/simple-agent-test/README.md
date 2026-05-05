# Simple Agent Test

A teaching demonstration of the **orchestrator + subagent** agentic AI pattern, where every decision — skill routing, task decomposition, code generation, and report synthesis — is made by an LLM.

## Overview

The system receives a natural-language query, discovers relevant capabilities from a file-based skill registry, spawns parallel code-writing subagents (one per skill), executes their output, and synthesizes a final report.

**Default demo query:** _"Read a FASTA file containing two amino acid sequences and compare their alignments in amino acid space and in embedding space."_

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│                 ORCHESTRATOR                      │
│                                                  │
│  1. Load skill registry (skills/*/SKILL.md)      │
│  2. LLM: select relevant skills                  │
│  3. LLM: decompose query into per-skill tasks    │
│  4. Spawn subagents in parallel (threads)        │
│  5. Collect results                              │
│  6. LLM: synthesize final report                 │
└───────┬──────────────────────────────┬───────────┘
        │ (thread 1)                   │ (thread 2)
 ┌──────▼──────┐                ┌──────▼──────┐
 │  SUBAGENT   │                │  SUBAGENT   │
 │             │                │             │
 │ Read SKILL.md docs           │ Read SKILL.md docs
 │ LLM → write code            │ LLM → write code
 │ exec(code)  │                │ exec(code)  │
 │ capture stdout               │ capture stdout
 └─────────────┘                └─────────────┘
```

## Structure

```
simple-agent-test/
├── agentic_demo.py        # Full orchestrator + subagent implementation
└── skills/
    ├── biopython/         # Sequence alignment & file parsing
    │   ├── SKILL.md
    │   └── references/
    ├── esm/               # Protein language model embeddings (ESM C/ESM3)
    │   ├── SKILL.md
    │   └── references/
    ├── paper-lookup/      # Academic paper search across 10 databases
    │   ├── SKILL.md
    │   └── references/
    └── scikit-learn/      # Machine learning utilities
        ├── SKILL.md
        ├── references/
        └── scripts/
```

## Skills

| Skill | Description |
|-------|-------------|
| **biopython** | Sequence manipulation, FASTA parsing, pairwise alignment (BLOSUM62), NCBI access |
| **esm** | Protein language models — ESM3 for generative design, ESM C for embeddings & representations |
| **paper-lookup** | Search 10 academic databases (PubMed, Semantic Scholar, bioRxiv, arXiv, OpenAlex, etc.) |
| **scikit-learn** | General-purpose machine learning (classification, clustering, dimensionality reduction) |

## How It Works

1. **Skill Discovery** — SKILL.md files are loaded from `skills/` at startup; YAML front-matter provides descriptions for LLM routing.
2. **Skill Matching** — The LLM is given the query + skill catalog and returns a JSON list of relevant skill names.
3. **Task Decomposition** — The LLM splits the user query into focused sub-tasks (one per matched skill) so subagents stay in scope.
4. **Parallel Subagents** — Each subagent reads its SKILL.md, asks the LLM to write a Python script, then `exec()`s it and captures stdout. Threads via `ThreadPoolExecutor`.
5. **Synthesis** — All subagent outputs are sent to the LLM, which produces a comparative scientific report.

## Usage

```bash
cd simple-agent-test
python agentic_demo.py
```

Expects a `sequences.fasta` file in the working directory (or parent — see `fasta_file` in `__main__`).

## Requirements

- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM API
- A valid `GITHUB_TOKEN` (model: `github_copilot/gpt-4o`)
- Python 3.10+
- Skill-specific libraries: `biopython`, `esm`, `torch`, `scikit-learn`, `numpy`

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Threads (not asyncio) | Simplest model for I/O-bound LLM calls; shared memory avoids serialization |
| File-based skill registry | Mirrors real agentic frameworks (LangChain, CrewAI) — capabilities discovered at runtime |
| LLM-generated code via `exec()` | Core agentic pattern — same approach used by Copilot, Cursor, and Devin |
| `as_completed()` | Process fast subagents immediately without waiting for the slowest |

> **Safety note:** `exec()` runs LLM-generated code unsandboxed. In production, use Docker, E2B, or Modal for isolation.
