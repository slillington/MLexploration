"""
agentic_demo.py — A teaching demonstration of agentic AI architecture in Python.

=== WHAT THIS SCRIPT TEACHES ===

This script implements a complete "orchestrator + subagent" pattern where every
decision is made by an LLM (github_copilot/gpt-4o via LiteLLM):

  1. An ORCHESTRATOR receives a user query in natural language.
  2. It loads a registry of available SKILLS from disk (skills/*/SKILL.md).
  3. It asks the LLM which skills are relevant to the query.
  4. It spawns SUBAGENTS — one per skill — on parallel threads.
  5. Each subagent asks the LLM to write Python code for its task, then executes it.
  6. The orchestrator collects all results and asks the LLM to synthesize a report.

The example query: "Read a FASTA file containing two amino acid sequences and compare
their alignments in amino acid space and in embedding space."

This requires two skills:
  - biopython: parse the FASTA file and perform pairwise sequence alignment (BLOSUM62)
  - esm:       generate protein language model embeddings and compute cosine similarity

Both subagents run in parallel on separate threads.

=== ARCHITECTURE DIAGRAM ===

    ┌─────────────────────────────────────────────────┐
    │                  USER QUERY                     │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │               ORCHESTRATOR                       │
    │                                                  │
    │  1. Load skill registry from skills/*/SKILL.md   │
    │  2. LLM call: "which skills match this query?"   │
    │  3. Spawn subagents on parallel threads          │
    │  4. Collect results                              │
    │  5. LLM call: "synthesize a final report"        │
    └───────┬──────────────────────────┬───────────────┘
            │  (thread 1)             │  (thread 2)
     ┌──────▼──────┐           ┌──────▼──────┐
     │  SUBAGENT   │           │  SUBAGENT   │
     │  (biopython)│           │    (esm)    │
     │             │           │             │
     │ LLM: write  │           │ LLM: write  │
     │ code for    │           │ code for    │
     │ alignment   │           │ embedding   │
     │             │           │             │
     │ exec(code)  │           │ exec(code)  │
     │ return      │           │ return      │
     └──────┬──────┘           └──────┬──────┘
            │                          │
            └──────────┬───────────────┘
                       ▼
    ┌──────────────────────────────────────────────────┐
    │         LLM SYNTHESIZED REPORT                   │
    └──────────────────────────────────────────────────┘

=== KEY DESIGN DECISIONS ===

WHY THREADS (not asyncio, not multiprocessing)?
  Threads are the simplest concurrency model for I/O-bound work (LLM API calls).
  They share memory so results come back without serialization. For CPU-heavy work
  you'd use multiprocessing; for many concurrent API calls, asyncio scales better.
  Threads are the right teaching choice: easy to read, easy to debug.

WHY A SKILL REGISTRY (not hardcoded logic)?
  Real agentic systems discover capabilities at runtime. Loading SKILL.md files
  from a directory mirrors how frameworks like LangChain or CrewAI register tools.

WHY LLM-GENERATED CODE (not hardcoded functions)?
  This is the core agentic pattern: the LLM reads documentation (SKILL.md),
  reasons about the task, and writes executable code. The subagent is a
  "code-writing agent" — the same pattern used by Copilot, Cursor, and Devin.

SAFETY NOTE:
  This demo uses exec() to run LLM-generated code. In production, you'd sandbox
  this (Docker container, E2B, Modal, etc.). For a teaching demo it's fine.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import textwrap
import traceback
from io import StringIO
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any

import litellm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "github_copilot/gpt-4o"

# Suppress litellm's verbose logging so our output is readable
litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"


def llm_call(messages: list[dict], temperature: float = 0.0) -> str:
    """
    Single wrapper around all LLM calls. Every interaction with the model
    goes through here, making it easy to swap providers, add retries, or
    add logging in one place.
    """
    resp = litellm.completion(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=4096,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# SECTION 1: SKILL REGISTRY
# ---------------------------------------------------------------------------
# A "skill" is a capability the system knows about. Each skill lives in a
# subdirectory under skills/ and is described by a SKILL.md file. The
# orchestrator reads these at startup to build a searchable registry.
#
# In production, skills might come from a plugin API, a database, or a
# tool-use schema (like OpenAI function calling).
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """
    Represents one discoverable skill.

    Attributes:
        name:        Short identifier (directory name, e.g. "biopython").
        description: One-line summary from the SKILL.md front-matter.
        skill_dir:   Path to the skill directory (contains SKILL.md + references/).
    """
    name: str
    description: str
    skill_dir: Path


def load_skill_registry(skills_root: Path) -> list[Skill]:
    """
    Scan a directory for skills. Each subdirectory containing a SKILL.md
    is treated as a skill. We parse the YAML front-matter to extract the
    'description' field.

    This is intentionally simple — no YAML library needed, just regex.
    """
    skills = []
    for child in sorted(skills_root.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            text = skill_md.read_text()
            # Extract 'description:' from YAML front-matter between --- fences
            match = re.search(
                r"^---\s*\n.*?^description:\s*(.+?)(?:\n[a-z]|\n---)",
                text,
                re.MULTILINE | re.DOTALL,
            )
            desc = match.group(1).strip() if match else "(no description)"
            skills.append(Skill(name=child.name, description=desc, skill_dir=child))
    return skills


def print_registry(skills: list[Skill]) -> None:
    """Pretty-print the discovered skills."""
    print("=" * 70)
    print("SKILL REGISTRY — discovered skills:")
    print("=" * 70)
    for s in skills:
        short = s.description[:90] + "…" if len(s.description) > 90 else s.description
        print(f"  [{s.name:15s}] {short}")
    print()


# ---------------------------------------------------------------------------
# SECTION 2: LLM-BASED SKILL MATCHING
# ---------------------------------------------------------------------------
# The orchestrator asks the LLM: "Given these skill descriptions and this
# user query, which skills are needed?" The LLM returns a JSON list of
# skill names. This replaces naive keyword-matching with genuine natural
# language understanding.
# ---------------------------------------------------------------------------


def match_skills_via_llm(query: str, skills: list[Skill]) -> list[Skill]:
    """
    Ask the LLM which skills from the registry are relevant to the query.

    Returns the matched Skill objects in the order the LLM recommends.
    """
    # Build a concise skill catalog for the prompt
    catalog = "\n".join(
        f"- {s.name}: {s.description[:200]}" for s in skills
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a skill-routing agent. Given a user query and a list of "
                "available skills, return ONLY a JSON array of skill names that are "
                "needed to accomplish the query. Return the most relevant skills only. "
                "Output raw JSON, no markdown fences, no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nAvailable skills:\n{catalog}",
        },
    ]

    raw = llm_call(messages)

    # Parse the JSON array of skill names
    try:
        selected_names = json.loads(raw)
    except json.JSONDecodeError:
        # If the LLM wrapped it in markdown fences, strip them
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        selected_names = json.loads(cleaned)

    # Map names back to Skill objects
    skill_map = {s.name: s for s in skills}
    matched = [skill_map[name] for name in selected_names if name in skill_map]
    return matched


# ---------------------------------------------------------------------------
# SECTION 3: QUERY DECOMPOSITION
# ---------------------------------------------------------------------------
# When multiple skills are selected, the orchestrator asks the LLM to split
# the user's query into focused sub-tasks — one per skill. This prevents a
# subagent from trying to do another skill's job (e.g., the biopython agent
# shouldn't also try to compute embeddings).
# ---------------------------------------------------------------------------


def decompose_query(query: str, skills: list[Skill]) -> dict[str, str]:
    """
    Ask the LLM to decompose the query into one sub-task per skill.

    Returns a dict mapping skill name → focused task description.
    """
    skill_names = [s.name for s in skills]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a task decomposition agent. Given a user query and a list "
                "of skills, split the query into one focused sub-task per skill. "
                "Each sub-task should describe ONLY the work that skill should do — "
                "do not ask one skill to do another skill's job.\n"
                "Each sub-task should be a complete, actionable instruction that "
                "produces meaningful output on its own (not just data loading).\n\n"
                "Output a JSON object mapping skill name to task description string. "
                "Output raw JSON only, no markdown fences, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Skills to decompose across: {json.dumps(skill_names)}"
            ),
        },
    ]

    raw = llm_call(messages)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        result = json.loads(cleaned)

    return result


# ---------------------------------------------------------------------------
# SECTION 4: SUBAGENT — THE CODE-WRITING AGENT
# ---------------------------------------------------------------------------
# Each subagent is given:
#   - A task description (from the orchestrator)
#   - The SKILL.md documentation (so it knows the library's API)
#   - Context (file paths, etc.)
#
# It asks the LLM to write Python code, then executes that code and captures
# the output. This is the same pattern used by AI coding assistants.
#
# The LLM sees the skill documentation and writes code that uses the actual
# libraries (biopython, esm, etc.) — it's not faking the output.
# ---------------------------------------------------------------------------


def run_subagent(skill: Skill, task: str, context: dict) -> dict:
    """
    A subagent that:
      1. Reads the skill's SKILL.md documentation.
      2. Asks the LLM to write Python code to accomplish the task.
      3. Executes the code and captures stdout.
      4. Returns the result.

    This function runs inside a worker thread.
    """
    print(f"  [Thread] Subagent '{skill.name}' starting...", flush=True)
    t0 = time.time()

    # --- Step 1: Load skill documentation ---
    # The subagent reads the SKILL.md so the LLM knows what APIs are available.
    # In a more sophisticated system, you'd also load relevant reference files.
    skill_doc = (skill.skill_dir / "SKILL.md").read_text()

    # Truncate very long docs to fit context window
    if len(skill_doc) > 8000:
        skill_doc = skill_doc[:8000] + "\n\n[... truncated for brevity ...]"

    # --- Step 2: Ask the LLM to write code ---
    fasta_path = context["fasta_path"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding agent. You will be given a task and "
                "documentation for a Python library. Write a complete, self-contained "
                "Python script that accomplishes the task.\n\n"
                "Rules:\n"
                "- Output ONLY the Python code, no markdown fences, no explanation.\n"
                "- The code must print its results to stdout as readable text.\n"
                "- Include all necessary imports.\n"
                "- Handle errors gracefully with try/except.\n"
                "- Do NOT use any interactive features (no input(), no plots).\n"
                "- The code will be executed with exec(), so do NOT use "
                "'if __name__ == \"__main__\"'. Just call your functions directly.\n"
                "- Always import every module you use (e.g. import torch).\n"
                "- Available libraries: biopython, esm, torch, scikit-learn, numpy.\n"
                "- ONLY use the library that matches your assigned skill.\n"
                "- For ESM embeddings, use exactly this pattern:\n"
                "    import torch\n"
                "    from esm.models.esmc import ESMC\n"
                "    from esm.sdk.api import ESMProtein\n"
                "    model = ESMC.from_pretrained('esmc_300m')\n"
                "    protein = ESMProtein(sequence=seq)\n"
                "    tokens = model.encode(protein).sequence.unsqueeze(0)\n"
                "    with torch.no_grad():\n"
                "        result = model.forward(tokens)\n"
                "    embedding = result.embeddings  # shape [1, seq_len, 960]\n"
                "    # Mean-pool over seq_len to get a fixed-size vector per protein:\n"
                "    pooled = embedding.mean(dim=1)  # shape [1, 960]\n"
                "- When comparing embeddings of different-length sequences, always "
                "mean-pool first to get fixed-size vectors, then compute cosine similarity.\n"
                "- torch is CPU-only in this environment.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"The input FASTA file is at: {fasta_path}\n\n"
                f"Library documentation:\n{skill_doc}"
            ),
        },
    ]

    code = llm_call(messages, temperature=0.0)

    # Strip markdown fences if the LLM included them despite instructions
    code = re.sub(r"^```python\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)

    # --- Step 3: Execute the generated code ---
    # Capture stdout so we can return it as the result.
    # In production, this would run in a sandbox (Docker, E2B, etc.).
    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = captured

    error_msg = None
    try:
        exec(code, {"__builtins__": __builtins__})
    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    elapsed = time.time() - t0
    print(f"  [Thread] Subagent '{skill.name}' finished in {elapsed:.2f}s", flush=True)

    return {
        "skill": skill.name,
        "output": output,
        "code": code,
        "error": error_msg,
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# SECTION 5: PARALLEL EXECUTION VIA ThreadPoolExecutor
# ---------------------------------------------------------------------------
# Each matched skill gets its own thread. ThreadPoolExecutor manages the
# thread lifecycle, and as_completed() lets us process results as they
# arrive rather than waiting for the slowest one.
#
# WHY as_completed()?
#   If subagent A finishes in 2s and subagent B takes 30s, we can log A's
#   completion immediately. This matters when you have many subagents with
#   varying latencies.
# ---------------------------------------------------------------------------


def run_subagents_parallel(
    skills: list[Skill], sub_tasks: dict[str, str], context: dict
) -> list[dict]:
    """
    Execute one subagent per skill, all in parallel.
    Each skill gets its own focused task from the sub_tasks dict.
    Returns results in completion order.
    """
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=len(skills)) as executor:
        future_to_skill: dict[Future, Skill] = {}
        for skill in skills:
            task = sub_tasks.get(skill.name, "Complete the user's request.")
            future = executor.submit(run_subagent, skill, task, context)
            future_to_skill[future] = skill

        for future in as_completed(future_to_skill):
            skill = future_to_skill[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "skill": skill.name,
                    "output": "",
                    "code": "",
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_seconds": 0,
                }
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# SECTION 6: LLM-BASED SYNTHESIS
# ---------------------------------------------------------------------------
# After all subagents finish, the orchestrator sends their outputs to the
# LLM and asks it to write a coherent, comparative report. This is where
# the "agentic" value really shows — the LLM can reason across results
# from different tools and draw conclusions a simple script can't.
# ---------------------------------------------------------------------------


def synthesize_report(query: str, results: list[dict]) -> str:
    """
    Ask the LLM to synthesize all subagent outputs into a final report.
    """
    # Build a summary of each subagent's output for the LLM
    result_summaries = []
    for r in results:
        summary = f"=== Skill: {r['skill']} (took {r['elapsed_seconds']}s) ===\n"
        if r.get("error"):
            summary += f"ERROR:\n{r['error']}\n"
        summary += f"OUTPUT:\n{r['output']}\n"
        result_summaries.append(summary)

    combined = "\n".join(result_summaries)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a scientific report writer. You will receive outputs from "
                "multiple analysis tools that were run on protein sequences. "
                "Synthesize them into a clear, comparative report. Highlight the "
                "key findings and explain what the differences between sequence-level "
                "alignment and embedding-level similarity tell us about the proteins. "
                "Be concise and scientific."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query: {query}\n\n"
                f"Results from parallel subagents:\n\n{combined}"
            ),
        },
    ]

    return llm_call(messages, temperature=0.2)


# ---------------------------------------------------------------------------
# SECTION 7: ORCHESTRATOR — ties everything together
# ---------------------------------------------------------------------------


def orchestrate(query: str, fasta_path: str, skills_root: str = "skills") -> None:
    """
    Main orchestrator entry point.

    This is the function you'd call from a chat interface, CLI, or API.
    It coordinates the full pipeline: discover → match → execute → synthesize.
    """
    total_t0 = time.time()

    print("\n" + "=" * 70)
    print("ORCHESTRATOR — received query:")
    print(textwrap.fill(query, width=68, initial_indent="  ", subsequent_indent="  "))
    print("=" * 70 + "\n")

    # --- Phase 1: Discover skills ---
    skills = load_skill_registry(Path(skills_root))
    print_registry(skills)

    # --- Phase 2: LLM selects relevant skills ---
    print("ORCHESTRATOR — asking LLM which skills to use...")
    matched = match_skills_via_llm(query, skills)
    print("ORCHESTRATOR — LLM selected:")
    for s in matched:
        print(f"  ✅ {s.name}")
    print()

    if not matched:
        print("No skills matched. Try a different query.")
        return

    # --- Phase 3: Build focused task descriptions for each subagent ---
    # The orchestrator asks the LLM to decompose the query into skill-specific
    # sub-tasks. Each subagent gets a narrow, focused instruction so it doesn't
    # try to do work that belongs to another skill.
    print("ORCHESTRATOR — asking LLM to decompose query into sub-tasks...")
    sub_tasks = decompose_query(query, matched)
    for skill, task in sub_tasks.items():
        print(f"  [{skill}] {task}")
    print()

    # --- Phase 4: Launch subagents in parallel ---
    print("ORCHESTRATOR — launching subagents in parallel...")
    print("-" * 70)
    context = {"fasta_path": fasta_path}
    results = run_subagents_parallel(matched, sub_tasks, context)
    print("-" * 70)

    # --- Phase 5: Show what each subagent did ---
    print("\n" + "=" * 70)
    print("SUBAGENT RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\n--- {r['skill']} ({r['elapsed_seconds']}s) ---")
        if r.get("error"):
            print(f"ERROR:\n{r['error']}")
        print(f"GENERATED CODE:\n{r['code']}\n")
        print(f"OUTPUT:\n{r['output']}")

    # --- Phase 6: LLM synthesizes final report ---
    print("=" * 70)
    print("ORCHESTRATOR — asking LLM to synthesize final report...")
    print("=" * 70)
    report = synthesize_report(query, results)
    print(f"\n{report}\n")

    total_elapsed = time.time() - total_t0
    print("=" * 70)
    print(f"Total pipeline time: {total_elapsed:.1f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# SECTION 8: ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_query = (
        "Read a FASTA file containing two amino acid sequences and compare "
        "their alignments in amino acid space and in embedding space."
    )

    fasta_file = "sequences.fasta"

    orchestrate(user_query, fasta_file)
