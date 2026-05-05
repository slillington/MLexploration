# Drug Target Discovery Agent System — Design & Development Plan

## 1. Problem Statement

Given a prompt like *"identify new drug targets for disease X"*, the system should:

1. Research the disease biology (pathways, genetics, existing treatments)
2. Generate hypotheses for novel targets grounded in primary literature and first-principles reasoning
3. Evaluate each target–therapeutic strategy pair on feasibility axes (druggability, genetic evidence, competitive landscape, safety)
4. Produce a ranked report with citations and reasoning chains

---

## 2. High-Level Architecture

```
                          ┌──────────────────────┐
                          │     User Prompt       │
                          │ "find targets for X"  │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │    Orchestrator       │
                          │                       │
                          │  • Parse disease      │
                          │  • Plan workflow       │
                          │  • Dispatch agents    │
                          │  • Collect & rank     │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │  Disease Intel   │   │  Target Hypo-   │   │  Evaluation     │
   │  Agent           │   │  thesis Agent   │   │  Agent          │
   │                  │   │                 │   │                 │
   │ • Literature     │   │ • Generate      │   │ • Druggability  │
   │   search         │   │   target–       │   │ • Genetic       │
   │ • Pathway        │   │   strategy      │   │   evidence      │
   │   mapping        │   │   pairs         │   │ • Competitive   │
   │ • Genetic        │   │ • First-         │   │   landscape     │
   │   evidence       │   │   principles    │   │ • Safety        │
   │ • Treatment      │   │   reasoning     │   │   liabilities   │
   │   landscape      │   │ • Literature-   │   │ • Feasibility   │
   │                  │   │   backed        │   │   scoring       │
   └────────┬─────────┘   └────────┬────────┘   └────────┬────────┘
            │                      │                      │
            │         ┌────────────┘                      │
            │         │                                   │
            ▼         ▼                                   ▼
   ┌──────────────────────┐                    ┌─────────────────┐
   │   Tool Registry      │                    │  Skills Registry │
   │   (deterministic)    │                    │  (prompt-based)  │
   │                      │                    │                  │
   │ @tool(tags=[...])    │                    │  SKILL.md files  │
   │ @tool(cache=True)    │                    │  loaded at       │
   │                      │                    │  runtime         │
   │ • pubmed_search      │                    │                  │
   │ • opentargets_query  │                    │ • lit_review      │
   │ • uniprot_lookup     │                    │ • hypothesis_gen  │
   │ • chembl_query       │                    │ • mechanism_      │
   │ • string_db_query    │                    │   reasoning       │
   │ • gene_ontology      │                    │ • safety_review   │
   │ • clinicaltrials_    │                    │                  │
   │   search             │                    │                  │
   └──────────────────────┘                    └─────────────────┘
```

---

## 3. Package Structure

```
agent_playground/
├── pyproject.toml                  # Project metadata, dependencies
├── DESIGN.md                       # This file
│
├── targetsearch/                   # Main package
│   ├── __init__.py
│   │
│   ├── core/                       # Framework primitives
│   │   ├── __init__.py
│   │   ├── registry.py             # ToolRegistry + @tool decorator
│   │   ├── skills.py               # SkillsRegistry (loads SKILL.md files)
│   │   ├── llm.py                  # LLM client wrapper (provider-agnostic)
│   │   └── config.py               # Global config (model, API keys, timeouts)
│   │
│   ├── tools/                      # Deterministic tools (API wrappers)
│   │   ├── __init__.py
│   │   ├── literature.py           # PubMed, PMC, bioRxiv, Semantic Scholar
│   │   ├── targets.py              # Open Targets, UniProt, STRING-DB
│   │   ├── chemistry.py            # ChEMBL, DrugBank lookups
│   │   ├── clinical.py             # ClinicalTrials.gov
│   │   ├── ontology.py             # Gene Ontology, Disease Ontology, HPO
│   │   └── genetics.py             # GWAS Catalog, gnomAD, ClinVar
│   │
│   ├── agents/                     # Agent definitions
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Top-level planner and dispatcher
│   │   ├── disease_intel.py        # Disease research agent
│   │   ├── hypothesis.py           # Target hypothesis generation agent
│   │   └── evaluation.py           # Target evaluation and scoring agent
│   │
│   ├── prompts/                    # Prompt templates (Jinja2 or plain text)
│   │   ├── orchestrator.txt
│   │   ├── disease_intel.txt
│   │   ├── hypothesis.txt
│   │   └── evaluation.txt
│   │
│   └── schemas/                    # Data models (Pydantic)
│       ├── __init__.py
│       ├── disease.py              # DiseaseProfile, Pathway, GeneticAssociation
│       ├── target.py               # Target, TherapeuticStrategy, TargetHypothesis
│       └── evaluation.py           # FeasibilityScore, EvaluationReport
│
├── skills/                         # Prompt-based skills (existing pattern)
│   ├── biopython/SKILL.md
│   ├── esm/SKILL.md
│   ├── paper-lookup/SKILL.md
│   ├── scikit-learn/SKILL.md
│   ├── lit-review/SKILL.md         # NEW — systematic literature review
│   ├── hypothesis-gen/SKILL.md     # NEW — structured hypothesis generation
│   └── mechanism-reasoning/SKILL.md # NEW — pathway/mechanism reasoning
│
└── tests/
    ├── __init__.py
    ├── test_registry.py
    ├── test_tools_literature.py
    └── test_schemas.py
```

---

## 4. Key Design Decisions

### 4a. Tool Registry with Decorators

Tools are plain Python functions decorated with `@tool`. The decorator registers metadata; agents query the registry by tags to get the subset they need.

```python
# targetsearch/core/registry.py

from dataclasses import dataclass, field
from typing import Callable, Any
import functools
import hashlib
import json

@dataclass
class ToolSpec:
    """Metadata for a registered tool."""
    name: str
    func: Callable
    description: str
    tags: list[str]
    cache: bool
    params: dict[str, str]       # param_name → description
    returns: str                 # return type description

class ToolRegistry:
    """Central registry of deterministic tools.

    Agents receive a filtered view via .get_tools(tags=...).
    """
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._cache: dict[str, Any] = {}

    def tool(
        self,
        description: str,
        tags: list[str] | None = None,
        cache: bool = False,
        params: dict[str, str] | None = None,
        returns: str = "",
    ):
        """Decorator to register a function as a tool.

        Usage:
            registry = ToolRegistry()

            @registry.tool(
                description="Search PubMed for papers matching a query.",
                tags=["literature", "pubmed"],
                cache=True,
                params={"query": "Search terms", "max_results": "Max papers to return"},
                returns="List of PubMed article dicts",
            )
            def pubmed_search(query: str, max_results: int = 10) -> list[dict]:
                ...
        """
        def decorator(func: Callable) -> Callable:
            spec = ToolSpec(
                name=func.__name__,
                func=func,
                description=description,
                tags=tags or [],
                cache=cache,
                params=params or {},
                returns=returns,
            )
            self._tools[func.__name__] = spec

            if cache:
                @functools.wraps(func)
                def cached_wrapper(*args, **kwargs):
                    key = hashlib.sha256(
                        json.dumps({"fn": func.__name__, "args": args, "kwargs": kwargs},
                                   sort_keys=True, default=str).encode()
                    ).hexdigest()
                    if key not in self._cache:
                        self._cache[key] = func(*args, **kwargs)
                    return self._cache[key]
                spec.func = cached_wrapper
                return cached_wrapper
            return func
        return decorator

    def get_tools(self, tags: list[str] | None = None) -> list[ToolSpec]:
        """Return tools matching ANY of the given tags, or all tools if tags is None."""
        if tags is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if set(tags) & set(t.tags)]

    def get_tool(self, name: str) -> ToolSpec:
        return self._tools[name]

    def describe_tools(self, tags: list[str] | None = None) -> str:
        """Produce an LLM-readable description of available tools."""
        tools = self.get_tools(tags)
        lines = []
        for t in tools:
            params_str = ", ".join(f"{k}: {v}" for k, v in t.params.items())
            lines.append(f"- {t.name}({params_str}) → {t.returns}\n  {t.description}")
        return "\n".join(lines)

# Singleton instance — tools register against this at import time
registry = ToolRegistry()
```

### 4b. Skills Registry

Extends the existing `skills/*/SKILL.md` pattern from `agentic_demo.py`. Skills are prompt-based "tools" — they contain instructions and context that an LLM uses to perform a task, rather than deterministic code.

```python
# targetsearch/core/skills.py

from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class Skill:
    name: str
    description: str
    content: str           # Full SKILL.md text
    skill_dir: Path

class SkillsRegistry:
    """Discovers and serves prompt-based skills from disk."""

    def __init__(self, skills_root: Path):
        self._skills: dict[str, Skill] = {}
        self._load(skills_root)

    def _load(self, root: Path):
        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                text = skill_md.read_text()
                match = re.search(
                    r"^---\s*\n.*?^description:\s*(.+?)(?:\n[a-z]|\n---)",
                    text, re.MULTILINE | re.DOTALL,
                )
                desc = match.group(1).strip() if match else "(no description)"
                self._skills[child.name] = Skill(
                    name=child.name, description=desc,
                    content=text, skill_dir=child,
                )

    def get_skill(self, name: str) -> Skill:
        return self._skills[name]

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def describe_skills(self) -> str:
        return "\n".join(
            f"- {s.name}: {s.description[:120]}" for s in self._skills.values()
        )
```

### 4c. Agent Base Pattern

Each agent is a class that holds its system prompt, its tool/skill subset, and a `run()` method. Agents don't inherit from a heavy framework — they're thin wrappers around an LLM call loop with tool access.

```python
# Sketch — not the full implementation

class Agent:
    def __init__(self, name, system_prompt, tools=None, skills=None, llm=None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []       # list of ToolSpec
        self.skills = skills or []     # list of Skill
        self.llm = llm or default_llm
        self.messages = [{"role": "system", "content": system_prompt}]

    def run(self, user_message: str) -> str:
        """Execute the agent's reasoning loop.

        The agent can call tools by emitting structured tool-call requests.
        The loop continues until the agent produces a final answer.
        """
        ...
```

---

## 5. Workflow: What Happens When You Run a Query

```
User: "Identify new drug targets for idiopathic pulmonary fibrosis"

Step 1 — ORCHESTRATOR parses the disease and plans the workflow
  → Identifies: disease = IPF, ICD codes, synonyms
  → Plans: disease_intel → hypothesis → evaluation

Step 2 — DISEASE INTEL AGENT runs
  Tools used: pubmed_search, opentargets_query, gene_ontology
  Skills used: lit-review
  Output: DiseaseProfile
    - Known pathways (TGF-β, Wnt, Hedgehog, integrin signaling, ...)
    - Genetic associations (MUC5B, TERT, DSP, ...)
    - Current treatments (nintedanib, pirfenidone — both anti-fibrotic)
    - Unmet needs (disease still progresses, no reversal)

Step 3 — HYPOTHESIS AGENT runs
  Input: DiseaseProfile from step 2
  Tools used: string_db_query, uniprot_lookup
  Skills used: hypothesis-gen, mechanism-reasoning
  Output: list[TargetHypothesis]
    Each hypothesis contains:
    - Target (protein/gene)
    - Therapeutic strategy (inhibitor, activator, degrader, antibody, ...)
    - Mechanistic rationale (first-principles reasoning chain)
    - Supporting literature (PMIDs, key findings)
    - Novelty assessment (is this target already pursued?)

Step 4 — EVALUATION AGENT runs
  Input: list[TargetHypothesis] from step 3
  Tools used: chembl_query, clinicaltrials_search, genetics tools
  Skills used: safety-review
  Output: list[EvaluationReport]
    Each report scores:
    - Genetic evidence strength (GWAS, Mendelian, eQTL)
    - Druggability (protein class, binding sites, existing chemical matter)
    - Competitive landscape (who else is working on this?)
    - Safety liabilities (essential gene? broad expression? known toxicity?)
    - Overall feasibility score

Step 5 — ORCHESTRATOR synthesizes
  → Ranks targets by composite score
  → Produces final report with citations
```

---

## 6. Data Models (Pydantic Schemas)

```python
# targetsearch/schemas/target.py

from pydantic import BaseModel

class Target(BaseModel):
    gene_symbol: str
    uniprot_id: str | None = None
    protein_name: str
    protein_class: str | None = None   # kinase, GPCR, ion channel, etc.

class TherapeuticStrategy(BaseModel):
    modality: str                       # small molecule, antibody, degrader, ASO, etc.
    mechanism: str                      # inhibitor, activator, degrader, etc.
    rationale: str                      # Why this modality for this target

class Citation(BaseModel):
    pmid: str | None = None
    doi: str | None = None
    title: str
    authors: list[str] = []
    year: int | None = None
    key_finding: str                    # One-sentence summary of relevance

class TargetHypothesis(BaseModel):
    target: Target
    strategy: TherapeuticStrategy
    mechanistic_rationale: str          # First-principles reasoning chain
    supporting_evidence: list[Citation]
    novelty_assessment: str             # Known target? Novel angle?
    confidence: str                     # high / medium / low

class FeasibilityScore(BaseModel):
    genetic_evidence: float             # 0-1
    druggability: float                 # 0-1
    competitive_landscape: float        # 0-1 (higher = less competition = better)
    safety: float                       # 0-1 (higher = safer)
    overall: float                      # Weighted composite

class EvaluationReport(BaseModel):
    hypothesis: TargetHypothesis
    scores: FeasibilityScore
    detailed_assessment: str
    key_risks: list[str]
    next_steps: list[str]              # Suggested validation experiments
```

---

## 7. Development Plan — Phased Build

### Phase 1: Foundation (start here)
**Goal:** Working package skeleton with registry, config, LLM wrapper, and one real tool.

| Task | What you build | What you learn |
|------|---------------|----------------|
| 1.1 | `pyproject.toml`, package structure, `__init__.py` files | Python packaging |
| 1.2 | `core/config.py` — model name, API settings | Centralized config |
| 1.3 | `core/llm.py` — thin wrapper around litellm | Provider abstraction |
| 1.4 | `core/registry.py` — `@tool` decorator + `ToolRegistry` | Decorator patterns, registries |
| 1.5 | `core/skills.py` — `SkillsRegistry` (port from agentic_demo.py) | Skill loading |
| 1.6 | `tools/literature.py` — `pubmed_search` as first real tool | HTTP APIs, tool implementation |
| 1.7 | `schemas/target.py` + `schemas/disease.py` — Pydantic models | Structured data |
| 1.8 | `tests/` — unit tests for registry, schemas | Testing patterns |
| 1.9 | Smoke test: call `pubmed_search` through the registry | End-to-end validation |

**Milestone:** You can do `from targetsearch.core.registry import registry` and call a registered PubMed search tool.

### Phase 2: First Agent
**Goal:** A single agent (Disease Intel) that uses tools and skills to produce a DiseaseProfile.

| Task | What you build |
|------|---------------|
| 2.1 | `agents/base.py` — Agent base class with tool-calling loop |
| 2.2 | `prompts/disease_intel.txt` — system prompt template |
| 2.3 | `agents/disease_intel.py` — DiseaseIntelAgent |
| 2.4 | `tools/targets.py` — Open Targets API wrapper |
| 2.5 | `tools/ontology.py` — Gene Ontology / Disease Ontology lookups |
| 2.6 | Wire it up: agent selects tools, calls them, produces DiseaseProfile |

**Milestone:** `DiseaseIntelAgent.run("idiopathic pulmonary fibrosis")` returns a structured `DiseaseProfile` with real data.

### Phase 3: Hypothesis Generation
**Goal:** Second agent that takes a DiseaseProfile and generates target hypotheses.

| Task | What you build |
|------|---------------|
| 3.1 | `agents/hypothesis.py` — HypothesisAgent |
| 3.2 | `prompts/hypothesis.txt` — prompt emphasizing first-principles reasoning |
| 3.3 | `tools/targets.py` additions — STRING-DB, UniProt |
| 3.4 | `skills/hypothesis-gen/SKILL.md` — structured hypothesis generation skill |
| 3.5 | `skills/mechanism-reasoning/SKILL.md` — pathway reasoning skill |

**Milestone:** Given a DiseaseProfile, produces a list of `TargetHypothesis` objects with citations and reasoning chains.

### Phase 4: Evaluation & Scoring
**Goal:** Third agent that evaluates and ranks hypotheses.

| Task | What you build |
|------|---------------|
| 4.1 | `agents/evaluation.py` — EvaluationAgent |
| 4.2 | `tools/chemistry.py` — ChEMBL queries |
| 4.3 | `tools/clinical.py` — ClinicalTrials.gov |
| 4.4 | `tools/genetics.py` — GWAS Catalog, ClinVar |
| 4.5 | Scoring logic — weighted composite feasibility score |

**Milestone:** Produces ranked `EvaluationReport` list with scores and risk assessments.

### Phase 5: Orchestrator & End-to-End
**Goal:** Wire all agents together with the orchestrator.

| Task | What you build |
|------|---------------|
| 5.1 | `agents/orchestrator.py` — plans workflow, dispatches agents, synthesizes |
| 5.2 | Parallel execution (ThreadPoolExecutor, as in agentic_demo.py) |
| 5.3 | Final report generation with citations |
| 5.4 | CLI entry point: `python -m targetsearch "find targets for IPF"` |

### Phase 6: Refinements (ongoing)
- Caching layer for API calls (avoid re-fetching during development)
- Retry/fallback logic for flaky APIs
- Structured logging
- More sophisticated scoring (ML-based druggability prediction, etc.)
- Interactive mode (user can steer hypothesis generation)

---

## 8. Design Rationale

**Why two registries (tools vs. skills)?**
Tools are deterministic functions with defined inputs/outputs — they call APIs, parse data, compute scores. Skills are prompt-based instructions that guide LLM reasoning — they're for tasks where the "logic" is in natural language (e.g., "evaluate this target's safety profile by considering..."). Keeping them separate makes it clear what's deterministic and what's LLM-driven.

**Why decorator-based registration?**
It keeps tool definition and registration co-located. You read `tools/literature.py` and immediately see what's available, what tags it has, and whether it's cached. No separate config file to keep in sync.

**Why tag-based tool filtering?**
Different agents need different tools. The Disease Intel agent needs `["literature", "ontology"]` tools; the Evaluation agent needs `["chemistry", "clinical", "genetics"]`. Tags let you define these subsets declaratively without hardcoding tool names in agent code.

**Why Pydantic schemas?**
Agents pass structured data between them. Pydantic validates the data at boundaries, provides serialization for free, and makes the data contracts explicit. When the Hypothesis agent produces a `TargetHypothesis`, the Evaluation agent knows exactly what fields to expect.

**Why not LangChain/CrewAI/AutoGen?**
You already have a working orchestrator pattern in `agentic_demo.py`. Building on that foundation teaches you more and gives you full control. The abstractions here (registry, agent base class, schemas) are lightweight and transparent — you can see exactly what's happening at every step. You can always adopt a framework later if needed.

---

## 9. Getting Started

After reviewing this design, the first implementation step is Phase 1: scaffold the package, build the registries, and get one tool working end-to-end. This gives you a working foundation to iterate on without getting bogged down in the full agent pipeline.
