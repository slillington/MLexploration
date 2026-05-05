"""Centralized configuration for the targetsearch package.

All tunables live here so you never hunt for magic strings in agent code.
Override via environment variables or by mutating the Config instance.

Place a .env file in the project root to set secrets:
    NCBI_API_KEY=your_key_here
    NCBI_EMAIL=your_email@example.com
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before reading any env vars
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    # --- LLM settings ---
    model: str = "github_copilot/gpt-5.4"
    summarization_model: str = "github_copilot/gpt-5-mini"
    max_tokens: int = 8192

    # --- Paths ---
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def skills_dir(self) -> Path:
        return self.project_root / "skills"

    @property
    def prompts_dir(self) -> Path:
        return self.project_root / "targetsearch" / "prompts"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "logs"

    # --- API settings ---
    ncbi_email: str = field(
        default_factory=lambda: os.environ.get("NCBI_EMAIL", "targetsearch@example.com")
    )
    ncbi_api_key: str | None = field(
        default_factory=lambda: os.environ.get("NCBI_API_KEY")
    )
    s2_api_key: str | None = field(
        default_factory=lambda: os.environ.get("S2_API_KEY")
    )
    request_timeout: float = 30.0
    max_retries: int = 3

    # --- Agent settings ---
    max_tool_calls_per_turn: int = 30  # safety limit per agent run
    parallel_workers: int = 4  # concurrent LLM calls and HTTP fetches
    max_papers_initial: int = 12  # paper budget for initial search pass
    max_papers_gap_fill: int = 8  # paper budget reserved for gap-fill passes

    @property
    def max_papers(self) -> int:
        """Total paper budget across all passes."""
        return self.max_papers_initial + self.max_papers_gap_fill
    synthesis_batch_size: int = 10  # papers per batch in map-reduce synthesis
    max_feedback_rounds: int = 1  # max search→synthesize→feedback cycles

    # Multi-pass synthesis
    synthesis_max_internal_passes: int = 1  # max refinement iterations
    synthesis_quality_threshold: float = 6.0  # min avg section score to pass (0-10)
    synthesis_refinement_enabled: bool = True  # enable Pass D refinement loop
    max_new_hard_failures_for_pass: int = 2  # tolerate this many persistent failures
    history_compaction_threshold: int = 80_000  # compact tool results above this char count (0=disable)


# Singleton — import this everywhere
config = Config()
