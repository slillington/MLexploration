"""Logging setup for targetsearch — file + console output.

Each pipeline run gets its own log file named by run ID and timestamp.
Console output stays at INFO; the file captures DEBUG, which includes
full LLM prompts, responses, and token usage for prompt iteration.

Usage:
    from targetsearch.core.logging import setup_logging

    run_id = setup_logging()          # auto-generated run ID
    run_id = setup_logging("my-run")  # explicit run ID
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from targetsearch.core.config import config

# Dedicated logger for LLM I/O — keeps LLM traffic separate from
# operational logs so you can grep one file for prompt/response pairs.
llm_logger = logging.getLogger("targetsearch.llm_io")


def setup_logging(run_id: str | None = None) -> str:
    """Configure logging with a per-run log file.

    Creates two files per run inside config.log_dir:
      - {run_id}.log       — full operational log (DEBUG)
      - {run_id}.llm.log   — LLM inputs/outputs only (DEBUG)

    Console output is left at INFO (or whatever the root logger has).

    Args:
        run_id: Identifier for this run.  Generated if not provided.

    Returns:
        The run_id used (useful when auto-generated).
    """
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        run_id = f"{ts}_{short_id}"

    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Main log file: everything at DEBUG ---
    main_path = log_dir / f"{run_id}.log"
    main_handler = logging.FileHandler(main_path, encoding="utf-8")
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(fmt)

    root = logging.getLogger("targetsearch")
    root.setLevel(logging.DEBUG)
    root.addHandler(main_handler)

    # --- LLM I/O log file: prompts and responses only ---
    llm_path = log_dir / f"{run_id}.llm.log"
    llm_handler = logging.FileHandler(llm_path, encoding="utf-8")
    llm_handler.setLevel(logging.DEBUG)
    llm_handler.setFormatter(fmt)

    llm_logger.setLevel(logging.DEBUG)
    llm_logger.addHandler(llm_handler)
    # Also write LLM I/O to the main log for a single-file view
    llm_logger.addHandler(main_handler)
    # Don't propagate to root to avoid duplicate console output
    llm_logger.propagate = False

    logging.getLogger("targetsearch").info(
        "Logging initialised — run_id=%s, log_dir=%s", run_id, log_dir
    )

    return run_id
