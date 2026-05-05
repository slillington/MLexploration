# Simple Agent (Coursera Project)

A minimal AI agent framework demonstrating the **GAME loop** pattern (Goals, Actions, Memory, Environment) using LLM function-calling.

## Overview

This project implements a simple autonomous agent that iteratively reasons about goals, selects tool-based actions, executes them in an environment, and updates its memory — looping until the task is complete.

## Structure

```
coursera_project/
├── simple_agent.py   # Agent framework and example usage
└── subdir/
    └── foo.py        # Dummy script used as test input for the agent
```

## Core Components (`simple_agent.py`)

| Component | Description |
|-----------|-------------|
| `Agent` | Orchestrates the GAME loop: constructs prompts, invokes the LLM, executes actions, and updates memory. |
| `Goal` | A named, prioritized objective that shapes the agent's system prompt. |
| `Action` / `ActionRegistry` | Encapsulates a callable tool with metadata; the registry manages available actions. |
| `Memory` | Stores the conversation history (user, assistant, and environment messages) fed back into each prompt. |
| `Environment` | Executes actions and returns timestamped results (or error traces). |
| `AgentLanguage` / `AgentFunctionCallingActionLanguage` | Translates goals, memory, and actions into an LLM-compatible prompt with OpenAI-style tool definitions, and parses the response. |
| `ToolRegistry` / `@tool` decorator | Provides a decorator-based DSL for defining and tagging tools, then filtering them into an `ActionRegistry`. |

## Registered Tools

- **`list_project_files`** — Lists `.py` files in the current directory.
- **`list_project_files_recursive`** — Lists `.py` files recursively.
- **`read_project_file`** — Reads and returns the contents of a given file.
- **`terminate`** — Ends the agent loop and surfaces a final message.

## How It Works

1. Goals are defined (e.g., "read each file", "produce a README").
2. The agent constructs a prompt containing goals + memory + tool schemas.
3. The LLM responds with a function call (tool name + arguments).
4. The environment executes the chosen tool and returns the result.
5. Memory is updated with the decision and outcome.
6. Steps 2–5 repeat until the `terminate` tool is called or the iteration limit is reached.

## Usage

```bash
python simple_agent.py
```

Requires a `GITHUB_TOKEN` or appropriate LiteLLM credentials for the `github_copilot/gpt-5-mini` model.

## Dependencies

- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM API client
- Python 3.10+ (dataclasses, typing)
