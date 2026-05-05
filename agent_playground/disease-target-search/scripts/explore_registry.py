"""Explore the tool registry — list tools, tags, and OpenAI schemas."""

import json

from targetsearch.core.registry import registry

# Import all tool modules to register them
import targetsearch.tools.literature  # noqa: F401
import targetsearch.tools.targets  # noqa: F401
import targetsearch.tools.ontology  # noqa: F401
import targetsearch.tools.fulltext  # noqa: F401
import targetsearch.tools.prompt_tools  # noqa: F401
import targetsearch.tools.paper_tools  # noqa: F401
import targetsearch.tools.synthesis_tools  # noqa: F401
import targetsearch.tools.coordination_tools  # noqa: F401
import targetsearch.tools.agent_tools  # noqa: F401

# List all tools and their tags
print("=== All registered tools ===\n")
for t in registry.get_tools():
    ctx = " [ctx]" if registry.tool_needs_context(t.name) else ""
    print(f"  {t.name:35s} tags={t.tags}  cache={t.cache}{ctx}")

# See what each agent sees
print("\n=== Orchestrator tools (tags: orchestration, synthesis) ===\n")
for t in registry.get_tools(tags=["orchestration", "synthesis"]):
    print(f"  {t.name}")

print("\n=== Searcher tools (tags: literature, targets, disease, ontology, coordination) ===\n")
for t in registry.get_tools(tags=["literature", "targets", "disease", "ontology", "coordination"]):
    print(f"  {t.name}")

print("\n=== Feedback tools (tags: prompts) ===\n")
for t in registry.get_tools(tags=["prompts"]):
    print(f"  {t.name}")

# Print the OpenAI schema the LLM receives
schemas = registry.tool_schemas(tags=["literature"])
print("\n=== OpenAI schema for first literature tool ===\n")
print(json.dumps(schemas[0], indent=2))
