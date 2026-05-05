"""Tool registry with decorator-based registration.

Design:
    Tools are plain Python functions. The @tool decorator registers metadata
    (description, tags, caching) without changing the function's behavior.
    Agents query the registry by tags to get the subset of tools they need.

    The registry also produces OpenAI-compatible tool schemas so the LLM can
    call tools via the standard function-calling protocol.

Usage:
    from targetsearch.core.registry import registry

    @registry.tool(
        description="Search PubMed for papers matching a query.",
        tags=["literature", "pubmed"],
        cache=True,
        params={
            "query": "Search terms (e.g. 'EGFR lung cancer')",
            "max_results": "Maximum number of papers to return (default 10)",
        },
        returns="List of dicts with keys: pmid, title, abstract, authors, year",
    )
    def pubmed_search(query: str, max_results: int = 10) -> list[dict]:
        ...

    # Agent gets only the tools it needs:
    lit_tools = registry.get_tools(tags=["literature"])

    # Generate OpenAI tool schemas for the LLM:
    schemas = registry.tool_schemas(tags=["literature"])
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

log = logging.getLogger(__name__)

def _is_action_context(tp: type) -> bool:
    """Check if a type annotation is ActionContext (without importing it at module level).

    Uses string comparison to avoid circular imports — context.py imports
    from schemas, and tools import from registry.
    """
    # Direct class check by name (handles the common case)
    name = getattr(tp, "__name__", "") or getattr(tp, "__qualname__", "")
    if name == "ActionContext":
        return True
    # Also check string annotations that haven't been resolved
    if isinstance(tp, str) and tp == "ActionContext":
        return True
    return False


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolSpec:
    """Metadata for a registered tool."""

    name: str
    func: Callable
    description: str
    tags: list[str] = field(default_factory=list)
    cache: bool = False
    params: dict[str, str] = field(default_factory=dict)  # name → description
    returns: str = ""

    def to_openai_schema(self) -> dict:
        """Produce an OpenAI function-calling tool schema.

        Inspects the function signature to build the JSON Schema for
        parameters. The `params` dict provides human-readable descriptions
        for each parameter.

        Parameters typed as ActionContext are excluded — the agent loop
        injects them automatically, so the LLM should never see them.
        """
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            ptype = hints.get(pname, str)

            # Skip ActionContext parameters — auto-injected, not LLM-visible
            if _is_action_context(ptype):
                continue

            # Unwrap Optional / Union with None
            origin = getattr(ptype, "__origin__", None)
            item_type = None
            if origin is not None:
                args = getattr(ptype, "__args__", ())
                # list[X] → array with items
                if origin is list:
                    item_type = args[0] if args else str
                    ptype = list
                # X | None → X
                elif type(None) in args:
                    ptype = next(a for a in args if a is not type(None))
                    # Check if the unwrapped type is also generic (e.g. list[str] | None)
                    inner_origin = getattr(ptype, "__origin__", None)
                    if inner_origin is list:
                        inner_args = getattr(ptype, "__args__", ())
                        item_type = inner_args[0] if inner_args else str
                        ptype = list

            json_type = _TYPE_MAP.get(ptype, "string")
            prop: dict[str, Any] = {"type": json_type}
            if json_type == "array" and item_type is not None:
                prop["items"] = {"type": _TYPE_MAP.get(item_type, "string")}
            elif json_type == "array":
                prop["items"] = {"type": "string"}
            if pname in self.params:
                prop["description"] = self.params[pname]
            properties[pname] = prop

            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Central registry of deterministic tools.

    Tools register themselves at import time via the @tool decorator.
    Agents receive a filtered view via .get_tools(tags=...).
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._cache: dict[str, Any] = {}

    def tool(
        self,
        description: str,
        tags: list[str] | None = None,
        cache: bool = False,
        params: dict[str, str] | None = None,
        returns: str = "",
    ) -> Callable:
        """Decorator to register a function as a tool.

        Args:
            description: What the tool does (shown to the LLM).
            tags: Categories for filtering (e.g. ["literature", "pubmed"]).
            cache: If True, memoize results by argument hash.
            params: Mapping of parameter name → description for the LLM.
            returns: Human-readable description of the return value.
        """

        def decorator(func: Callable) -> Callable:
            wrapped = func

            if cache:
                @functools.wraps(func)
                def cached_wrapper(*args: Any, **kwargs: Any) -> Any:
                    key = hashlib.sha256(
                        json.dumps(
                            {"fn": func.__name__, "args": args, "kwargs": kwargs},
                            sort_keys=True,
                            default=str,
                        ).encode()
                    ).hexdigest()
                    if key not in self._cache:
                        self._cache[key] = func(*args, **kwargs)
                    return self._cache[key]

                wrapped = cached_wrapper

            spec = ToolSpec(
                name=func.__name__,
                func=wrapped,
                description=description,
                tags=tags or [],
                cache=cache,
                params=params or {},
                returns=returns,
            )
            self._tools[func.__name__] = spec
            return wrapped

        return decorator

    # --- Querying ---

    def get_tools(self, tags: list[str] | None = None) -> list[ToolSpec]:
        """Return tools matching ANY of the given tags, or all if tags is None."""
        if tags is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if set(tags) & set(t.tags)]

    def get_tool(self, name: str) -> ToolSpec:
        """Look up a single tool by name. Raises KeyError if not found."""
        return self._tools[name]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        """Call a registered tool by name with the given arguments.

        This is the method agents use to dispatch tool calls from the LLM.
        If `context` is provided and the tool's signature includes an
        ActionContext parameter, it is injected automatically.
        """
        spec = self.get_tool(name)
        log.info("Calling tool %s(%s)", name, arguments)

        # Auto-inject ActionContext if the tool expects it
        if context is not None:
            hints = get_type_hints(spec.func)
            for pname, ptype in hints.items():
                if pname == "return":
                    continue
                if _is_action_context(ptype):
                    arguments = {**arguments, pname: context}
                    break

        return spec.func(**arguments)

    # --- Schema generation for LLM function calling ---

    def tool_schemas(self, tags: list[str] | None = None) -> list[dict]:
        """Produce OpenAI-compatible tool schemas for the given tag filter."""
        return [t.to_openai_schema() for t in self.get_tools(tags)]

    def describe_tools(self, tags: list[str] | None = None) -> str:
        """Produce a human-readable description of available tools.

        Useful for injecting into system prompts when the model doesn't
        support native function calling.
        """
        tools = self.get_tools(tags)
        lines = []
        for t in tools:
            params_str = ", ".join(f"{k}: {v}" for k, v in t.params.items())
            lines.append(
                f"- {t.name}({params_str}) → {t.returns}\n  {t.description}"
            )
        return "\n".join(lines)

    def tool_needs_context(self, name: str) -> bool:
        """Check if a tool's signature includes an ActionContext parameter."""
        spec = self.get_tool(name)
        hints = get_type_hints(spec.func)
        return any(
            _is_action_context(ptype)
            for pname, ptype in hints.items()
            if pname != "return"
        )

    # --- Introspection ---

    def list_names(self, tags: list[str] | None = None) -> list[str]:
        return [t.name for t in self.get_tools(tags)]

    def clear_cache(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools)"


# Singleton — all tool modules register against this at import time
registry = ToolRegistry()
