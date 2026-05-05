"""Tests for the tool registry and @tool decorator."""

from targetsearch.core.context import ActionContext
from targetsearch.core.registry import ToolRegistry


def make_registry() -> ToolRegistry:
    """Create a fresh registry with a few test tools."""
    reg = ToolRegistry()

    @reg.tool(
        description="Add two numbers.",
        tags=["math"],
        params={"a": "First number", "b": "Second number"},
        returns="Sum of a and b",
    )
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool(
        description="Multiply two numbers.",
        tags=["math"],
        params={"a": "First number", "b": "Second number"},
        returns="Product of a and b",
    )
    def multiply(a: int, b: int) -> int:
        return a * b

    @reg.tool(
        description="Greet someone.",
        tags=["text"],
        params={"name": "Person's name"},
        returns="Greeting string",
    )
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    @reg.tool(
        description="Cached doubler.",
        tags=["math"],
        cache=True,
        params={"x": "Number to double"},
        returns="x * 2",
    )
    def double(x: int) -> int:
        # We'll track calls via a side effect to verify caching
        double._call_count = getattr(double, "_call_count", 0) + 1
        return x * 2

    return reg


class TestToolRegistration:
    def test_tools_are_registered(self):
        reg = make_registry()
        assert len(reg) == 4
        assert set(reg.list_names()) == {"add", "multiply", "greet", "double"}

    def test_get_tool_by_name(self):
        reg = make_registry()
        spec = reg.get_tool("add")
        assert spec.name == "add"
        assert spec.description == "Add two numbers."
        assert spec.tags == ["math"]

    def test_get_tool_missing_raises(self):
        reg = make_registry()
        try:
            reg.get_tool("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestTagFiltering:
    def test_filter_by_tag(self):
        reg = make_registry()
        math_tools = reg.get_tools(tags=["math"])
        names = {t.name for t in math_tools}
        assert names == {"add", "multiply", "double"}

    def test_filter_by_multiple_tags(self):
        reg = make_registry()
        # Should return tools matching ANY tag
        tools = reg.get_tools(tags=["math", "text"])
        assert len(tools) == 4

    def test_filter_no_match(self):
        reg = make_registry()
        tools = reg.get_tools(tags=["nonexistent"])
        assert tools == []

    def test_no_filter_returns_all(self):
        reg = make_registry()
        tools = reg.get_tools(tags=None)
        assert len(tools) == 4


class TestToolExecution:
    def test_call_tool(self):
        reg = make_registry()
        result = reg.call_tool("add", {"a": 3, "b": 5})
        assert result == 8

    def test_call_tool_string_result(self):
        reg = make_registry()
        result = reg.call_tool("greet", {"name": "World"})
        assert result == "Hello, World!"


class TestCaching:
    def test_cached_tool_returns_same_result(self):
        reg = make_registry()
        r1 = reg.call_tool("double", {"x": 7})
        r2 = reg.call_tool("double", {"x": 7})
        assert r1 == 14
        assert r2 == 14

    def test_cache_clear(self):
        reg = make_registry()
        reg.call_tool("double", {"x": 3})
        assert len(reg._cache) > 0
        reg.clear_cache()
        assert len(reg._cache) == 0


class TestOpenAISchema:
    def test_schema_structure(self):
        reg = make_registry()
        schema = reg.get_tool("add").to_openai_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "add"
        assert fn["description"] == "Add two numbers."
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert set(params["required"]) == {"a", "b"}

    def test_schema_param_types(self):
        reg = make_registry()
        schema = reg.get_tool("add").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "integer"

    def test_tool_schemas_filtered(self):
        reg = make_registry()
        schemas = reg.tool_schemas(tags=["text"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "greet"

    def test_describe_tools(self):
        reg = make_registry()
        desc = reg.describe_tools(tags=["math"])
        assert "add" in desc
        assert "multiply" in desc
        assert "double" in desc
        # Should NOT include the text tool
        assert "greet" not in desc


def _make_registry_with_context_tool() -> ToolRegistry:
    """Registry with a tool that accepts ActionContext."""
    reg = ToolRegistry()

    @reg.tool(
        description="A leaf tool (no context).",
        tags=["leaf"],
        params={"x": "Input value"},
    )
    def leaf_tool(x: int) -> int:
        return x * 2

    @reg.tool(
        description="A coordination tool (accepts context).",
        tags=["coordination"],
        params={"query": "Search query"},
    )
    def coord_tool(query: str, ctx: ActionContext) -> str:
        ctx.search_state.queries_executed.append(query)
        return f"searched: {query}"

    return reg


class TestActionContextSchemaExclusion:
    def test_context_param_excluded_from_schema(self):
        reg = _make_registry_with_context_tool()
        schema = reg.get_tool("coord_tool").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert "ctx" not in props, "ActionContext should be excluded from schema"

    def test_context_param_excluded_from_required(self):
        reg = _make_registry_with_context_tool()
        schema = reg.get_tool("coord_tool").to_openai_schema()
        required = schema["function"]["parameters"]["required"]
        assert "query" in required
        assert "ctx" not in required

    def test_leaf_tool_schema_unchanged(self):
        reg = _make_registry_with_context_tool()
        schema = reg.get_tool("leaf_tool").to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "x" in props
        assert len(props) == 1


class TestActionContextAutoInjection:
    def test_context_injected_when_present(self):
        reg = _make_registry_with_context_tool()
        ctx = ActionContext()
        result = reg.call_tool("coord_tool", {"query": "test query"}, context=ctx)
        assert result == "searched: test query"
        assert "test query" in ctx.search_state.queries_executed

    def test_context_not_injected_for_leaf_tools(self):
        reg = _make_registry_with_context_tool()
        ctx = ActionContext()
        result = reg.call_tool("leaf_tool", {"x": 5}, context=ctx)
        assert result == 10

    def test_context_none_works_for_all_tools(self):
        reg = _make_registry_with_context_tool()
        # Leaf tool works without context
        result = reg.call_tool("leaf_tool", {"x": 3})
        assert result == 6

    def test_tool_needs_context(self):
        reg = _make_registry_with_context_tool()
        assert reg.tool_needs_context("coord_tool") is True
        assert reg.tool_needs_context("leaf_tool") is False
