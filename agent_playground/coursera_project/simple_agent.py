

import json
import time
import traceback
from litellm import completion
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any
import os
from functools import wraps

@dataclass
class Prompt:
    messages: List[Dict] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Fixing mutable default issue


def generate_response(prompt: Prompt) -> str:
    """Call LLM to get response"""

    messages = prompt.messages
    tools = prompt.tools

    result = None

    if not tools:
        response = completion(
            model="github_copilot/gpt-5-mini",
            messages=messages,
            max_tokens=1024
        )
        result = response.choices[0].message.content
    else:
        response = completion(
            model="github_copilot/gpt-5-mini",
            messages=messages,
            tools=tools,
            max_tokens=1024
        )

        if response.choices[0].message.tool_calls:
            tool = response.choices[0].message.tool_calls[0]
            result = {
                "tool": tool.function.name,
                "args": json.loads(tool.function.arguments),
            }
            result = json.dumps(result)
        else:
            result = response.choices[0].message.content


    return result


@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str


class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)


class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> [Action, None]:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())


class Memory:
    def __init__(self):
        self.items = []  # Basic conversation histor

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory


class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }


class AgentLanguage:
    def __init__(self):
        pass

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")


    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")



class AgentFunctionCallingActionLanguage(AgentLanguage):

    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]) -> List:
        # Map all goals to a single string that concatenates their description
        # and combine into a single message of type system
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join([f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals])
        return [
            {"role": "system", "content": goal_instructions}
        ]

    def format_memory(self, memory: Memory) -> List:
        """Generate response from language model"""
        # Map all environment results to a role:user messages
        # Map all assistant messages to a role:assistant messages
        # Map all user messages to a role:user messages
        items = memory.get_memories()
        mapped_items = []
        for item in items:

            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> [List,List]:
        """Generate response from language model"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    # Include up to 1024 characters of the description
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            } for action in actions
        ]

        return tools

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:

        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(self,
                                         prompt: Prompt,
                                         response: str,
                                         traceback: str,
                                         error: Any,
                                         retries_left: int) -> Prompt:

        return prompt

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format by extracting the ```json block"""

        try:
            return json.loads(response)

        except Exception as e:
            return {
                "tool": "terminate",
                "args": {"message":response}
            }


class Agent:
    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment):
        """
        Initialize an agent with its core GAME components
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment

    def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry) -> Prompt:
        """Build prompt with memory context"""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory
        )

    def get_action(self, response):
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """
        Update memory with the agent's decision and the environment's response.
        """
        new_memories = [
            {"type": "assistant", "content": response},
            {"type": "environment", "content": json.dumps(result)}
        ]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        response = self.generate_response(full_prompt)
        return response

    def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
        """
        Execute the GAME loop for this agent with a maximum iteration limit.
        """
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        for _ in range(max_iterations):
            # Construct a prompt that includes the Goals, Actions, and the current Memory
            prompt = self.construct_prompt(self.goals, memory, self.actions)

            print("Agent thinking...")
            # Generate a response from the agent
            response = self.prompt_llm_for_action(prompt)
            print(f"Agent Decision: {response}")

            # Determine which action the agent wants to execute
            action, invocation = self.get_action(response)

            # Execute the action in the environment
            result = self.environment.execute_action(action, invocation["args"])
            print(f"Action Result: {result}")

            # Update the agent's memory with information about what happened
            self.update_memory(memory, response, result)

            # Check if the agent has decided to terminate
            if self.should_terminate(response):
                break

        return memory

class ToolRegistry:
    def __init__(self):
        self.tools = []

    def register_tool(self, action: Action, tags: List[str]):
        """Register a tool with associated tags."""
        self.tools.append({"action": action, "tags": tags})

    def get_tools_by_tags(self, tags: List[str]) -> List[Action]:
        """Retrieve tools that match any of the given tags."""
        return [
            tool["action"]
            for tool in self.tools
            if any(tag in tool["tags"] for tag in tags)
        ]

    def get_all_tools(self) -> List[Action]:
        """Retrieve all registered tools."""
        return [tool["action"] for tool in self.tools]
    
tool_registry = ToolRegistry()  # Global instance of ToolRegistry

def tool(name: str, description: str, parameters: dict, tags: List[str], terminal: bool = False):
    """
    A decorator to define and register tools in the ToolRegistry.

    Args:
        name (str): The name of the tool.
        description (str): A brief description of what the tool does.
        parameters (dict): A dictionary defining the tool's parameters.
        tags (List[str]): Tags to categorize the tool.
        terminal (bool): Whether this tool terminates the agent's workflow.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Create an Action object for the tool
        action = Action(
            name=name,
            function=wrapper,
            description=description,
            parameters=parameters,
            terminal=terminal
        )

        # Register the tool in the ToolRegistry
        tool_registry.register_tool(action, tags)

        return wrapper
    return decorator

@tool(
    name="list_project_files",
    description="Lists all files in the project.",
    parameters={},
    tags=["file_operations"],
    terminal=False
)
def list_project_files() -> List[str]:
    """Lists all .py files in the current directory."""
    return sorted([file for file in os.listdir(".") if file.endswith(".py")])

@tool(
        name="list_project_files_recursive",
        description="Lists all files in the project recursively.",
        parameters={},
        tags=["file_operations"],
        terminal=False
)
def list_project_files_recursive() -> List[str]:
    """Lists all .py files in the current directory and subdirectories."""
    project_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                project_files.append(os.path.join(root, file))
    return sorted(project_files)


@tool(
    name="read_project_file",
    description="Reads a file from the project.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    },
    tags=["file_operations"],
    terminal=False
)
def read_project_file(name: str) -> str:
    with open(name, "r") as f:
        return f.read()

@tool(
        name="terminate",
        description="Terminates the session and prints the message to the user.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}
            },
            "required": []
        },
        tags=["control"],
        terminal=True
)
def terminate(message: str) -> str:
    return f"{message}\nTerminating..."


# Creating an action registry and registering the decorated tools
def create_action_registry_from_tools(tags: List[str]) -> ActionRegistry:
    """Create an ActionRegistry and register tools that match the given tags."""
    action_registry = ActionRegistry()
    for action in tool_registry.get_tools_by_tags(tags):
        action_registry.register(action)
    return action_registry

def main():
    # Define the agent's goals
    goals = [
        Goal(priority=1, name="Gather Information", description="Read each file in the project"),
        Goal(priority=1, name="Terminate", description="Call the terminate call when you have read all the files "
                                                       "and provide the content of the README in the terminate message")
    ]

    # Define the agent's language
    agent_language = AgentFunctionCallingActionLanguage()

    # Define the environment
    environment = Environment()

    # Create an agent instance
    agent = Agent(goals=goals, 
                  agent_language=agent_language,
                  action_registry=create_action_registry_from_tools(["file_operations", "control"]),
                  generate_response=generate_response, 
                  environment=environment)

    # Run the agent with user input
    user_input = "Write a README for this project."
    final_memory = agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())


if __name__ == "__main__":
    main()