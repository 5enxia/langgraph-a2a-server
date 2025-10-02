"""Example of a multi-step agent with tool usage."""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from langgraph_a2a_server import A2AServer


# Define tools
@tool
def get_weather(location: str) -> str:
    """Get the weather for a location.

    Args:
        location: The city name or location to get weather for
    """
    # This is a mock implementation
    return f"The weather in {location} is sunny and 72°F"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate
    """
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


tools = [get_weather, calculate]


# Define the state
class State(TypedDict):
    """State with messages."""

    messages: Annotated[list, add_messages]


# Simple agent that decides whether to use tools or respond
def agent_node(state: State):
    """Agent decision node."""
    messages = state["messages"]
    last_message = messages[-1]

    # Simple rule-based routing (in production, use LLM for this)
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()

        if "weather" in content:
            # Extract location (simple implementation)
            words = content.split()
            location_idx = words.index("in") + 1 if "in" in words else -1
            location = words[location_idx] if location_idx != -1 and location_idx < len(words) else "Unknown"

            # Create tool call
            tool_call = {
                "name": "get_weather",
                "args": {"location": location},
                "id": "weather_call_1",
            }
            return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

        elif any(op in content for op in ["+", "-", "*", "/", "calculate"]):
            # Extract expression
            expression = content.replace("calculate", "").strip()
            tool_call = {
                "name": "calculate",
                "args": {"expression": expression},
                "id": "calc_call_1",
            }
            return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

    # Default response
    return {"messages": [AIMessage(content="I can help you with weather information or calculations. Just ask!")]}


def should_continue(state: State) -> Literal["tools", "end"]:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Set entry point
graph.set_entry_point("agent")

# Add conditional edges
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": "__end__",
    },
)

# Add edge from tools back to agent
graph.add_edge("tools", "agent")

compiled_graph = graph.compile()

# Create and start the A2A server
if __name__ == "__main__":
    from a2a.types import AgentSkill

    server = A2AServer(
        graph=compiled_graph,
        name="Tool-Using Agent",
        description="An agent that can use tools for weather and calculations",
        host="127.0.0.1",
        port=9001,
        version="1.0.0",
        skills=[
            AgentSkill(
                name="weather",
                id="weather",
                description="Get weather information for any location",
                tags=["weather", "tools"],
            ),
            AgentSkill(
                name="calculator",
                id="calc",
                description="Perform mathematical calculations",
                tags=["math", "calculator", "tools"],
            ),
        ],
    )

    print("Starting multi-step tool agent...")
    print("Server URL: http://127.0.0.1:9001")
    print("Agent Card: http://127.0.0.1:9001/.well-known/agent.json")
    print("\nTry asking:")
    print("  - What's the weather in Tokyo?")
    print("  - Calculate 42 * 38")
    server.serve(app_type="fastapi")
