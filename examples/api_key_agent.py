"""Example of a LangGraph agent with custom header authentication.

This example demonstrates how to add authentication to an A2A server using
custom HTTP headers (e.g., API keys).
"""

from typing import Annotated

from a2a.types import AgentCapabilities, AgentCard
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_a2a_server import A2AServer, HeaderAuthContextBuilder


# Define the state
class State(TypedDict):
    """Simple state with messages."""

    messages: Annotated[list, add_messages]


# Define a simple node
def chatbot(state: State):
    """Simple chatbot that echoes the user's message."""
    messages = state["messages"]
    if messages:
        last_message = messages[-1]
        response = f"[Authenticated via API Key] You said: {last_message.content}"
        return {"messages": [{"role": "assistant", "content": response}]}
    return {"messages": [{"role": "assistant", "content": "Hello! How can I help you?"}]}


# Build the graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

compiled_graph = graph.compile()


# Define a header validation function
def validate_api_key(headers: dict[str, str]) -> str | None:
    """Validate API key from custom headers.

    In a real application, you would:
    - Query a database to verify the API key
    - Check rate limits for the API key
    - Verify permissions associated with the key

    Args:
        headers: Dictionary of header names and values

    Returns:
        Username/user ID if the API key is valid, None otherwise
    """
    api_key = headers.get("X-API-Key", "")

    # Simple example: accept specific API keys
    valid_keys = {
        "sk-test-key-123": "company-a@example.com",
        "sk-test-key-456": "company-b@example.com",
        "sk-demo-key": "demo-user@example.com",
    }

    return valid_keys.get(api_key)


# Create and start the A2A server with authentication
if __name__ == "__main__":
    # Create the authentication context builder with specific headers
    context_builder = HeaderAuthContextBuilder(
        validate_api_key,
        header_names=["X-API-Key"],  # Only extract this specific header
    )

    agent_card = AgentCard(
        name="API Key Authenticated Agent",
        description="An agent that requires API key authentication via X-API-Key header",
        url="http://127.0.0.1:9000",
        version="0.0.1",
        skills=[],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
    )

    server = A2AServer(
        graph=compiled_graph,
        agent_card=agent_card,
        host="127.0.0.1",
        port=9000,
        context_builder=context_builder,
    )

    print("Starting A2A server with API Key authentication at http://127.0.0.1:9000")
    print("Agent Card available at: http://127.0.0.1:9000/.well-known/agent.json")
    print("\nValid API keys for testing:")
    print("  - sk-test-key-123 (user: company-a@example.com)")
    print("  - sk-test-key-456 (user: company-b@example.com)")
    print("  - sk-demo-key (user: demo-user@example.com)")
    print("\nTest with curl:")
    print('  curl -H "X-API-Key: sk-demo-key" -H "Content-Type: application/json" \\')
    print('    -d \'{"jsonrpc":"2.0","method":"execute","params":{"input":"Hello"},"id":1}\' \\')
    print("    http://127.0.0.1:9000/jsonrpc")

    server.serve(app_type="fastapi")
