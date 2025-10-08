"""Example of a LangGraph agent with Bearer token authentication.

This example demonstrates how to add authentication to an A2A server using
Bearer token authentication. The server will validate tokens from the
Authorization header.
"""

from typing import Annotated

from a2a.types import AgentCapabilities, AgentCard
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_a2a_server import A2AServer, BearerTokenAuthContextBuilder


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
        response = f"[Authenticated] You said: {last_message.content}"
        return {"messages": [{"role": "assistant", "content": response}]}
    return {"messages": [{"role": "assistant", "content": "Hello! How can I help you?"}]}


# Build the graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

compiled_graph = graph.compile()


# Define a token validation function
def validate_token(token: str) -> str | None:
    """Validate the bearer token and return username if valid.

    In a real application, you would:
    - Query a database or authentication service
    - Verify JWT token signatures
    - Check token expiration
    - Validate against a list of valid API keys

    Args:
        token: The bearer token from the Authorization header

    Returns:
        Username/user ID if the token is valid, None otherwise
    """
    # Simple example: accept specific tokens
    valid_tokens = {
        "secret-token-123": "user1@example.com",
        "another-token-456": "user2@example.com",
        "demo-token": "demo@example.com",
    }

    return valid_tokens.get(token)


# Create and start the A2A server with authentication
if __name__ == "__main__":
    # Create the authentication context builder
    context_builder = BearerTokenAuthContextBuilder(validate_token)

    agent_card = AgentCard(
        name="Authenticated Echo Agent",
        description="A simple authenticated agent that echoes your messages",
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

    print("Starting A2A server with Bearer token authentication at http://127.0.0.1:9000")
    print("Agent Card available at: http://127.0.0.1:9000/.well-known/agent.json")
    print("\nValid tokens for testing:")
    print("  - secret-token-123 (user: user1@example.com)")
    print("  - another-token-456 (user: user2@example.com)")
    print("  - demo-token (user: demo@example.com)")
    print("\nTest with curl:")
    print('  curl -H "Authorization: Bearer demo-token" -H "Content-Type: application/json" \\')
    print('    -d \'{"jsonrpc":"2.0","method":"execute","params":{"input":"Hello"},"id":1}\' \\')
    print("    http://127.0.0.1:9000/jsonrpc")

    server.serve(app_type="fastapi")
