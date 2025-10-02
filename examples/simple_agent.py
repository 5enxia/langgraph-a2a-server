"""Simple example of a LangGraph agent exposed via A2A protocol."""

from typing import Annotated

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_a2a_server import A2AServer


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
        response = f"You said: {last_message.content}"
        return {"messages": [{"role": "assistant", "content": response}]}
    return {"messages": [{"role": "assistant", "content": "Hello! How can I help you?"}]}


# Build the graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

compiled_graph = graph.compile()

# Create and start the A2A server
if __name__ == "__main__":
    server = A2AServer(
        graph=compiled_graph,
        name="Simple Echo Agent",
        description="A simple agent that echoes your messages",
        host="127.0.0.1",
        port=9000,
    )

    print("Starting A2A server at http://127.0.0.1:9000")
    print("Agent Card available at: http://127.0.0.1:9000/.well-known/agent.json")
    server.serve(app_type="fastapi")
