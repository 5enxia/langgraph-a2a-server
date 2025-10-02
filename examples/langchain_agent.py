"""Advanced example with LangChain LLM integration."""

from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_a2a_server import A2AServer


# Define the state
class State(TypedDict):
    """State with messages."""

    messages: Annotated[list, add_messages]


# Initialize the LLM (requires OPENAI_API_KEY environment variable)
llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)


# Define the agent node
def agent_node(state: State):
    """Agent node that uses LLM to generate responses."""
    messages = state["messages"]

    # Add system message if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are a helpful Coding assistant. Be concise and friendly.")] + messages

    # Get response from LLM
    response = llm.invoke(messages)

    return {"messages": [response]}


# Build the graph
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

compiled_graph = graph.compile()

# Create and start the A2A server
if __name__ == "__main__":
    from a2a.types import AgentSkill

    server = A2AServer(
        graph=compiled_graph,
        name="Coding Assistant",
        description="A coding assistant",
        host="127.0.0.1",
        port=9000,
        version="1.0.0",
        skills=[
            AgentSkill(
                name="write_code", id="write_code", description="Generate code snippets", tags=["code", "generation"]
            )
        ],
    )

    print("Starting A2A server with LangChain integration...")
    print("Server URL: http://127.0.0.1:9000")
    print("Agent Card: http://127.0.0.1:9000/.well-known/agent.json")
    print("\nNote: Make sure OPENAI_API_KEY is set in your environment")
    server.serve(app_type="fastapi")
