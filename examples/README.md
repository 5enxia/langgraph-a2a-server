# Examples

This directory contains example implementations of LangGraph agents using the A2A protocol.

## Available Examples

### 1. Simple Agent (`simple_agent.py`)

A basic echo agent that demonstrates the minimal setup required.

**Features:**
- Simple message echoing
- Basic LangGraph state management
- FastAPI integration

**Run:**
```bash
python examples/simple_agent.py
```

**Access:**
- Server: http://127.0.0.1:9000
- Agent Card: http://127.0.0.1:9000/.well-known/agent.json
- API Docs: http://127.0.0.1:9000/docs

### 2. LangChain Agent (`langchain_agent.py`)

An AI assistant powered by OpenAI's GPT-4 via LangChain.

**Features:**
- LangChain LLM integration
- Custom agent skills
- System message configuration

**Requirements:**
- `langchain-openai` package
- `OPENAI_API_KEY` environment variable

**Run:**
```bash
export OPENAI_API_KEY=your_api_key_here
python examples/langchain_agent.py
```

**Access:**
- Server: http://127.0.0.1:9000
- Agent Card: http://127.0.0.1:9000/.well-known/agent.json

### 3. Tool Agent (`tool_agent.py`)

A multi-step agent that can use tools for weather information and calculations.

**Features:**
- Tool usage with LangChain tools
- Conditional routing
- Multi-step reasoning

**Run:**
```bash
python examples/tool_agent.py
```

**Access:**
- Server: http://127.0.0.1:9001
- Agent Card: http://127.0.0.1:9001/.well-known/agent.json

**Try these queries:**
- "What's the weather in Tokyo?"
- "Calculate 42 * 38"

## Testing Examples

You can test these agents using curl:

```bash
# Test simple agent
curl -X POST http://127.0.0.1:9000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "parts": [
        {"text": {"text": "Hello, world!"}}
      ]
    }
  }'
```

Or use the A2A client library (if available):

```python
from a2a.client import A2AClient

client = A2AClient("http://127.0.0.1:9000")
response = client.send_message("Hello, world!")
print(response)
```

## Creating Your Own Agent

1. Define your LangGraph state and nodes
2. Compile your graph
3. Create an A2AServer instance
4. Call `server.serve()`

Example:

```python
from langgraph.graph import StateGraph
from langgraph_a2a_server import A2AServer

# Build your graph
graph = StateGraph(YourState)
# ... add nodes and edges ...
compiled_graph = graph.compile()

# Create A2A server
server = A2AServer(
    graph=compiled_graph,
    name="Your Agent",
    description="What your agent does",
)

# Start server
server.serve()
```
