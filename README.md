# LangGraph A2A Server

A2A (Agent-to-Agent) Protocol server implementation for LangGraph agents.

## Overview

This library provides a wrapper for LangGraph agents to be exposed via the A2A protocol, enabling seamless agent-to-agent communication and integration.

## Installation

```bash
pip install langgraph-a2a-server
```

Or with uv:

```bash
uv add langgraph-a2a-server
```

## Usage

```python
from langgraph_a2a_server import A2AServer
from your_langgraph_app import your_graph

# Create an A2A server with your LangGraph agent
server = A2AServer(
    graph=your_graph,
    name="My LangGraph Agent",
    description="An intelligent agent built with LangGraph",
    host="127.0.0.1",
    port=9000,
)

# Start the server
server.serve()
```

## Features

- Easy integration with existing LangGraph applications

## Examples

### simple_agent.py (no llm)

```sh
uv run --extra examples examples/simple_agent.py
```

### langchain_agent.py (with llm)

```sh
uv run --extra examples examples/langchain_agent.py
```

### tools_agent.py (with tools, no llm)

```sh
uv run --extra examples examples/tools_agent.py
```

## License

MIT
